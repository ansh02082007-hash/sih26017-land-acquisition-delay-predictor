"""Dual-Head Neural Network Architecture for Land Acquisition Delay Prediction.

This module implements the DelayPredictionNet model, combining a Wide & Deep paradigm
with multi-task dual heads for simultaneous delay classification and delay duration regression.
Every architectural decision is explicitly justified with inline comments for presentation to
hackathon evaluators and Ministry of Rural Development stakeholders.
"""

from typing import Dict, NamedTuple, Optional, Union
import torch
import torch.nn as nn
from config import ModelConfig
from preprocessing import ProcessedBatch


class ModelOutput(NamedTuple):
    """Container holding forward pass predictions and intermediate representations.

    Attributes:
        delay_probability: Tensor of shape (B, 1) in range [0, 1] representing the probability
            that a land acquisition case will experience statutory schedule delays.
        delay_months: Tensor of shape (B, 1) non-negative values representing predicted delay
            duration in months (meaningful when delay_probability is high).
        fused_representation: Tensor of shape (B, fused_dim) representing the shared multi-task
            latent bottleneck vector (used for SHAP attribution and downstream downstream hooks).
        deep_features: Tensor of shape (B, 16) from the bottleneck of the deep neural tower.
        wide_features: Tensor of shape (B, wide_output_dim) from the linear memorization path.
    """

    delay_probability: torch.Tensor
    delay_months: torch.Tensor
    fused_representation: torch.Tensor
    deep_features: torch.Tensor
    wide_features: torch.Tensor


class DelayPredictionNet(nn.Module):
    """Wide & Deep Dual-Head Neural Network for tabular land acquisition delay prediction.

    Architecture summary:
    1. Entity Embeddings for high-cardinality geographic codes (PIN, District).
    2. Embedding Dropout (0.1) regularizing spatial lookup tables.
    3. Wide component: Direct linear memorization from raw binary one-hot & continuous features.
    4. Deep component: Non-linear representation learning via 3 tapering layers (64 -> 32 -> 16).
    5. Fusion: Concatenation of wide memorization + deep generalization into shared bottleneck.
    6. Dual Heads: Head A (Classification via Sigmoid) & Head B (Regression via Softplus).
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embedding_dims: Dict[str, int],
        num_one_hot_features: int,
        num_numerical_features: int,
        config: Optional[ModelConfig] = None,
    ) -> None:
        """Initialize the Wide & Deep network layers and parameter initializations.

        Args:
            vocab_sizes: Dictionary mapping categorical column name to vocabulary cardinality.
            embedding_dims: Dictionary mapping categorical column name to embedding dimension.
            num_one_hot_features: Number of binary one-hot encoded features.
            num_numerical_features: Number of continuous scaled features.
            config: ModelConfig containing layer dimensions, dropouts, and batchnorm momentum.
        """
        super().__init__()
        self.config = config or ModelConfig()
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims
        self.num_one_hot = num_one_hot_features
        self.num_numerical = num_numerical_features

        # 1. High-Cardinality Entity Embeddings
        # Dense embeddings project discrete PINs/districts into a continuous latent geometry where
        # geographically or socio-economically proximate administrative regions cluster together.
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(
                num_embeddings=vocab_sizes[col],
                embedding_dim=embedding_dims[col],
                padding_idx=0,  # Index 0 is reserved for unknown/unseen categories (<UNK>)
            )
            for col in vocab_sizes
        })

        total_embedding_dim = sum(embedding_dims.values())

        # Embedding Dropout (0.1) prevents the network from over-relying on single specific PIN codes,
        # forcing the deep tower to learn robust interactions with general case attributes.
        self.embedding_dropout = nn.Dropout(p=self.config.embedding_dropout)

        # 2. Wide Component (Memorization Path)
        # Directly connects raw one-hot indicators and numericals to the fusion layer via linear mapping,
        # capturing deterministic linear interactions without distortion from deep non-linearities.
        wide_input_dim = num_one_hot_features + num_numerical_features
        self.wide_linear = nn.Linear(wide_input_dim, self.config.wide_output_dim)

        # 3. Deep Component (Generalization Path)
        # Deep dense layers learn high-order non-linear combinations across spatial embeddings,
        # RFCTLARR statutory stages, financial compensation scale, and demographic metrics.
        deep_input_dim = total_embedding_dim + num_one_hot_features + num_numerical_features
        h1, h2, h3 = self.config.hidden_dims
        d1, d2, d3 = self.config.dropouts

        # Layer 1: 64 units + BatchNorm + GELU + Dropout(0.3)
        self.deep_fc1 = nn.Linear(deep_input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1, momentum=self.config.batch_norm_momentum)
        self.act1 = nn.GELU()  # GELU provides smooth non-zero gradients in the negative regime unlike ReLU
        self.drop1 = nn.Dropout(p=d1)

        # Layer 2: 32 units + BatchNorm + GELU + Dropout(0.2)
        self.deep_fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2, momentum=self.config.batch_norm_momentum)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(p=d2)

        # Layer 3: 16 units + BatchNorm + GELU + Dropout(0.1)
        # Tapering dropouts (0.3 -> 0.2 -> 0.1) protect representations as layer capacity narrows.
        self.deep_fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3, momentum=self.config.batch_norm_momentum)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(p=d3)

        # 4. Multi-Task Shared Representation Fusion
        # Fusing the 16-dim deep feature abstraction with the 16-dim wide memorization feature vector
        # yields a balanced 32-dimensional multi-task bottleneck.
        fused_dim = self.config.wide_output_dim + h3

        # 5. Dual Output Heads
        # Multi-task sharing forces the representation to capture features predictive of BOTH delay
        # likelihood and severity, preventing overfitting compared to two isolated models.

        # Head A: Delay Probability (Binary Classification)
        # Linear -> 1 unit -> Sigmoid bounds output strictly to valid probability range [0, 1].
        self.head_classifier = nn.Linear(fused_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Head B: Delay Duration in Months (Continuous Regression)
        # Linear -> 1 unit -> Softplus guarantees strictly non-negative delay months while preserving
        # smooth gradients across the entire positive domain (avoiding dead neurons from ReLU).
        self.head_regressor = nn.Linear(fused_dim, 1)
        self.softplus = nn.Softplus()

        # Initialize weights with Kaiming Normal (GELU/ReLU family) and Xavier for output heads
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights with Kaiming normal for hidden layers and Xavier for heads.

        Kaiming normal preserves gradient variance across GELU activations; Xavier ensures
        stable initial probability logits and regression scales prior to sigmoid/softplus.
        """
        for m in [self.wide_linear, self.deep_fc1, self.deep_fc2, self.deep_fc3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for head in [self.head_classifier, self.head_regressor]:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def freeze_early_layers(self) -> None:
        """Freeze embedding layers and Deep Layer 1 for Human-In-The-Loop continual learning.

        Freezing early generic layers during incremental retraining prevents small new batches
        from corrupting global spatial embedding tables or basic feature relationships.
        """
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.deep_fc1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self) -> None:
        """Restore gradient tracking to all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        batch_or_embeddings: Union[ProcessedBatch, Dict[str, torch.Tensor]],
        one_hot: Optional[torch.Tensor] = None,
        numericals: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Execute forward computation across wide, deep, and dual head branches.

        Args:
            batch_or_embeddings: Either a ProcessedBatch instance or a dictionary of embedding index tensors.
            one_hot: Tensor of one-hot features (required if batch_or_embeddings is a dict).
            numericals: Tensor of scaled numerical features (required if batch_or_embeddings is a dict).

        Returns:
            ModelOutput containing delay_probability, delay_months, and intermediate latent vectors.
        """
        # Unpack arguments from ProcessedBatch if provided
        if isinstance(batch_or_embeddings, ProcessedBatch):
            embedding_dict = batch_or_embeddings.embedding_indices
            one_hot_tensor = batch_or_embeddings.one_hot_features
            num_tensor = batch_or_embeddings.numerical_features
        else:
            embedding_dict = batch_or_embeddings
            if one_hot is None or numericals is None:
                raise ValueError("one_hot and numericals tensors are required when passing an embedding dict.")
            one_hot_tensor = one_hot
            num_tensor = numericals

        # 1. Lookup and Concatenate Entity Embeddings
        embed_list = []
        for col, emb_layer in self.embeddings.items():
            col_indices = embedding_dict[col]
            embed_list.append(emb_layer(col_indices))
        concat_embeddings = torch.cat(embed_list, dim=1)
        dropped_embeddings = self.embedding_dropout(concat_embeddings)

        # 2. Wide Path: Direct memorization of simple linear patterns
        wide_input = torch.cat([one_hot_tensor, num_tensor], dim=1)
        wide_features = self.wide_linear(wide_input)

        # 3. Deep Path: High-order feature interactions
        deep_input = torch.cat([dropped_embeddings, one_hot_tensor, num_tensor], dim=1)

        # PHASE 2 HOOK: TabNet-style attention
        # In Phase 2, a sequential sparse attention masking mechanism (sparsemax/entmax)
        # can be inserted here before the deep representation to provide instance-wise
        # dynamic feature selection and interpretability directly within the forward pass.

        x_deep = self.deep_fc1(deep_input)
        # Handle BatchNorm edge case when batch size is 1 during single-case inference
        if x_deep.shape[0] > 1 or not self.training:
            x_deep = self.bn1(x_deep)
        x_deep = self.drop1(self.act1(x_deep))

        x_deep = self.deep_fc2(x_deep)
        if x_deep.shape[0] > 1 or not self.training:
            x_deep = self.bn2(x_deep)
        x_deep = self.drop2(self.act2(x_deep))

        x_deep = self.deep_fc3(x_deep)
        if x_deep.shape[0] > 1 or not self.training:
            x_deep = self.bn3(x_deep)
        deep_features = self.drop3(self.act3(x_deep))

        # 4. Fusion Layer
        fused_representation = torch.cat([wide_features, deep_features], dim=1)

        # 5. Dual Heads
        prob_logits = self.head_classifier(fused_representation)
        delay_probability = self.sigmoid(prob_logits)

        month_logits = self.head_regressor(fused_representation)
        delay_months = self.softplus(month_logits)

        return ModelOutput(
            delay_probability=delay_probability,
            delay_months=delay_months,
            fused_representation=fused_representation,
            deep_features=deep_features,
            wide_features=wide_features,
        )
