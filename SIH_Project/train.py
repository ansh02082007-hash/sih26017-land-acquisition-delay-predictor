"""Training pipeline for Land Acquisition Delay Prediction Dual-Head Neural Network.

This module orchestrates data loading, stratified train/val/test splitting, FeatureProcessor
fitting, Wide & Deep dual-head model instantiation, AdamW + CosineAnnealingWarmRestarts training,
early stopping on validation combined loss, and full epoch metric logging to console and CSV.
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

from config import TrainingConfig, ModelConfig
from preprocessing import FeatureProcessor, ProcessedBatch
from model import DelayPredictionNet, ModelOutput
from losses import MultiTaskDelayLoss

logger = logging.getLogger("train")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TabularBatchDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed batch tensors for mini-batch iteration."""

    def __init__(self, batch: ProcessedBatch) -> None:
        """Store tensors from ProcessedBatch."""
        self.embedding_indices = batch.embedding_indices
        self.one_hot_features = batch.one_hot_features
        self.numerical_features = batch.numerical_features
        self.target_prob = batch.target_prob
        self.target_months = batch.target_months
        self.case_ids = batch.case_ids or ["" for _ in range(len(batch.one_hot_features))]

    def __len__(self) -> int:
        """Return total number of records."""
        return self.numerical_features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch a single record by index."""
        item = {
            "one_hot": self.one_hot_features[idx],
            "numericals": self.numerical_features[idx],
        }
        for col, tensor in self.embedding_indices.items():
            item[f"emb_{col}"] = tensor[idx]

        if self.target_prob is not None:
            item["target_prob"] = self.target_prob[idx]
        if self.target_months is not None:
            item["target_months"] = self.target_months[idx]
        return item


def collate_tabular_batch(items: list) -> ProcessedBatch:
    """Collate individual sample dictionaries into a unified ProcessedBatch."""
    one_hot = torch.stack([item["one_hot"] for item in items], dim=0)
    numericals = torch.stack([item["numericals"] for item in items], dim=0)

    emb_keys = [k for k in items[0].keys() if k.startswith("emb_")]
    embedding_indices = {}
    for k in emb_keys:
        col_name = k.replace("emb_", "")
        embedding_indices[col_name] = torch.stack([item[k] for item in items], dim=0)

    target_prob = None
    if "target_prob" in items[0]:
        target_prob = torch.stack([item["target_prob"] for item in items], dim=0)

    target_months = None
    if "target_months" in items[0]:
        target_months = torch.stack([item["target_months"] for item in items], dim=0)

    return ProcessedBatch(
        embedding_indices=embedding_indices,
        one_hot_features=one_hot,
        numerical_features=numericals,
        target_prob=target_prob,
        target_months=target_months,
    )


def compute_metrics(
    y_prob_true: np.ndarray,
    y_prob_pred: np.ndarray,
    y_months_true: np.ndarray,
    y_months_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute classification and regression evaluation metrics."""
    metrics: Dict[str, float] = {}

    # Classification Metrics across all cases
    binary_preds = (y_prob_pred >= 0.5).astype(int)
    y_prob_true_int = y_prob_true.astype(int)

    try:
        metrics["auroc"] = float(roc_auc_score(y_prob_true_int, y_prob_pred))
    except Exception:
        metrics["auroc"] = 0.5

    metrics["f1"] = float(f1_score(y_prob_true_int, binary_preds, zero_division=0))
    metrics["precision"] = float(precision_score(y_prob_true_int, binary_preds, zero_division=0))
    metrics["recall"] = float(recall_score(y_prob_true_int, binary_preds, zero_division=0))

    # Regression Metrics evaluated STRICTLY on ground truth delayed cases
    delayed_mask = (y_prob_true_int == 1).flatten()
    if delayed_mask.sum() > 0:
        true_m = y_months_true[delayed_mask]
        pred_m = y_months_pred[delayed_mask]
        metrics["mae_delayed"] = float(mean_absolute_error(true_m, pred_m))
        metrics["rmse_delayed"] = float(root_mean_squared_error(true_m, pred_m))
    else:
        metrics["mae_delayed"] = 0.0
        metrics["rmse_delayed"] = 0.0

    return metrics


def train_epoch(
    model: DelayPredictionNet,
    dataloader: DataLoader,
    criterion: MultiTaskDelayLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Execute one training epoch over the dataset."""
    model.train()
    total_loss_sum = 0.0
    focal_loss_sum = 0.0
    huber_loss_sum = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        output: ModelOutput = model(batch)
        tot_loss, foc_loss, hub_loss = criterion(
            pred_prob=output.delay_probability,
            pred_months=output.delay_months,
            target_prob=batch.target_prob,
            target_months=batch.target_months,
        )

        tot_loss.backward()
        # Gradient clipping prevents gradient explosion during early warm restarts
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss_sum += tot_loss.item()
        focal_loss_sum += foc_loss.item()
        huber_loss_sum += hub_loss.item()
        n_batches += 1

    return (
        total_loss_sum / max(1, n_batches),
        focal_loss_sum / max(1, n_batches),
        huber_loss_sum / max(1, n_batches),
    )


@torch.no_grad()
def evaluate(
    model: DelayPredictionNet,
    dataloader: DataLoader,
    criterion: MultiTaskDelayLoss,
    device: torch.device,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Evaluate model on validation or test dataset."""
    model.eval()
    total_loss_sum = 0.0
    focal_loss_sum = 0.0
    huber_loss_sum = 0.0
    n_batches = 0

    all_prob_preds = []
    all_prob_trues = []
    all_month_preds = []
    all_month_trues = []

    for batch in dataloader:
        batch = batch.to(device)
        output: ModelOutput = model(batch)
        tot_loss, foc_loss, hub_loss = criterion(
            pred_prob=output.delay_probability,
            pred_months=output.delay_months,
            target_prob=batch.target_prob,
            target_months=batch.target_months,
        )

        total_loss_sum += tot_loss.item()
        focal_loss_sum += foc_loss.item()
        huber_loss_sum += hub_loss.item()
        n_batches += 1

        all_prob_preds.append(output.delay_probability.cpu().numpy())
        all_prob_trues.append(batch.target_prob.cpu().numpy())
        all_month_preds.append(output.delay_months.cpu().numpy())
        all_month_trues.append(batch.target_months.cpu().numpy())

    y_prob_pred = np.vstack(all_prob_preds)
    y_prob_true = np.vstack(all_prob_trues)
    y_month_pred = np.vstack(all_month_preds)
    y_month_true = np.vstack(all_month_trues)

    metrics = compute_metrics(y_prob_true, y_prob_pred, y_month_true, y_month_pred)
    return (
        total_loss_sum / max(1, n_batches),
        focal_loss_sum / max(1, n_batches),
        huber_loss_sum / max(1, n_batches),
        metrics,
    )


def train_model(
    config: Optional[TrainingConfig] = None,
    model_config: Optional[ModelConfig] = None,
) -> Tuple[DelayPredictionNet, FeatureProcessor, pd.DataFrame]:
    """Execute end-to-end training, early stopping, and persistence."""
    cfg = config or TrainingConfig()
    mcfg = model_config or ModelConfig()

    # Deterministic seeding for reproducible benchmarking
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Load Dataset
    data_file = Path(cfg.data_path)
    if not data_file.exists():
        logger.info(f"Dataset not found at {data_file}. Generating synthetic dataset...")
        from data.synthetic_generator import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=5000, seed=cfg.seed)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)
    logger.info(f"Loaded dataset with {len(df)} records from {data_file}")

    # 2. Stratified Train / Val / Test Split
    # Preserves the exact delayed-to-nondelayed ratio across partitions
    train_val_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=df["delay_probability"],
    )
    val_rel_size = cfg.val_size / (1.0 - cfg.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_size,
        random_state=cfg.seed,
        stratify=train_val_df["delay_probability"],
    )
    logger.info(f"Data partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 3. Fit FeatureProcessor on Train Set Only (Prevents Data Leakage)
    processor = FeatureProcessor(embedding_dim_max=mcfg.embedding_dim_max)
    train_batch = processor.fit_transform(train_df)
    val_batch = processor.transform(val_df)
    test_batch = processor.transform(test_df)

    # 4. Create PyTorch DataLoaders
    train_loader = DataLoader(
        TabularBatchDataset(train_batch),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_tabular_batch,
    )
    val_loader = DataLoader(
        TabularBatchDataset(val_batch),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_tabular_batch,
    )
    test_loader = DataLoader(
        TabularBatchDataset(test_batch),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_tabular_batch,
    )

    # 5. Instantiate Model and Loss
    model = DelayPredictionNet(
        vocab_sizes=processor.vocab_sizes_,
        embedding_dims=processor.embedding_dims_,
        num_one_hot_features=processor.num_one_hot_features_,
        num_numerical_features=processor.num_numerical_features_,
        config=mcfg,
    ).to(device)

    criterion = MultiTaskDelayLoss(
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        huber_beta=cfg.huber_beta,
        lambda_focal=cfg.lambda_focal,
        lambda_huber=cfg.lambda_huber,
    ).to(device)

    # AdamW decouples weight decay from gradient updates for clean L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Cosine Annealing with Warm Restarts escapes sharp local minima on tabular surfaces
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.scheduler_t0,
        T_mult=cfg.scheduler_tmult,
        eta_min=cfg.scheduler_eta_min,
    )

    # 6. Initialize Training Log CSV
    log_path = Path(cfg.training_log_path)
    log_headers = [
        "epoch", "train_loss", "train_focal", "train_huber",
        "val_loss", "val_focal", "val_huber",
        "val_auroc", "val_f1", "val_precision", "val_recall",
        "val_mae_delayed", "val_rmse_delayed", "lr"
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(log_headers)

    # 7. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    log_records = []

    logger.info("Starting training loop...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_foc, train_hub = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_foc, val_hub, val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log to list and CSV
        row = [
            epoch, train_loss, train_foc, train_hub,
            val_loss, val_foc, val_hub,
            val_metrics["auroc"], val_metrics["f1"],
            val_metrics["precision"], val_metrics["recall"],
            val_metrics["mae_delayed"], val_metrics["rmse_delayed"],
            current_lr,
        ]
        log_records.append(row)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        logger.info(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"AUROC: {val_metrics['auroc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"MAE (Delayed): {val_metrics['mae_delayed']:.2f}m | LR: {current_lr:.6f}"
        )

        # Early stopping tracking validation combined loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save checkpoint
            torch.save(best_model_state, cfg.model_save_path)
            processor.save(cfg.processor_save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}! "
                    f"Validation loss did not improve for {cfg.early_stopping_patience} consecutive epochs."
                )
                break

    # 8. Restore Best Model and Evaluate on Test Set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, _, _, test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    logger.info("=" * 60)
    logger.info("FINAL TEST SET PERFORMANCE (Unseen Generalization):")
    logger.info(f"Test Combined Loss: {test_loss:.4f}")
    logger.info(f"Test AUROC:         {test_metrics['auroc']:.4f}")
    logger.info(f"Test F1 Score:      {test_metrics['f1']:.4f}")
    logger.info(f"Test Precision:     {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall:        {test_metrics['recall']:.4f}")
    logger.info(f"Test MAE (Delayed): {test_metrics['mae_delayed']:.2f} months")
    logger.info(f"Test RMSE(Delayed): {test_metrics['rmse_delayed']:.2f} months")
    logger.info("=" * 60)

    log_df = pd.DataFrame(log_records, columns=log_headers)
    return model, processor, log_df


def main() -> None:
    """CLI entry point supporting python train.py --config config.py."""
    parser = argparse.ArgumentParser(description="Train Land Acquisition Delay Neural Network.")
    parser.add_argument("--config", type=str, default="config.py", help="Path to config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr

    train_model(config=cfg)


if __name__ == "__main__":
    main()
