"""
监督学习训练器 — fine-tune 模式 + 离线评估。

核心设计:
- Fine-tune: optimizer 跨迭代保持
- Loss: MSE（预测动作 vs 真实动作）
- 梯度裁剪: max_grad_norm=1.0
- 正则化: dropout + weight_decay (AdamW)
- 验证集: 10% held out
- Early stopping: val loss 连续 N epoch 不降则停止
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

if TYPE_CHECKING:
    from .config import TrainConfig
    from .dataset import TrajectoryDataset
    from .model import ActionPredictor

logger = logging.getLogger(__name__)


class Trainer:
    """监督学习训练器，支持跨迭代 fine-tune。"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.optimizer: AdamW | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        model: ActionPredictor,
        dataset: TrajectoryDataset,
        config: TrainConfig,
    ) -> dict[str, float]:
        """
        训练一轮（多 epoch），返回 train/val loss。

        模型在 GPU/CPU 上训练，optimizer state 跨迭代保持。
        """
        model = model.to(self.device)
        logger.info("训练设备: %s", self.device)

        if len(dataset) == 0:
            logger.warning("数据集为空，跳过训练")
            return {"train_loss": float("inf"), "val_loss": float("inf")}

        # 划分 train/val
        val_size = max(1, int(len(dataset) * config.val_ratio))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        use_cuda = self.device.type == "cuda"
        nw = config.num_workers

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=nw,
            pin_memory=use_cuda,
            persistent_workers=nw > 0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            drop_last=False,
            num_workers=nw,
            pin_memory=use_cuda,
            persistent_workers=nw > 0,
        )

        if self.optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0
        avg_train = float("inf")

        for epoch in range(config.epochs):
            # ---- Train ----
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            for dynamics, patches, actions, valid_mask, target in train_loader:
                dynamics = dynamics.to(self.device)
                patches = patches.to(self.device)
                actions = actions.to(self.device)
                valid_mask = valid_mask.to(self.device)
                target = target.to(self.device)

                pred = model(dynamics, patches, actions, valid_mask=valid_mask)
                loss = F.mse_loss(pred, target)

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                self.optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            # ---- Validate ----
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for dynamics, patches, actions, valid_mask, target in val_loader:
                    dynamics = dynamics.to(self.device)
                    patches = patches.to(self.device)
                    actions = actions.to(self.device)
                    valid_mask = valid_mask.to(self.device)
                    target = target.to(self.device)

                    pred = model(dynamics, patches, actions, valid_mask=valid_mask)
                    val_loss_sum += F.mse_loss(pred, target).item()
                    val_batches += 1

            avg_train = train_loss_sum / max(train_batches, 1)
            avg_val = val_loss_sum / max(val_batches, 1)
            logger.info(
                "Epoch %d/%d: train_loss=%.6f  val_loss=%.6f",
                epoch + 1, config.epochs, avg_train, avg_val,
            )

            # ---- Early Stopping ----
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    logger.info(
                        "Early stopping: val loss 连续 %d epoch 未下降",
                        patience_counter,
                    )
                    break

        if best_weights is not None:
            model.load_state_dict(best_weights)

        return {"train_loss": avg_train, "val_loss": best_val_loss}
