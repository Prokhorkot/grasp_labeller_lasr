import torch
import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC


class GraspLitModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight])
        else:
            pos_weight_tensor = None

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

    def forward(self, sample):
        return self.model(sample)

    def training_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)

        self.train_acc(probs, labels.int())
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)

        labels_int = labels.int()
        self.val_acc(probs, labels_int)
        self.val_f1(probs, labels_int)
        self.val_auroc(probs, labels_int)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/f1", self.val_f1, prog_bar=True)
        self.log("val/auroc", self.val_auroc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)

        labels_int = labels.int()
        self.test_acc(probs, labels_int)
        self.test_f1(probs, labels_int)
        self.test_auroc(probs, labels_int)

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)
        self.log("test/f1", self.test_f1)
        self.log("test/auroc", self.test_auroc)

    def _shared_step(self, batch):
        sample, labels = batch

        logits = self.model(sample).squeeze(-1)
        labels = labels.float().to(logits.device)

        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        return loss, probs, labels

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
