from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
import lightning as L
from hydra.utils import get_method, instantiate, to_absolute_path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from grasp_labeller_lasr.models.heads import MLPClassifierHead
from grasp_labeller_lasr.models.sparsh_classifier import SparshGraspClassifier
from grasp_labeller_lasr.training.datamodule import GraspDataModule
from grasp_labeller_lasr.training.lightning_module import GraspLitModule


from dotenv import load_dotenv
load_dotenv()


LOGGER = logging.getLogger(__name__)


def _setup_file_logger(log_path: Path = Path("training.log")) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def _clean_run_name_part(value: object) -> str:
    text = str(value)
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in text)
    return cleaned.strip("-")


def _build_run_name(cfg: DictConfig, tags: dict[str, str]) -> str | None:
    if cfg.logger.run_name not in (None, "null", ""):
        return cfg.logger.run_name

    parts = [
        _clean_run_name_part(tags[tag])
        for tag in cfg.logger.run_name_tags
        if tag in tags
    ]
    return "-".join(part for part in parts if part) or None


def _build_mlflow_tags(cfg: DictConfig) -> dict[str, str]:
    tags = OmegaConf.to_container(cfg.logger.tags, resolve=True)
    if not isinstance(tags, dict):
        raise ValueError("cfg.logger.tags must be a mapping.")
    return {str(key): str(value) for key, value in tags.items()}


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    _setup_file_logger()
    torch.set_float32_matmul_precision("high")
    datamodule = GraspDataModule(
        dataset_root=Path(to_absolute_path(cfg.data.dataset_root)),
        frames_to_stack=cfg.data.frames_to_stack,
        downsampling_size=cfg.data.downsampling_size,
        crop_size=cfg.data.crop_size,
        finger_names=tuple(cfg.data.finger_names),
        label_method=cfg.data.label_method,
        n_aug_copies=cfg.data.n_aug_copies,
        include_original=cfg.data.include_original,
        color_jitter_brightness=cfg.data.color_jitter_brightness,
        remove_background=cfg.data.remove_background,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        split_seed=cfg.data.split_seed,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        cache_enabled=cfg.data.cache_enabled,
        cache_dir=cfg.data.cache_dir,
    )

    patch_pooling = get_method(cfg.head.patch_pooling)
    patch_pooler = (
        instantiate(cfg.head.patch_pooler)
        if cfg.head.patch_pooler is not None
        else None
    )
    head = MLPClassifierHead(
        num_inputs=cfg.head.num_inputs,
        embedding_dim=cfg.head.embedding_dim,
        hidden_dim=cfg.head.hidden_dim,
        dropout=cfg.head.dropout,
        output_dim=cfg.head.output_dim,
        patch_pooling=patch_pooling,
        patch_pooler=patch_pooler,
    )

    model = SparshGraspClassifier(
        encoder_name=cfg.model.encoder_name,
        encoder_path=cfg.model.encoder_path,
        device=cfg.model.device,
        head=head,
        image_size=tuple(cfg.model.image_size),
        finger_names=tuple(cfg.data.finger_names),
    )

    lit_model = GraspLitModule(
        model=model,
        lr=cfg.lightning.learning_rate,
        weight_decay=cfg.lightning.weight_decay,
        pos_weight=cfg.lightning.pos_weight,
        lr_scheduler_cfg=cfg.lightning.lr_scheduler,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode,
        save_top_k=cfg.callbacks.checkpoint.save_top_k,
        filename=cfg.callbacks.checkpoint.filename,
    )
    early_stopping = EarlyStopping(
        monitor=cfg.callbacks.early_stopping.monitor,
        mode=cfg.callbacks.early_stopping.mode,
        patience=cfg.callbacks.early_stopping.patience,
    )
    callbacks = [checkpoint_callback, early_stopping]
    if cfg.callbacks.learning_rate_monitor.enabled:
        callbacks.append(instantiate(cfg.callbacks.learning_rate_monitor.callback))
    logger = None
    if cfg.logger.enabled:
        tracking_uri = cfg.logger.tracking_uri
        if tracking_uri == "null":
            tracking_uri = None
        tags = _build_mlflow_tags(cfg)
        run_name = _build_run_name(cfg, tags)

        logger = MLFlowLogger(
            experiment_name=cfg.logger.experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            log_model=cfg.logger.log_model,
            prefix=cfg.logger.prefix,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
    )
    try:
        LOGGER.info("Starting training.")
        trainer.fit(lit_model, datamodule=datamodule)
        LOGGER.info("Training finished.")
    except Exception:
        LOGGER.exception("Training failed.")
        raise


if __name__ == "__main__":
    main()
