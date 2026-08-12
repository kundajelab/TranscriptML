"""Structured RBPNet training integrated with TranscriptML checkpoints/configs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from transcriptml.data.bundle import DatasetBundle
from transcriptml.devices import resolve_device
from transcriptml.models.rbpnet import RBPNet
from transcriptml.models.registry import build_model, normalize_model_config, save_checkpoint
from transcriptml.progress import ProgressReporter, log_progress
from transcriptml.rbpnet.dataset import (
    RBPNetBatch,
    RBPNetDataset,
    collate_rbpnet,
    deduplicate_locus_indices,
)
from transcriptml.rbpnet.losses import RBPNetLossConfig, RBPNetObjective
from transcriptml.training.splits import (
    group_split_indices,
    normalize_splits,
    predefined_split_indices,
    random_split_indices,
    validate_group_disjoint,
)


def _config_dict(value: str | Mapping[str, Any] | None, default_name: str) -> dict[str, Any]:
    if value is None:
        return {"name": default_name, "params": {}}
    if isinstance(value, str):
        return {"name": value, "params": {}}
    result = dict(value)
    params = dict(result.pop("params", {}) or {})
    params.update(result)
    return {"name": str(params.pop("name", default_name)).lower(), "params": params}


def _build_optimizer(model: torch.nn.Module, cfg) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    config = _config_dict(getattr(cfg, "optimizer", None), "adamw")
    name = config["name"]
    params = dict(config["params"])
    params.setdefault("lr", float(cfg.learning_rate))
    params.setdefault("weight_decay", float(cfg.weight_decay))
    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **params)
    elif name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **params)
    elif name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **params)
    else:
        raise ValueError("RBPNet optimizer must be one of: adamw, adam, sgd")
    return optimizer, {"name": name, "params": params}


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    value: str | Mapping[str, Any] | None,
    *,
    epochs: int,
) -> tuple[object | None, dict[str, Any] | None, bool]:
    if value is None:
        return None, None, False
    config = _config_dict(value, "none")
    name = config["name"]
    params = dict(config["params"])
    if name in {"none", "off", "disabled"}:
        return None, {"name": "none", "params": {}}, False
    if name in {"reduce_on_plateau", "plateau"}:
        params.setdefault("mode", "min")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        return scheduler, {"name": "reduce_on_plateau", "params": params}, True
    if name in {"cosine", "cosine_annealing"}:
        params.setdefault("T_max", int(epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        return scheduler, {"name": "cosine", "params": params}, False
    if name in {"step", "step_lr"}:
        params.setdefault("step_size", 10)
        params.setdefault("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
        return scheduler, {"name": "step", "params": params}, False
    raise ValueError("RBPNet lr_scheduler must be one of: none, reduce_on_plateau, cosine, step")


def _build_grad_scaler(*, enabled: bool):
    """Build a CUDA scaler across the supported PyTorch API variants."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # PyTorch versions without the device argument.
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_config_splits(bundle: DatasetBundle, cfg) -> tuple[dict[str, list[int]], str, bool]:
    split_cfg = dict(cfg.split or {})
    method = str(split_cfg.get("method", "group")).lower()
    if method == "group":
        if bundle.metadata is None:
            raise ValueError("group split requested but RBPNet bundle has no metadata")
        group_col = str(split_cfg.get("group_col", "group_gene_id"))
        return (
            group_split_indices(
                bundle.metadata,
                group_col=group_col,
                val_frac=float(split_cfg.get("val_frac", 0.1)),
                test_frac=float(split_cfg.get("test_frac", 0.1)),
                seed=int(split_cfg.get("seed", cfg.seed)),
            ),
            group_col,
            False,
        )
    if method == "metadata":
        if bundle.metadata is None:
            raise ValueError("metadata split requested but RBPNet bundle has no metadata")
        return (
            predefined_split_indices(
                bundle.metadata,
                split_col=str(split_cfg.get("split_col", "split")),
            ),
            str(split_cfg.get("group_col", "group_gene_id")),
            False,
        )
    if method == "predefined":
        return (
            normalize_splits(split_cfg["splits"]),
            str(split_cfg.get("group_col", "group_gene_id")),
            False,
        )
    if method == "random":
        if not bool(getattr(cfg, "allow_random_window_split", False)):
            raise ValueError(
                "row-level random splitting is unsafe for overlapping RBPNet windows; "
                "use split.method='group' (recommended) or explicitly set "
                "allow_random_window_split=true"
            )
        return (
            random_split_indices(
                int(bundle.X.shape[0]),
                val_frac=float(split_cfg.get("val_frac", 0.1)),
                test_frac=float(split_cfg.get("test_frac", 0.1)),
                seed=int(split_cfg.get("seed", cfg.seed)),
            ),
            str(split_cfg.get("group_col", "group_gene_id")),
            True,
        )
    raise ValueError(f"Unknown RBPNet split method {method!r}")


def _select_rbpnet_splits(
    bundle: DatasetBundle,
    cfg,
) -> tuple[dict[str, list[int]], str, str, bool]:
    source = str(cfg.split_source or "auto").strip().lower()
    if source not in {"auto", "bundle", "config"}:
        raise ValueError("split_source must be one of: auto, bundle, config")
    if source == "bundle" or (source == "auto" and bundle.splits is not None):
        if bundle.splits is None:
            raise ValueError("split_source='bundle' requested but dataset bundle has no splits")
        splits = normalize_splits(bundle.splits)
        group_col = str(dict(cfg.split or {}).get("group_col", "group_gene_id"))
        allow_random = False
        source_used = "bundle"
    else:
        splits, group_col, allow_random = _make_config_splits(bundle, cfg)
        source_used = "config"
    if not allow_random:
        if bundle.metadata is None:
            raise ValueError("leakage validation requires RBPNet bundle metadata")
        validate_group_disjoint(splits, bundle.metadata, group_col=group_col)
    return splits, source_used, group_col, allow_random


def _deduplicate_splits(
    bundle: DatasetBundle,
    splits: Mapping[str, Sequence[int]],
    *,
    enabled: bool,
) -> tuple[dict[str, list[int]], int]:
    normalized = normalize_splits(splits)
    if not enabled:
        return normalized, 0
    result: dict[str, list[int]] = {}
    dropped = 0
    for name, indices in normalized.items():
        result[name], count = deduplicate_locus_indices(bundle, indices)
        dropped += count
    return result, dropped


def _loader(
    dataset: RBPNetDataset,
    indices: Sequence[int],
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader | None:
    if not indices:
        return None
    return DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
        collate_fn=collate_rbpnet,
    )


def _aggregate_loss(
    numerators: Mapping[str, float],
    denominators: Mapping[str, float],
    loss_config: RBPNetLossConfig,
    *,
    enrichment_enabled: bool,
) -> dict[str, float]:
    components = {
        name: float(numerators[name] / denominators[name])
        if denominators[name] > 0
        else 0.0
        for name in numerators
    }
    total = (
        float(loss_config.lambda_ip_profile) * components["ip_profile_loss"]
        + float(loss_config.lambda_sm_profile) * components["sm_profile_loss"]
        + (
            float(loss_config.lambda_enrichment) * components["enrichment_loss"]
            if enrichment_enabled
            else 0.0
        )
    )
    return {"loss": total, **components}


def _run_loader(
    model: RBPNet,
    loader: DataLoader | None,
    *,
    device: torch.device,
    objective: RBPNetObjective,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    mixed_precision: bool = False,
    scaler: Any | None = None,
    progress: bool = True,
    progress_label: str = "RBPNet batches",
    return_predictions: bool = False,
) -> dict[str, Any]:
    names = ("ip_profile_loss", "sm_profile_loss", "enrichment_loss")
    if loader is None:
        return {
            "loss": float("nan"),
            **{name: float("nan") for name in names},
            "n_examples": 0,
            "indices": [],
            "example_ids": [],
            "pi": np.empty(0, dtype=np.float32),
            "enrichment_logit": None,
        }
    training = optimizer is not None
    model.train(training)
    numerators = {name: 0.0 for name in names}
    denominators = {name: 0.0 for name in names}
    pis: list[np.ndarray] = []
    etas: list[np.ndarray] = []
    indices: list[int] = []
    example_ids: list[str] = []
    n_examples = 0
    reporter = ProgressReporter(
        progress_label,
        total=len(loader),
        unit="batches",
        enabled=progress,
        percent_step=25.0,
    )
    amp_enabled = bool(mixed_precision)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for batch in loader:
        assert isinstance(batch, RBPNetBatch)
        batch = batch.to(device)
        n_examples += int(batch.sequence.shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(
                    batch.sequence,
                    measurement_mask=batch.measurement_mask,
                    profile_mask=batch.profile_valid_mask,
                )
                loss_output = objective(output, batch)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_output.loss).backward()
                    if gradient_clip_norm is not None and float(gradient_clip_norm) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(gradient_clip_norm)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_output.loss.backward()
                    if gradient_clip_norm is not None and float(gradient_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(gradient_clip_norm)
                        )
                    optimizer.step()
        for name in names:
            numerators[name] += float(loss_output.numerators[name].detach().cpu())
            denominators[name] += float(loss_output.denominators[name].detach().cpu())
        if return_predictions:
            pis.append(output.pi.detach().float().cpu().numpy())
            if output.enrichment_logit is not None:
                etas.append(output.enrichment_logit.detach().float().cpu().numpy())
            indices.extend(int(value) for value in batch.indices.detach().cpu().tolist())
            example_ids.extend(batch.example_ids)
        reporter.update()
    reporter.close()
    metrics: dict[str, Any] = _aggregate_loss(
        numerators,
        denominators,
        objective.config,
        enrichment_enabled=objective.enrichment_enabled,
    )
    metrics["n_examples"] = n_examples
    if return_predictions:
        metrics.update(
            {
                "indices": indices,
                "example_ids": example_ids,
                "pi": np.concatenate(pis) if pis else np.empty(0, dtype=np.float32),
                "enrichment_logit": (
                    np.concatenate(etas) if etas else None
                ),
            }
        )
    return metrics


def _is_better(value: float, best: float | None, name: str) -> bool:
    if not np.isfinite(value):
        return False
    if best is None:
        return True
    return value < best if name.endswith("loss") else value > best


def _monitor_names(value: str | Sequence[str]) -> tuple[str, ...]:
    names = (
        [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, str)
        else [str(part).strip() for part in value if str(part).strip()]
    )
    if not names:
        raise ValueError("monitor must name at least one metric")
    return tuple(names)


def _write_predictions(
    path: str | Path,
    metrics: Mapping[str, Any],
    *,
    depth_offsets: np.ndarray,
    replicate_names: Sequence[str],
) -> None:
    eta = metrics.get("enrichment_logit")
    fieldnames = ["index", "id", "pi"]
    if eta is not None:
        fieldnames.append("enrichment_logit")
        fieldnames.extend(f"predicted_ip_fraction_{name}" for name in replicate_names)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, (index, identifier, pi) in enumerate(
            zip(metrics["indices"], metrics["example_ids"], metrics["pi"])
        ):
            row: dict[str, object] = {
                "index": int(index),
                "id": str(identifier),
                "pi": float(pi),
            }
            if eta is not None:
                value = float(eta[position])
                row["enrichment_logit"] = value
                probabilities = 1.0 / (1.0 + np.exp(-(value + depth_offsets)))
                for name, probability in zip(replicate_names, probabilities):
                    row[f"predicted_ip_fraction_{name}"] = float(probability)
            writer.writerow(row)


def train_rbpnet_model(bundle: DatasetBundle, cfg) -> dict[str, Any]:
    """Train a registered RBPNet model without routing through scalar targets."""

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    device = resolve_device(cfg.device)
    model_config = normalize_model_config(cfg.model)
    if model_config.name != "rbpnet":
        raise ValueError("train_rbpnet_model requires model.name='rbpnet'")
    model = build_model(model_config).to(device)
    if not isinstance(model, RBPNet):
        raise TypeError("registered rbpnet model did not build an RBPNet instance")
    crop_length = int(bundle.config.get("input_length", 0))
    if model.profile_length is not None and model.profile_length != crop_length:
        raise ValueError(
            f"model profile_length {model.profile_length} does not match bundle crop "
            f"length {crop_length}"
        )
    loss_config = RBPNetLossConfig.from_config(cfg.loss)
    objective = RBPNetObjective(
        loss_config,
        enrichment_enabled=model.enrichment_enabled,
    ).to(device)
    splits, split_source, group_col, random_split = _select_rbpnet_splits(bundle, cfg)
    splits, n_deduplicated = _deduplicate_splits(
        bundle,
        splits,
        enabled=bool(getattr(cfg, "deduplicate_loci", True)),
    )
    if not splits.get("train"):
        raise ValueError("RBPNet training split is empty after locus deduplication")

    train_dataset = RBPNetDataset(
        bundle,
        crop_length=crop_length,
        max_train_jitter=int(getattr(cfg, "max_train_jitter", 0)),
        training=True,
        seed=int(cfg.seed),
        require_full_measurement_interval=model.enrichment_enabled,
    )
    eval_dataset = RBPNetDataset(
        bundle,
        crop_length=crop_length,
        max_train_jitter=0,
        training=False,
        seed=int(cfg.seed),
        require_full_measurement_interval=model.enrichment_enabled,
    )
    pin_memory = device.type == "cuda"
    train_loader = _loader(
        train_dataset,
        splits["train"],
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _loader(
        eval_dataset,
        splits.get("val", []),
        cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    optimizer, optimizer_config = _build_optimizer(model, cfg)
    scheduler, scheduler_config, scheduler_uses_metric = _build_scheduler(
        optimizer,
        getattr(cfg, "lr_scheduler", None),
        epochs=int(cfg.epochs),
    )
    mixed_precision = bool(getattr(cfg, "mixed_precision", False))
    if mixed_precision and device.type not in {"cpu", "cuda"}:
        raise ValueError("RBPNet mixed precision currently supports CPU and CUDA devices")
    scaler = _build_grad_scaler(enabled=mixed_precision and device.type == "cuda")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    split_counts = {name: len(splits.get(name, [])) for name in ("train", "val", "test")}
    log_progress(
        (
            f"RBPNet training: device={device}, parameters={parameter_count:,}, "
            f"receptive_field={model.receptive_field}, jitter={train_dataset.max_train_jitter}, "
            f"train={split_counts['train']}, val={split_counts['val']}, "
            f"test={split_counts['test']}"
        ),
        enabled=cfg.progress,
    )

    history: list[dict[str, float | int]] = []
    monitors = _monitor_names(cfg.monitor)
    best_values: dict[str, float | None] = {name: None for name in monitors}
    best_epoch = -1
    stale = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = _run_loader(
            model,
            train_loader,
            device=device,
            objective=objective,
            optimizer=optimizer,
            gradient_clip_norm=cfg.gradient_clip_norm,
            mixed_precision=mixed_precision,
            scaler=scaler,
            progress=cfg.progress,
            progress_label=f"epoch {epoch} RBPNet train",
        )
        val_metrics = _run_loader(
            model,
            val_loader,
            device=device,
            objective=objective,
            mixed_precision=mixed_precision,
            progress=cfg.progress,
            progress_label=f"epoch {epoch} RBPNet val",
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
        }
        for component in ("ip_profile_loss", "sm_profile_loss", "enrichment_loss"):
            row[f"train_{component}"] = float(train_metrics[component])
            row[f"val_{component}"] = float(val_metrics[component])
        missing = [name for name in monitors if name not in row]
        if missing:
            raise ValueError(f"Unknown RBPNet monitor metric(s): {', '.join(missing)}")
        values = {name: float(row[name]) for name in monitors}
        improved = any(_is_better(values[name], best_values[name], name) for name in monitors)
        history.append(row)
        checkpoint_extra = {
            "splits": splits,
            "split_source_used": split_source,
            "train_config": asdict(cfg),
            "loss_config": loss_config.to_dict(
                enrichment_enabled=model.enrichment_enabled
            ),
            "optimizer_config": optimizer_config,
            "lr_scheduler_config": scheduler_config,
            "coordinate_space": bundle.config.get("coordinate_space"),
            "sample_metadata": bundle.config.get("sample_metadata"),
            "parameter_count": parameter_count,
            "receptive_field": model.receptive_field,
        }
        if improved:
            best_values = values
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                out / "best.pt",
                model,
                model_config,
                epoch=epoch,
                metrics=row,
                optimizer_state=optimizer.state_dict(),
                extra=checkpoint_extra,
            )
        else:
            stale += 1
        save_checkpoint(
            out / "last.pt",
            model,
            model_config,
            epoch=epoch,
            metrics=row,
            optimizer_state=optimizer.state_dict(),
            extra=checkpoint_extra,
        )
        if scheduler is not None:
            metric = float(val_metrics["loss"])
            if not np.isfinite(metric):
                metric = float(train_metrics["loss"])
            scheduler.step(metric) if scheduler_uses_metric else scheduler.step()
        log_progress(
            (
                f"epoch {epoch}/{cfg.epochs}: train={row['train_loss']:.6g}, "
                f"val={row['val_loss']:.6g}, "
                f"IP={row['val_ip_profile_loss']:.6g}, "
                f"SM={row['val_sm_profile_loss']:.6g}, "
                f"enrichment={row['val_enrichment_loss']:.6g}"
            ),
            enabled=cfg.progress,
        )
        if int(cfg.patience) >= 0 and stale > int(cfg.patience):
            break

    test_loader = _loader(
        eval_dataset,
        splits.get("test", []),
        cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_metrics = _run_loader(
        model,
        test_loader,
        device=device,
        objective=objective,
        mixed_precision=mixed_precision,
        progress=cfg.progress,
        progress_label="RBPNet test",
        return_predictions=True,
    )
    if test_metrics["indices"]:
        _write_predictions(
            out / "test_predictions.csv",
            test_metrics,
            depth_offsets=eval_dataset.depth_offsets,
            replicate_names=eval_dataset.replicate_names,
        )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary = {
        "trainer": "rbpnet",
        "best_epoch": best_epoch,
        "monitor": list(monitors),
        "best_monitor_values": best_values,
        "epochs_run": len(history),
        "loss": loss_config.to_dict(enrichment_enabled=model.enrichment_enabled),
        "optimizer": optimizer_config,
        "lr_scheduler": scheduler_config,
        "mixed_precision": mixed_precision,
        "max_train_jitter": train_dataset.max_train_jitter,
        "split_source_used": split_source,
        "split_group_col": group_col,
        "unsafe_random_window_split": random_split,
        "split_counts": split_counts,
        "deduplicate_loci": bool(getattr(cfg, "deduplicate_loci", True)),
        "n_deduplicated_rows": n_deduplicated,
        "parameter_count": parameter_count,
        "receptive_field": model.receptive_field,
        "receptive_field_extents": list(model.receptive_field_extents),
        "replicate_names": list(eval_dataset.replicate_names),
        "depth_offsets": eval_dataset.depth_offsets.tolist(),
        "test_loss": float(test_metrics["loss"]),
        "test_ip_profile_loss": float(test_metrics["ip_profile_loss"]),
        "test_sm_profile_loss": float(test_metrics["sm_profile_loss"]),
        "test_enrichment_loss": float(test_metrics["enrichment_loss"]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"model": model, "history": history, "splits": splits, "summary": summary}


@torch.no_grad()
def evaluate_rbpnet_model(
    model: RBPNet,
    bundle: DatasetBundle,
    *,
    indices: Sequence[int] | None = None,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    loss_config: str | Mapping[str, object] | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Deterministically evaluate structured profile/enrichment likelihoods."""

    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    dataset = RBPNetDataset(
        bundle,
        max_train_jitter=0,
        training=False,
        require_full_measurement_interval=model.enrichment_enabled,
    )
    selected = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    loader = _loader(
        dataset,
        selected,
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=resolved_device.type == "cuda",
    )
    objective = RBPNetObjective(
        RBPNetLossConfig.from_config(loss_config),
        enrichment_enabled=model.enrichment_enabled,
    ).to(resolved_device)
    metrics = _run_loader(
        model,
        loader,
        device=resolved_device,
        objective=objective,
        progress=progress,
        progress_label="evaluate RBPNet",
        return_predictions=True,
    )
    metrics["depth_offsets"] = dataset.depth_offsets
    metrics["replicate_names"] = dataset.replicate_names
    return metrics


def write_rbpnet_predictions(
    path: str | Path,
    metrics: Mapping[str, Any],
) -> None:
    """Write pi, eta, and replicate-specific predicted IP fractions."""

    _write_predictions(
        path,
        metrics,
        depth_offsets=np.asarray(metrics["depth_offsets"], dtype=np.float64),
        replicate_names=metrics["replicate_names"],
    )
