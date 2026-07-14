import numpy as np

from transcriptml.data.bundle import DatasetBundle, save_bundle
from transcriptml.devices import resolve_device
from transcriptml.interpret.predictor import Predictor
from transcriptml.models.registry import build_model, save_checkpoint
from transcriptml.training.evaluation import evaluate_checkpoint


def _small_model_config() -> dict[str, object]:
    return {
        "name": "small_cnn",
        "params": {
            "in_ch": 4,
            "n_filters": 4,
            "kernel_size": 3,
            "n_layers": 1,
            "dropout": 0.0,
            "head_hidden": 4,
        },
    }


def _write_checkpoint_and_bundle(tmp_path):
    model_config = _small_model_config()
    model = build_model(model_config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, model_config)

    X = np.zeros((3, 4, 12), dtype=np.float32)
    X[:, 0, :] = 1.0
    bundle = DatasetBundle(
        X=X,
        y=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        ids=["a", "b", "c"],
        schema="rna4",
        splits={"train": [0], "val": [1], "test": [2]},
    )
    dataset = tmp_path / "dataset"
    save_bundle(bundle, dataset)
    return checkpoint, dataset


def test_resolve_device_auto_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert str(resolve_device("auto")) == "cpu"


def test_evaluate_checkpoint_accepts_auto_device(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    checkpoint, dataset = _write_checkpoint_and_bundle(tmp_path)

    result = evaluate_checkpoint(
        checkpoint,
        dataset,
        tmp_path / "predictions.csv",
        split="test",
        device="auto",
        progress=False,
    )

    assert len(result["predictions"]) == 1
    assert (tmp_path / "predictions.csv").exists()


def test_predictor_from_checkpoint_accepts_auto_device(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    checkpoint, _ = _write_checkpoint_and_bundle(tmp_path)

    predictor = Predictor.from_checkpoint(checkpoint, device="auto", batch_size=2)
    preds = predictor.predict(np.zeros((2, 4, 12), dtype=np.float32))

    assert preds.shape == (2,)
