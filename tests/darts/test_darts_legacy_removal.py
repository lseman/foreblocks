import inspect

from darts.config import DARTSTrainConfig, PC_DARTSEngineConfig
from darts.trainer import DARTSTrainer


def test_trainer_search_api_accepts_config_instead_of_legacy_overrides() -> None:
    parameters = inspect.signature(DARTSTrainer.train_darts_model).parameters

    assert set(parameters) == {
        "self",
        "model",
        "train_loader",
        "val_loader",
        "train_config",
        "compute_metrics",
    }
    assert parameters["train_config"].default is None


def test_train_config_has_no_deprecated_alias_fields() -> None:
    config = DARTSTrainConfig()

    for field in (
        "identity_dominance_cap",
        "edge_sharpening_strength",
        "progressive_training",
        "pruning_enabled",
        "pruning_start_epoch",
        "pruning_threshold",
        "log_arch_gradients",
    ):
        assert not hasattr(config, field)


def test_pc_darts_config_contains_only_canonical_controls() -> None:
    assert set(PC_DARTSEngineConfig.__dataclass_fields__) == {
        "enable_partial_channels",
        "enable_edge_normalization",
    }
