"""Smoke tests: todos os módulos do projeto precisam importar sem erro."""


def test_import_src_packages():
    import src  # noqa: F401
    import src.paths  # noqa: F401
    import src.models  # noqa: F401
    import src.models.checkpoints  # noqa: F401
    import src.datasets  # noqa: F401
    import src.datasets.affectnet_dataset  # noqa: F401
    import src.training  # noqa: F401
    import src.training.config  # noqa: F401
    import src.training.utils  # noqa: F401
    import src.tensor_utils  # noqa: F401
    import src.evaluation  # noqa: F401
    import src.evaluation.inference  # noqa: F401
    import src.evaluation.model_loading  # noqa: F401
    import src.evaluation.dataloaders  # noqa: F401
    import src.evaluation.per_model  # noqa: F401
    import src.evaluation.cross_label  # noqa: F401
    import src.preprocessing  # noqa: F401


def test_import_root_scripts():
    import run_evaluation  # noqa: F401
    import run_cross_label_evaluation  # noqa: F401
    import run_impact_study  # noqa: F401
    import train_cnn3d  # noqa: F401
    import train_emotion_model  # noqa: F401
    import train_multimodal  # noqa: F401
