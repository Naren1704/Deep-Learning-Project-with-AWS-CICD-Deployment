from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_num_classes: int
    params_dense_1: int
    params_dense_2: int
    params_dropout_1: float
    params_dropout_2: float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs_phase1: int
    params_epochs_phase2: int
    params_lr_phase1: float
    params_lr_phase2: float
    params_batch_size: int
    params_image_size: list
    params_classes: list
    params_seed: int
    params_fine_tune_layers: int
    params_class_weights: dict

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    mlflow_uri: str
    all_params: dict
    params_image_size: list
    params_batch_size: int
    params_classes: list