from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list



@dataclass(frozen = True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path


@dataclass(frozen= True)
class ModeTrainerConfig:
  root_dir: Path
  train_data_path: Path
  test_data_path: Path
  model_ckpt: Path
  output_dir: Path
  learning_rate: float
  per_device_train_batch_size: int
  per_device_eval_batch_size: int
  num_train_epochs: int
  weight_decay: float
  eval_steps: int
  evaluation_strategy: str
  save_strategy: str
  load_best_model_at_end: bool   


  
@dataclass(frozen= True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path:Path
    model_path:  Path
    tokenizer_path: Path
    metric_file_name: Path
