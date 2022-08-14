from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataInSilico:
    datasets_names: List[str]
    dir: str


@dataclass
class DataFos:
    datasets_names: List[str]
    dir: str


@dataclass
class ModelParameters:
    extra_layers_dropout: float
    embedding_size: int
    learning_rate: float
    momentum: float
    epochs: int
    batch_size: int


@dataclass
class PretrainTokenizer:
    padding: bool
    truncation: bool
    return_tensors: str
    max_length: int
    folder_name: str


@dataclass
class ModelConfig:
    name: str
    quantized_name: str
    freeze_params: bool
    hf_model_name: str
    build_own_encoder_lang: str
    dir: str
    conf_data_insilico: Optional[DataInSilico]
    conf_data_fos: Optional[DataFos]
    parameters: ModelParameters
    conf_tokenizer: PretrainTokenizer
    build_own_encoder: bool
    gpu: bool
