import json
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoTokenizer, PreTrainedTokenizer

from config import ModelConfig
from utils import FiledsOfStudy, Publication

# STRATEGY PATTModelConfigER ALLOWS YOU TO INJECT BEHAVIOR INTO AN APPLICATION
# WITHOUT THE CODE KNOWING EXACTLY WHAT IS DOES.

# using protocols
atom_elem = Callable[[Any], str]


class Pipeline(Protocol):
    def build_pipelines(self) -> Tuple[Callable[[str], torch.Tensor], atom_elem]:
        raise NotImplementedError

    @property
    def get_tokenizer(self) -> Union[PreTrainedTokenizer, atom_elem]:
        raise NotImplementedError

    def save_tokenizer(self, dir: str) -> None:
        raise NotImplementedError

    @staticmethod
    def load_tokenizer(dir: str) -> Tokenizer:
        raise NotImplementedError


class BasicPipeline:
    def __init__(
        self,
        config: ModelConfig,
        data_iter: Iterator[Tuple[str, str]],
        labels_encoder: Optional[Dict[str, int]],
    ):

        self.config: ModelConfig = config
        self.tokenizer: atom_elem = get_tokenizer(config.model.build_own_encoder_lang)
        self.data_iter: Iterator[Tuple[str, str]] = data_iter
        self.labels_encoder: Optional[Dict[str, int]] = (
            labels_encoder if labels_encoder else None
        )

    def _vovabulary_iter(self) -> Iterator[str]:
        """clean the input string"""
        for _, text in self.data_iter:
            yield self.tokenizer(text)

    def build_pipelines(self) -> Tuple[atom_elem, Optional[atom_elem]]:
        vovabulary_iter_func = partial(self._vovabulary_iter, tokenizer=self.tokenizer)
        vocab = build_vocab_from_iterator(
            vovabulary_iter_func(self.data_iter), specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        text_pipeline = lambda x: self.tokenizer(x)
        if self.labels_encoder:
            label_pipeline = lambda x: self.labels_encoder.get(x.lower(), -1)
        else:
            label_pipeline = None

        return (text_pipeline, label_pipeline)

    @property
    def get_tokenizer(self) -> Union[PreTrainedTokenizer, atom_elem]:
        return self.tokenizer


class PreTrainedEncoder:
    def __init__(
        self,
        config: ModelConfig,
        data_iter: Iterator[Tuple[str, str]],
        labels_encoder: Optional[Dict[str, int]],
    ):
        self.config: ModelConfig = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.hf_model_name)
        self.data_iter: Iterator[Tuple[str, str]] = data_iter
        self.labels_encoder: Dict[str, int] = labels_encoder if labels_encoder else None

    def _one_hot_encode(self, label: str) -> np.ndarray:
        code = self.labels_encoder.get(label.lower(), -1)
        if code < 0:
            raise Exception("Label not found in dict") from None
        _zeros = np.zeros(len(self.labels_encoder), dtype=np.float64)
        _zeros[code] = 1.0
        return _zeros

    def build_pipelines(
        self,
    ) -> Tuple[Callable[[str], torch.Tensor], Optional[atom_elem]]:
        text_pipeline = lambda x: self.tokenizer(
            x,
            add_special_tokens=True,
            padding=self.config.conf_tokenizer.padding,
            truncation=self.config.conf_tokenizer.truncation,
            return_tensors=self.config.conf_tokenizer.return_tensors,
            max_length=self.config.conf_tokenizer.max_length,
        )
        if self.labels_encoder:
            label_pipeline = lambda x: self._one_hot_encode(x)
        else:
            label_pipeline = None
        return (text_pipeline, label_pipeline)

    @property
    def get_tokenizer(self) -> Union[PreTrainedTokenizer, atom_elem]:
        return self.tokenizer

    def save_tokenizer(self, dir: str) -> None:
        self.tokenizer.save_pretrained(save_directory=dir)

    def load_tokenizer(dir: str) -> Tokenizer:
        return AutoTokenizer.from_pretrained(dir)


# RELAYING ON DUNDER METHODS ( IT IMPLIES TO USE __CALL__).
# AS IT IS WE CAN NOT USE IT AS WE HAVE CLASSS PROPERTY


# USING FUNCTIONS TO IMPLEMENT STRATEGY PATTERN


class ModelDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, publications: Union[List[Publication], List[FiledsOfStudy]]):
        "Initialization"
        self.publications = publications

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.publications)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.publications[index].text
        y = self.publications[index].label
        return X, y


def fos_get_data(data_file_path: str, labels_to_keep: List[int]) -> List[FiledsOfStudy]:
    data = pd.read(data_file_path, engine="pyarrow")
    labels_to_keep_set = set(labels_to_keep)
    publications = []
    """nn.BCEWithLogitsLoss takes the raw logits of your model (without any non-linearity) and applies the sigmoid internally.
    If you would like to add the sigmoid activation to your model, you should use nn.BCELoss instead."""
    for row in data.itertuples(index=True, name="Pandas"):
        labels = np.zeros(len(labels_to_keep), dtype=np.float16)
        targets = getattr(row, "fos")
        valid_targets = list(set(targets).intersection(labels_to_keep_set))
        for elem in valid_targets:
            try:
                labels[labels_to_keep.index(elem)] = 1.0
            except ValueError:
                pass
        if elem.sum() > 0:
            publications.append(
                FiledsOfStudy(
                    publication_id=getattr(row, "PublicationId"),
                    label=labels,
                    text=getattr(row, "Abstract"),
                )
            )
    return publications


def inSilico_get_data_and_label_encoder(
    data_file_path: str,
) -> Tuple[List[Publication], Dict[str, int]]:
    records = []
    labels_encoding = set()
    with open(data_file_path, "r") as f:
        for l in f.readlines():
            data_point = json.loads(l)
            # we could use BaseModel here, as the pydantic Class implements
            # a method to build obejcts from dictionaries
            publication = Publication(
                article_id=data_point["Article_ID"],
                creation_date=data_point["CreateDate"],
                journal=data_point["Journal"],
                label=data_point["Status"].lower(),
                verified=data_point["verified?"],
                text=data_point["Body_Text"],
            )
            records.append(publication)
            labels_encoding.add(publication.label.lower())

    labels_encoder = {v: idx for idx, v in enumerate(sorted(labels_encoding))}
    return (records, labels_encoder)


def get_label_coded(publications: Iterator[Tuple[int, str]]) -> int:
    """convert the label to a numeric label"""
    labels = set()
    for label, _ in publications:
        labels.add(label)
    return


def get_pipelines(
    pipeline: Pipeline,
) -> Tuple[atom_elem, Optional[atom_elem]]:
    return pipeline.build_pipelines()


def predict(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    text: str,
    labels_encoder: Dict[str, int],
) -> str:
    """predict the label of a text"""
    text_tensor = tokenizer(text, return_tensors="pt")
    predictions = model(**text_tensor)
    _, predicted = torch.max(predictions, 1)
    return labels_encoder[predicted.item()]


def collate_batch(
    batch: List[Tuple[str, Any]],
    label_pipeline: Optional[atom_elem],
    text_pipeline: atom_elem,
    device: torch.device,
):
    """collate the batch"""
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        if label_pipeline:
            label_list.append(
                torch.unsqueeze(
                    torch.tensor(label_pipeline(_label), dtype=torch.float64), dim=0
                )
            )
        else:
            label_list.append(torch.tensor(_label, dtype=torch.float64))
        tokenized_text = text_pipeline(_text)
        processed_text = tokenized_text["input_ids"]
        text_list.append(torch.tensor(processed_text, dtype=torch.int64))
        offsets.append(len(processed_text))
    label_list = torch.cat(label_list, dim=0)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)
