import logging
import os
from dataclasses import dataclass
from typing import List

import torch


def get_system_device(with_gpu: bool) -> torch.device:
    """Get system device"""
    if with_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif with_gpu and torch.has_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_up_folders(dir: List[str]) -> None:
    """Set up folders"""
    for folder in dir:
        logging.info(f"Creating folder '{folder}' if does not exist")
        os.makedirs(folder, exist_ok=True)


@dataclass
class Publication:
    article_id: int
    creation_date: int
    journal: str
    label: str
    verified: str
    text: str


@dataclass
class FiledsOfStudy:
    """['PublicationId', 'fos', 'Abstract', 'Title', 'Doi', 'PublishedDate',
    'JournalId', 'Issn']"""

    publication_id: int
    label: List[int]
    text: str
