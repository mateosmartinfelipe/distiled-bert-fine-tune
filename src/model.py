# all functional
import logging
from functools import partial
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from config import ModelConfig


class CustomModel(nn.Module):
    def __init__(self, conf: ModelConfig, num_of_labels: int):
        """Initialize the model"""
        super(CustomModel, self).__init__()
        self.conf = conf
        self.num_of_labels = num_of_labels
        self.quant = torch.quantization.QuantStub()
        self.model = AutoModel.from_pretrained(conf.model.hf_model_name)
        self.dropout = nn.Dropout(conf.conf_model.extra_layers_dropout)
        self.linear = nn.Linear(conf.conf_model.embedding_size, self.num_of_labels)
        self.dequant = torch.quantization.DeQuantStub()
        self._freeze_params()

    def _freeze_params(self) -> None:
        if self.conf.model.hf_model_freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(
        self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None
    ):
        """Forward pass of the model"""
        # Forward pass
        last_hidden_layer = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Get the hidden state from the last layer , [ CLS ] sentence representation
        # first input token
        sentence_embedding = last_hidden_layer[0]
        # Apply dropout
        sentence_embedding = self.dropout(sentence_embedding)
        # Apply the linear layer
        class_scores = self.linear(sentence_embedding[:, 0, :])

        return class_scores


class QuantizeCustomModel(nn.Module):
    def __init__(self, conf: ModelConfig, num_of_labels: int):
        """Initialize the model"""
        super(CustomModel, self).__init__()
        self.conf = conf
        self.num_of_labels = num_of_labels
        self.quant = torch.quantization.QuantStub()
        self.model = AutoModel.from_pretrained(conf.model.hf_model_name)
        self.dropout = nn.Dropout(conf.conf_model.extra_layers_dropout)
        self.linear = nn.Linear(conf.conf_model.embedding_size, self.num_of_labels)
        self.dequant = torch.quantization.DeQuantStub()
        self._freeze_params()

    def _freeze_params(self) -> None:
        if self.conf.model.hf_model_freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(
        self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None
    ):
        """Forward pass of the model"""
        # Forward pass
        last_hidden_layer = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Get the hidden state from the last layer , [ CLS ] sentence representation
        # first input token
        sentence_embedding = last_hidden_layer[0]
        # Apply dropout
        sentence_embedding = self.dropout(sentence_embedding)
        # Apply the linear layer
        class_scores = self.linear(sentence_embedding[:, 0, :])

        return class_scores


def build_model(
    conf: ModelConfig, labels: int, device: torch.device, model_to_build: nn.Module
) -> Tuple[nn.Module, optim.Optimizer]:
    """Build the model"""
    model = model_to_build(conf, labels)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf.conf_model.learning_rate)
    return model, optimizer


def _train_step(
    model: nn.Module,
    loss: Callable[[torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    metric: Callable[[torch.Tensor], torch.Tensor],
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Individual train step"""
    labels, inputs, _ = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss_value = loss(outputs, labels)
    loss_value.backward()
    optimizer.step()

    return loss_value.item(), metric(outputs, labels)


def _accuracy_function(outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multiclass accuracy function"""
    pred = torch.argmax(outputs, dim=1)
    observed = torch.argmax(target, dim=1)
    correct = torch.sum(pred == observed).item()
    return correct / len(target)


def _loss_function(outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss function"""
    loss = nn.CrossEntropyLoss()(outputs, target)
    return loss


def train_loop(
    model: nn.Module,
    loss: Callable[[torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    metric: Callable[[torch.Tensor], torch.Tensor],
    data: DataLoader,
    epochs: int,
    data_size: int,
) -> None:
    """Train loop"""
    cumulative_loss = 0.0
    for epoch in range(epochs):
        batch_counter = 0
        num_records = 0
        cumulative_loss = 0.0
        cumulative_metric = 0.0
        with tqdm(total=data_size) as b:
            for batch in data:
                batch_counter += 1
                num_records = num_records + len(batch[1])
                loss_value, metric_value = _train_step(
                    model, loss, optimizer, metric, batch
                )
                cumulative_loss += loss_value
                cumulative_metric += metric_value
                if batch_counter % 5 == 0:
                    logging.info(
                        f"Epoch: {epoch} Batch: {batch_counter} Records: {num_records}  loss: {cumulative_loss / batch_counter:.3f} accuracy: {cumulative_metric / batch_counter:.3f}"
                    )
                b.update(len(batch[1]))


def predict(model: nn.Module, data: DataLoader) -> None:
    """Predict"""
    for batch in data:
        labels, inputs, _ = batch
        outputs = model(inputs)
        pred = torch.argmax(outputs, dim=1)
        print(pred)


train_loop_fn = partial(train_loop, loss=_loss_function, metric=_accuracy_function)
