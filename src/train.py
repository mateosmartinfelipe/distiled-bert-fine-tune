from calendar import EPOCH
import gzip
import shutil
import time

import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchtext

import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
import os

# We are downloading a pre-train model and qwe are fine tuning it
# by running backpropagation on the labelled small data set
# as opposed fix the model and used the encoded items as input
# to another model

# Some house keeping
torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("gpu:3" if torch.cuda.is_available() else "cpu")
SPLIT = 0.8
LR = 0.0004
BATCH_SIZE = 16
EPOCHS = 3
# data
url = "https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz"
filename_compress = url.split("/")[-1]
filename_uncompressed = ".".join(filename_compress.split(".")[:-2])

if os.path.exists(filename_uncompressed) is False:
    with open(filename_compress, "wb") as f:
        request = requests.get(url)
        f.write(request.content)

    with gzip.open(filename_compress, "rb") as f_in:
        with open(filename_uncompressed, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

data = pd.read_csv(filename_uncompressed)
print(data.head())
print(data.shape)

# train , test , and validation
train_size = int(data.shape[0] * SPLIT)
validation_size = train_size + int(data.shape[0] * (1 - SPLIT) / 2)

train_feature = data.iloc[:train_size]["review"].values
train_label = data.iloc[:train_size]["sentiment"].values
print(train_feature.shape)

validation_feature = data.iloc[train_size:validation_size]["review"].values
validation_label = data.iloc[train_size:validation_size]["sentiment"].values
print(validation_feature.shape)

test_feature = data.iloc[validation_size:]["review"].values
test_label = data.iloc[validation_size:]["sentiment"].values
print(test_feature.shape)

# create embeddings

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_feature_token = tokenizer(list(train_feature), truncation=True, padding=True)
validation_feature_token = tokenizer(
    list(validation_feature), truncation=True, padding=True
)
test_feature_token = tokenizer(list(test_feature), truncation=True, padding=True)

print(train_feature_token[0])

# data class and loaders


class ImbdClassLoader(torch.utils.data.Dataset):
    def __init__(self, features, labels) -> None:
        super(ImbdClassLoader, self).__init__()
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.features.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ImbdClassLoader(train_feature_token, train_label)
validation_dataset = ImbdClassLoader(validation_feature_token, validation_label)
test_dataset = ImbdClassLoader(test_feature_token, test_label)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)
# Load the model

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.to(DEVICE)
model.train()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Train the model


def accuracy_fn(model, data_loader):

    with model.no_grad():
        correct_prediction, num_samples = 0, 0

        for batch in data_loader:
            # prepare data
            input_id = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            num_samples += labels.size(0)
            outputs = model(input_id, attention_mask=attention_mask, labels=labels)
            _, logits = outputs["loss"], outputs["logits"]
            _, predicted_labels = torch.max(logits, 1)
            correct_prediction += (predicted_labels == labels).sum()
        return (correct_prediction.float() / num_samples) * 100


for epoch in EPOCHS:
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Forward
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs["loss"], outputs["logits"]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
