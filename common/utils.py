from pathlib import Path
from transformers import ElectraTokenizer
from transformers import ElectraForSequenceClassification
from transformers import ElectraConfig  

import random
import logging
import numpy as np
import torch

from seqeval import metrics as seqeval_metrics

INTERIM_CATEGORY_DIR = Path("data/interim/categories")
CATEGORY_PATH = Path("data/processed/category_set.json")

INTERIM_DATASET_DIR = Path("data/interim/dataset")
DATASET_PATH = Path("data/processed/labeled_reviews.jsonl")

RAW_DATA_DIR = Path("data/raw/")



# Training Arguments
CONFIG_CLASSES = {
    "koelectra-base": ElectraConfig,
}

TOKENIZER_CLASS = {
    "koelectra-base": ElectraTokenizer,
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "koelectra-base": ElectraForSequenceClassification,
}