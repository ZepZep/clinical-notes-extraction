import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold

from datasets import Dataset
from transformers import AutoTokenizer

tqdm.pandas()

inname = "nurse"
outname = "nurse-medBERT"
modelname = "Charangan/MedBERT"

parts = pd.read_feather(f"dataset/{inname}-parts.feather")
relevant = parts.query("label >= 0").reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(modelname)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=512,
        truncation=True,
    )

def make_hf_dataset(relevant, indexer, tokenize_function, numproc=12):
    df = pd.concat([
        pd.DataFrame({
            "text": relevant.text[indexer],
            "label": relevant.label[indexer],
        }),
        pd.DataFrame({
            "text": relevant.stext[indexer],
            "label": relevant.label[indexer],
        })
    ])
    ds = Dataset.from_pandas(df).map(
        tokenize_function, batched=True, num_proc=numproc, desc="Tokenizing")
    ds = ds.remove_columns(['__index_level_0__', 'text'])
    return ds


def make_train_test(relevant):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(relevant, relevant.label):
        break

    ds = {
        "train": make_hf_dataset(relevant, train_index, tokenize_function),
        "test":  make_hf_dataset(relevant, test_index, tokenize_function),
    }
    return ds

print("--> Tokenizing")
ds = make_train_test(relevant)

print("--> Saving dataset")
path = "dataset"
ds["train"].save_to_disk(f"{path}/{outname}-train.hf")
ds["test"].save_to_disk(f"{path}/{outname}-test.hf")
