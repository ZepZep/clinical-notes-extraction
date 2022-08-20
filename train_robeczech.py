import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

import torch
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

tqdm.pandas()


print("--> Loading dataset")
dtrain = Dataset.load_from_disk("dataset/train.hf")
dtest = Dataset.load_from_disk("dataset/test.hf")

def reduce_dataset(ds):
    df = ds.to_pandas()
    labels = df.label
    vc = labels.value_counts()
    base = vc.min()
    vc = vc.map(lambda x: base + (x-base)**(2/3))
    counts = defaultdict(int)

    def reduce_(label):
        counts[label] += 1
        return counts[label] <= vc[label]

    ri = labels.progress_apply(reduce_)
    return Dataset.from_pandas(df[ri]).remove_columns(['__index_level_0__'])

#print("--> Reducing dataset")
#dtrain = reduce_dataset(dtrain)
#dtest = reduce_dataset(dtest)

print("--> Loading model")
model = AutoModelForSequenceClassification.from_pretrained(
    "ufal/robeczech-base",
    num_labels=max(dtest["label"])+1
)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    return acc

training_args = TrainingArguments(
    output_dir="models/robeczech",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=5e-05,
    logging_steps=60,
    save_steps=10000,
    fp16=True,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=dtrain,
    eval_dataset=dtest,
    compute_metrics=compute_metrics,
)

print("--> Training")
trainer.train()


print("--> Prediction")
y = trainer.predict(dtest).predictions.astype(np.float16)
print(y.shape)

np.savez_compressed("pred_robeczech.npz", y=y)
