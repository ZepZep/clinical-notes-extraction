import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

import torch
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from eval_utils import create_metrics

tqdm.pandas()

inname = "nurse-medBERT"
outname = "nurse-medBERT"
modelname = "Charangan/MedBERT"


print("--> Loading dataset")
dtrain = Dataset.load_from_disk(f"dataset/{inname}-train.hf")
dtest = Dataset.load_from_disk(f"dataset/{inname}-test.hf")

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
    modelname,
    num_labels=max(dtest["label"])+1
)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    return acc

training_args = TrainingArguments(
    output_dir=f"models/{outname}",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    gradient_accumulation_steps=1,
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
dtest.set_format("torch")
model_fcn = lambda x: trainer.predict(Dataset.from_dict(x)).predictions.astype(np.float16)
create_metrics(model_fcn, dtest, dtest.with_format(type="numpy")["label"], outname, 100000)

# np.savez_compressed("pred_robeczech.npz", y=y)
