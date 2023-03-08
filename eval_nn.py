import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from eval_utils import create_metrics


inname = "nurse-medBERT"
outname = "nurse-medBERT"
modelname = "medbert"

print("--> Loading dataset")
dtest = Dataset.load_from_disk(f"dataset/{inname}-test.hf")


print("--> Loading model")
model = AutoModelForSequenceClassification.from_pretrained(
    modelname,
    # local_files_only=True,
    # num_labels=max(dtest["label"])+1
)


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
)


print("--> Prediction")
dtest.set_format("torch")
model_fcn = lambda x: trainer.predict(Dataset.from_dict(x)).predictions.astype(np.float16)
create_metrics(model_fcn, dtest, dtest.with_format(type="numpy")["label"], outname, 100000)
