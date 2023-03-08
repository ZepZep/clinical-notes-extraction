import pandas as pd
import numpy as np
from itertools import product
from tqdm.auto import tqdm

from datasets import Dataset
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.callbacks import Callback

from eval_utils import create_metrics


inname = "nurse-medBERT"
outname = "nurse-medBERT-biLSTM"
cut=150
modelname = f"models/{outname}"
vocab_size = 28996
n_labels = 2297
dropout = 0.1

print("--> Loading dataset")
def load_ds(path):
    ds = Dataset.load_from_disk(path)
    ds.set_format("numpy")
    return ds["input_ids"], ds["label"]

xt, yt = load_ds(f"dataset/{inname}-train.hf")
xv, yv = load_ds(f"dataset/{inname}-test.hf")

xt = xt[:, 1:cut+1]
xv = xv[:, 1:cut+1]


print("--> Building NN")

def build_network(vocab=vocab_size, labels=n_labels):
    inptW = Input(shape=(cut,))
    embedding = Embedding(input_dim=vocab, output_dim=128,
                         input_length=cut, mask_zero=True)
    embW = embedding(inptW)
    embW = Dropout(dropout)(embW)

    # biLSTM
    bilstm1 = Bidirectional(
      LSTM(units=300, return_sequences=False, recurrent_dropout=0.1)
    )(embW)
    bilstm1 = Dropout(dropout)(bilstm1)

    out = Dense(labels, activation="softmax")(bilstm1)

    # build and compile model
    model = Model(inptW, out)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

model = build_network()
model.summary()


print("\n--> Training")

class SaveCallback(Callback):
   def on_epoch_end(self, *args):
      print("\nsaving model")
      model.save(modelname)

model.fit(
    xt, yt, batch_size=512, epochs=10,
    validation_data=(xv,yv),
    verbose=1,
    callbacks=[SaveCallback()],
)

model.save(modelname)

print("\n--> Predicting")
model_fcn = lambda x: model.predict(x)
create_metrics(model_fcn, xv, yv, outname, 100000)
