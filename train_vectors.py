import pandas as pd
import numpy as np
from collections import defaultdict, namedtuple
import re
from tqdm.auto import tqdm

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tqdm.pandas()


dim=50
hidden_size = 64


print("--> Loading Dataset")
parts = pd.read_feather("dataset/parts.feather")
titles = pd.read_feather("dataset/titles.feather")
relevant = parts.query("label >= 0").reset_index(drop=True)


print("--> Prepairing train/test")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(relevant, relevant.label):
    break

def tokenize_doc(text):
    text = re.sub(r"[0-9]", "#", text)
    text = re.sub(r"([\.\,\:])(?!#)", r" \1 ", text)
    text = re.sub(r"\n", r" <br> ", text)
    return text.split()

def make_dataset(relevant, indexer, tokenize_function, numproc=12):
    df = pd.concat([
        pd.DataFrame({
            "text": relevant.text[indexer],
            "label": relevant.label[indexer],
        }),
        pd.DataFrame({
            "text": relevant.stext[indexer],
            "label": relevant.label[indexer],
        })
    ]).reset_index(drop=True)

    tqdm.pandas(desc=f'Tokenizing')
    df["doc"] = df.text.progress_apply(tokenize_function)
    return df

ds = {
    "train": make_dataset(relevant, train_index, tokenize_doc),
    "test":  make_dataset(relevant, test_index, tokenize_doc),
}


print("--> Vector Models")
# make LSA vectors
LSAResult = namedtuple("LSAModel", ["vectorizer", "decomposer"])

def vectorize_LSA(ds, dim):
    print("TF-IDF vectorizing")
    vectorizer = TfidfVectorizer(lowercase=False, min_df=3)
    tqdm.pandas(desc=f'Joining tokens')
    docs = ds["train"].doc.progress_apply(" ".join)
    x = vectorizer.fit_transform(docs)

    print("SVD Decomposition")
    svd = TruncatedSVD(n_components=dim, n_iter=5, random_state=42)
    # vectors_train = svd.fit_transform(x)
    vectors_train = svd.fit_transform(x)

    print("Infering vectors")
    docs = ds["test"].doc.progress_apply(" ".join)
    vectors_test = svd.transform(vectorizer.transform(docs))

    return vectors_train, vectors_test, LSAResult(vectorizer, svd)

# make Doc2Vec vectors
class TqdmProgress(CallbackAny2Vec):
    def __init__(self, total, inc=1, **kwargs):
        self.pbar = tqdm(total=total, **kwargs)
        self.inc = inc
    def on_epoch_end(self, model):
        self.pbar.update(self.inc)
    def on_train_end(self, model):
        self.pbar.close()

def make_tagged_document(row):
    return TaggedDocument(row["doc"], [row["label"]])

def vectorize_d2v(ds, dim, window=5, min_count=5, workers=4, epochs=10):
    tqdm.pandas(desc=f'Tagging documments')
    x = ds["train"].progress_apply(make_tagged_document, axis=1)
    tqdmcb = TqdmProgress(epochs, desc="Training doc2vec, epoch")
    model = Doc2Vec(x.to_list(), vector_size=dim, window=window,
                    min_count=min_count, workers=workers, epochs=epochs,
                    callbacks=[tqdmcb]
                   )

    tqdm.pandas(desc=f'Infering test vectors')
    vectors_train = ds["train"].doc.progress_apply(model.infer_vector)

    tqdm.pandas(desc=f'Infering test vectors')
    vectors_test = ds["test"].doc.progress_apply(model.infer_vector)
    return vectors_train, vectors_test, model


print("--> Training LSA Vectorizer")
vectors_lsa_train, vectors_lsa_test, model_lsa = vectorize_LSA(ds, dim)


print("--> Training Doc2Vec Vectorizer")
vectors_d2v_train, vectors_d2v_test,  model_d2v = vectorize_d2v(ds, dim)
np.save("predictions/d2v_titles.npy", model_d2v.dv.vectors)

tqdm.pandas(desc=f'Infering test vectors')
vectors_d2v_train = ds["train"].doc.progress_apply(model_d2v.infer_vector)

tqdm.pandas(desc=f'Infering test vectors')
vectors_d2v_test = ds["test"].doc.progress_apply(model_d2v.infer_vector)


print("--> Defining Classification NN")
def make_model(vectors, n_titles, dropout=0.0):
    lin = tf.keras.Input(shape=vectors[0].shape, name="input")
    x = lin
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_size, activation="relu", name="hidden")(x)
    x = layers.Dropout(dropout)(x)
    lout = layers.Dense(n_titles, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(lin, lout)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    return model


print("--> Training LSA Classifier")
nn_lsa = model = make_model(vectors_lsa_train, len(t2tid), dropout=0.0)
nn_lsa.summary()

nn_lsa.fit(
    x=vectors_lsa_train, y=ds["train"].label,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
)

print("--> Making LSA Predictions")
pred_lsa = nn_lsa.predict(vectors_lsa_test)
np.savez_compressed("predictions/pred_lsa.npz", y=pred_lsa.astype(np.float16))


print("--> Training Doc2Vec Classifier")
nn_d2v = model = make_model(vectors_d2v_train, len(t2tid), dropout=0.0)
nn_d2v.summary()

nn_d2v.fit(
    x=np.stack(vectors_d2v_train), y=ds["train"].label,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
)

print("--> Making Doc2Vec Predictions")
pred_d2v = nn_d2v.predict(np.stack(vectors_d2v_test))
np.savez_compressed("predictions/pred_d2v.npz", y=pred_d2v.astype(np.float16))
