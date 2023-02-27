import pandas as pd
import numpy as np
from collections import defaultdict, namedtuple
import re
from tqdm.auto import tqdm

from nltk.tokenize import word_tokenize

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
# import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from eval_utils import create_metrics

tqdm.pandas()

tf.keras.backend.clear_session()

dim=50
hidden_size = 64
inname = "nurse"
outname = "nurse_nltk"
per_category_limit = None

def limit_samples(df, group, max_count):
    return df.groupby(group).apply(lambda x: x if len(x) <= max_count else x.sample(max_count)).droplevel(0)


print("--> Loading Dataset")
parts = pd.read_feather(f"dataset/{inname}-parts.feather")
titles = pd.read_feather(f"dataset/{inname}-titles.feather")

relevant = parts
if per_category_limit is not None:
    relevant = limit_samples(parts, "label", per_category_limit)
relevant = relevant.query("label >= 0").reset_index(drop=True)


print("--> Prepairing train/test")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(relevant, relevant.label):
    break

def tokenize_doc(text):
    text = re.sub(r"[0-9]", "9", text)
    text = re.sub(r"([\.\,\:])(?!#)", r" \1 ", text)
    text = re.sub(r"\n", r" <br> ", text)
    return text.split()

def tokenize_nltk(text):
    text = re.sub(r"[0-9]", "9", text)
    text = re.sub(r"\n", r" <br> ", text)
    return word_tokenize(text)

tokenize = tokenize_nltk

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
    "train": make_dataset(relevant, train_index, tokenize),
    "test":  make_dataset(relevant, test_index, tokenize),
}


print("--> Vector Models")
# make LSA vectors
LSAResult = namedtuple("LSAModel", ["vectorizer", "decomposer"])

def vectorize_LSA(ds, dim):
    print("----> TF-IDF vectorizing")
    vectorizer = TfidfVectorizer(lowercase=False, min_df=3)
    tqdm.pandas(desc=f'Joining tokens')
    docs = ds["train"].doc.progress_apply(" ".join)
    x = vectorizer.fit_transform(docs)

    print("----> SVD Decomposition")
    svd = TruncatedSVD(n_components=dim, n_iter=5, random_state=42)
    # vectors_train = svd.fit_transform(x)
    vectors_train = svd.fit_transform(x)

    print("----> Infering vectors")
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
    model = Doc2Vec(
        x.to_list(), vector_size=dim, window=window,
        min_count=min_count, workers=workers, epochs=epochs,
        callbacks=[tqdmcb]
    )

    tqdm.pandas(desc=f'Infering train vectors')
    vectors_train = ds["train"].doc.progress_apply(model.infer_vector)

    tqdm.pandas(desc=f'Infering test vectors')
    vectors_test = ds["test"].doc.progress_apply(model.infer_vector)
    return np.vstack(vectors_train), np.vstack(vectors_test), model


print("--> Training LSA Vectorizer")
vectors_lsa_train, vectors_lsa_test, model_lsa = vectorize_LSA(ds, dim)


print("--> Training Doc2Vec Vectorizer")
vectors_d2v_train, vectors_d2v_test,  model_d2v = vectorize_d2v(ds, dim)
np.save(f"predictions/{outname}-d2v_titles_sample.npy", model_d2v.dv.vectors)


print("--> Defining Classification NN")
def make_model(vectors, n_titles, dropout=0.0):
    lin = tf.keras.Input(shape=vectors[0].shape, name="input")
    x = lin
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_size, activation="relu", name="hidden")(x)
    x = layers.Dropout(dropout)(x)
    lout = layers.Dense(n_titles, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(lin, lout)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    return model


print("--> Training LSA Classifier")
nn_lsa = make_model(vectors_lsa_train, len(titles), dropout=0.0)
nn_lsa.summary()

nn_lsa.fit(
    x=vectors_lsa_train, y=ds["train"].label,
    batch_size=512,
    epochs=10,
    validation_split=0.1,
)

print("--> Making LSA Predictions")
model_fcn = lambda x: nn_lsa.predict(x, batch_size=512, verbose=0)
create_metrics(model_fcn, vectors_lsa_test, ds["test"]["label"], f"{outname}-lsa", 100000)

print("--> Training Doc2Vec Classifier")
nn_d2v = make_model(vectors_d2v_train, len(titles), dropout=0.0)
nn_d2v.summary()

nn_d2v.fit(
    x=np.stack(vectors_d2v_train), y=ds["train"].label,
    batch_size=512,
    epochs=10,
    validation_split=0.1,
)

print("--> Making Doc2Vec Predictions")
model_fcn = lambda x: nn_d2v.predict(x, batch_size=512, verbose=0)
create_metrics(model_fcn, vectors_d2v_test, ds["test"]["label"], f"{outname}-d2v", 100000)
