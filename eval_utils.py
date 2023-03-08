import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

from multiprocessing import Pool
from datasets import Dataset

from sklearn.metrics import classification_report, top_k_accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
import json


def get_every_n_zip(arrays, n, split_half=False):
    size = len(arrays[0])
    for i, a in enumerate(arrays):
        if len(a) != size:
            raise Exception("Non-matching sizes", f"len(a_0) = {size} != {len(a)} = len(a_{i})")

    half = size // 2
    start = 0
    while start < size:
        # print("get_every_n_zip iter")
        if split_half and start < half < start+n:
            yield tuple( a[ start : half ] for a in arrays )
            start = half
        else:
            yield tuple( a[ start : start+n ] for a in arrays )
            start = start + n

def eval_scores(y_true, y_scores, labels):
    return {
        "acc@5":  top_k_accuracy_score(y_true, y_scores, k=5, labels=labels),
        "acc@10": top_k_accuracy_score(y_true, y_scores, k=10, labels=labels),
    }

# def eval_scores(y_true, y_scores, labels):
#     return {
#         "acc@5":  0.5,
#         "acc@10": 0.9,
#     }


def eval_predictions(y_true, y_pred, labels):
    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=False)
    f1s = [cr[f"{i}"]["f1-score"] for i in labels]
    return {
        "acc": cr["accuracy"],
        "MF1": cr["macro avg"]["f1-score"],
        "wF1": cr["weighted avg"]["f1-score"],
    }, f1s

def plot_with_interval(f1s, window=10):
    ylabel = "F1 score"
    xlabel = "Center of title ID bucket"
    data = pd.DataFrame({
        ylabel: f1s,
        xlabel: np.arange(len(f1s)) // window * window + window//2
    })
    maxlabel = data[xlabel].max()
    data.loc[data[xlabel] == maxlabel, xlabel] = (maxlabel - window//2 + len(f1s))//2

    fig, axs = plt.subplots(ncols=2,  sharey=True, gridspec_kw={'width_ratios':[3,1]})
    g = sns.boxplot(data=data, x=xlabel, y=ylabel,  ax=axs[0])
    h = sns.histplot(y=f1s, bins=20, ax=axs[1])
    h.set_xlabel("Number of titles with given F1 score")
    fig.tight_layout()
    # g.set(ylim=(0, 1))
    return fig


subsets = {
    "with": (lambda arrays: tuple(a[:len(a)//2] for a in arrays)),
    "wo": (lambda arrays: tuple(a[len(a)//2:] for a in arrays)),
    "all": (lambda arrays: tuple(arrays))
}


def get_subset_slices(subset_name, arrays):
    return subsets[subset_name](arrays)


def eval_subset(model_name, subset_name, metrics_score, y_test, y_pred, labels):
    metrics_score, y_test, y_pred = get_subset_slices(subset_name, [metrics_score, y_test, y_pred])

    wsum = metrics_score["weight"].sum()
    metrics_score = metrics_score.multiply(metrics_score["weight"]/wsum, axis=0).sum().to_dict()
    del metrics_score["weight"]

    metrics, f1s = eval_predictions(y_test, y_pred, labels)
    metrics.update(metrics_score)

    sns.set(rc={'figure.figsize':(16, 8)})
    fig = plot_with_interval(f1s, window=100)
    fig.savefig(f"metrics/{model_name}-{subset_name}-f1_boxes.pdf")

    return metrics


def create_metrics(model_fcn, x_test, y_test, model_name, batch_size=100000):
    labels = list(range(y_test.max()+1))
    records_scores = []
    y_pred = []

    it = get_every_n_zip([x_test, y_test], batch_size, split_half=True)
    for x, y_true in tqdm(it, desc="Calculating metrics", total=len(x_test)//batch_size+1):
        # print(x)
        # print(y_true)
        y_scores = model_fcn(x)
        y_pred.append(y_scores.argmax(axis=1))
        metrics = eval_scores(y_true, y_scores, labels)
        metrics["weight"] = len(x) /  len(x_test)
        records_scores.append(metrics)
        del y_scores

    metrics_score = pd.DataFrame.from_records(records_scores)
    y_pred = np.concatenate(y_pred, axis=0)

    metrics = {}

    for subset_name in subsets.keys():
        cur = eval_subset(model_name, subset_name, metrics_score, y_test, y_pred, labels)
        metrics[subset_name] = cur

    with open(f"metrics/{model_name}-metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    np.savez_compressed(f"metrics/{model_name}-predictions.npz", y=y_pred)


