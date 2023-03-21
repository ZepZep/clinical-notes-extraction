from collections import defaultdict
from functools import lru_cache
import re

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unidecode

import os
MIMIC_DATA = os.environ.get("AICOPE_SCRATCH") + "/datasets/physionet.org/files/mimiciii/1.4"

def subsplit(text):
    l = re.split(r"\n(.{1,30}:)(?![0-9])", text)
    if len(l) == 1:
        yield text
        return
    if l[0]:
        yield l[0]
    for i in range(1, len(l), 2):
        yield l[i] + l[i+1]

def cut_record(text):
    top_split_pattern = r"\n\n|\n ?__+\n"
    for part in re.split(top_split_pattern, text):
        part = re.sub(r"^\s*\[\*\*[0-9\-]*\*\*\]\s+([0-9]{4}|[0-9]{1,2}:[0-9]{1,2} (PM|AM))", "", part)
        part = re.sub(r" +FINAL REPORT\n", "", part)
        part = part.strip()
        if not part:
            continue
        yield from subsplit(part)


def get_title(text):
    m = re.search(r"^(.*?)(?:\:|\.{3,4})(?![0-9])", text)
    if not m:
        return None, text
    l, r = m.span()
    title = m.group(1).strip()
    body = text[r:].strip()
    return title, body

def extract_and_normalize(text):
    title, body = get_title(text)
    return body, title, normalize_title(title)

# def extract_title(text):
#     m = re.search(r"^.*?:", text)
#     if m:
#         return m.group(0)
#     return None

# def anonymize_text(text):
#     m = re.search(r"^.*?:", text)
#     if m:
#         return text[m.span()[1]:]
#     return text

# @lru_cache(maxsize=None)
def normalize_title(title):
    if title is None:
        return None
    title = re.sub(r"\s+", " ", title)
    title = title.strip()
    title = title.lower()
    title = unidecode.unidecode(title)
    title = re.sub(r"[0-9]", "9", title)
    return title

def select_good_titles(titles, repeats=20, words=6):
    mask = titles["count"] >= repeats
    mask &= titles["title"].str.len() > 0
    mask &= titles["title"].str.count(" ") < words
    mask &= ~titles["title"].str.contains(",")
    return mask

def get_good_titles(parts, col="stitle"):
    titles = parts[col].value_counts().reset_index()
    titles.columns = ["title", "count"]
    titles = titles[select_good_titles(titles)].reset_index(drop=True)

    tid2t = titles.title.to_dict()
    tid2t = {k+1: v for k,v in tid2t.items()}
    t2tid = {v: k for k,v in tid2t.items()}
    return tid2t, t2tid


def filter_parts(parts, min_stext_length=10):
    """Remove segments with little context"""
    mask = parts["stext"].str.len() >= min_stext_length
    return parts[mask].reset_index(drop=True)


def create_parts(records):
    # records should be a dataframe with multiindex (record_id, patient_id, record_number)
    # and a column called text
    it = records.text
    tqdm.pandas(desc="> Cutting records")
    parts = it.progress_apply(lambda x: pd.Series(cut_record(x))).stack()
    parts.index.names = ['rid', 'pid', 'rord',  'srord']
    parts.name = "text"
    parts = parts.reset_index()

    tqdm.pandas(desc="> Extract and normalize")
    derived_columns = pd.DataFrame.from_records(
        parts["text"].progress_apply(extract_and_normalize),
        columns=["stext", "title", "stitle"]
    )
    parts = pd.concat([parts, derived_columns], axis=1)

    print("> Filtering segments")
    parts = filter_parts(parts)
    parts.reset_index(inplace=True, drop=True)

    print("> Filtering titles")
    tid2t, t2tid = get_good_titles(parts)

    print("> Adding labels")
    parts["label"] = parts["stitle"].map(defaultdict(int, t2tid))-1

    titles = pd.DataFrame({
        "title": tid2t.values(),
        "freq": parts.label.value_counts().iloc[1:].sort_index()
    })

    return parts, titles


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=512,
        truncation=True,
    )



def load_records(cutoff=None):
    types = {
        'CHARTDATE': pd.StringDtype(),
        'CHARTTIME': pd.StringDtype(),
        'STORETIME': pd.StringDtype(),
        'CATEGORY': pd.StringDtype(),
        'DESCRIPTION': pd.StringDtype(),
        'ISERROR': pd.StringDtype(),
        'TEXT': pd.StringDtype()
    }
    good_categories = {
        'Nursing/other': 11, # 10
        'Radiology': 9,
        'Nursing': 6, # mostly Action, response, plan
        'ECG': 0,
        'Physician ': 10, # 10
        'Discharge summary': 10,
        'Echo': 10,
        'Respiratory ': 10,
        'Nutrition': 9,
        'General': 8,
        'Rehab Services': 9,
        'Social Work': 8, # no good titles
        'Case Management ': 5, # Action, response, plan
        'Pharmacy': 4, # assesment, recommanation
        'Consult': 10,
    }

    print("> Loading dataset")

    notes = pd.read_csv(f"{MIMIC_DATA}/NOTEEVENTS.csv.gz", dtype=types, nrows=cutoff)
    stats = pd.DataFrame({
        "count": notes["CATEGORY"].sort_index(inplace=False).value_counts(),
        "goodness": good_categories
    }).sort_values(["goodness", "count"], ascending=[False, False])
    print("  > loaded")

    notes["CHARTDATE"] = pd.to_datetime(notes["CHARTDATE"])
    notes = notes.sort_values(["SUBJECT_ID", "CHARTDATE"])
    print("  > sorted")

    note_relevance = notes["CATEGORY"].isin(stats.query("goodness == 11").index)
    notes = notes[note_relevance]
    print("  > filtered for relevant categories")
    if len(notes) == 0:
        raise Exception("Filtering removed all notes")

    notes = notes.groupby('SUBJECT_ID', group_keys=False).apply(lambda group: group.assign(record_number=range(len(group))))
    print("  > grouped")

    notes = notes[["ROW_ID", "SUBJECT_ID", "record_number", "TEXT"]]
    notes = notes.rename(columns={"ROW_ID": "rid", "SUBJECT_ID": "pid", "record_number": "rord", "TEXT": "text"})
    notes = notes.set_index(["rid", "pid", "rord"])
    return notes

def create_name(pre, name, post):
    if name:
        return f"{pre}{name}-{post}"
    return f"{pre}{post}"

# records should be a dataframe with multiindex (record_id, patient_id, record_number)
# and a column called text
cutoff = None
# cutoff = 1000
name = "c_nurse"
# name = None

records = load_records(cutoff)
parts, titles = create_parts(records)

parts.reset_index(drop=True).to_feather(create_name("dataset/", name, "parts.feather"))
titles.to_feather(create_name("dataset/", name, "titles.feather"))
