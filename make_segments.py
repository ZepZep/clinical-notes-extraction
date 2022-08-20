from collections import defaultdict
from functools import lru_cache
import re

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unidecode

def cut_record(text):
    lines = text.split("\n")
    record = ""
    nlines = 0
    for line in lines:
        ## split on empty lines, or non continuing lines
        if line.strip() == "":
            if record:
                yield record
                record = ""
                nlines = 0
            continue

        if record and record.rstrip()[-1] == ":":
            record += line.rstrip() + "\n"
            nlines += 1
            continue
        if line[0] in " -•":
            record += line.rstrip() + "\n"
            nlines += 1
            continue
        elif nlines == 1:
            # try split current line
            pass

        if record:
            yield record
        record = line.rstrip() + "\n"
        nlines = 1

    if record:
        yield record
        record = ""
        nlines = 0


def extract_title(text):
    m = re.search(r"^.*?:", text)
    if m:
        return m.group(0)
    return None

def anonymize_text(text):
    m = re.search(r"^.*?:", text)
    if m:
        return text[m.span()[1]:]
    return text

@lru_cache(maxsize=None)
def normalize_title(title):
    if title is None:
        return None
    title = title.rstrip()[:-1].rstrip()
    title = title.lower()
    title = unidecode.unidecode(title)
    title = re.sub(r"[0-9]", "#", title)
    title = re.sub(r"\s+", " ", title)
    return title

def select_good_titles(titles, repeats=10, words=4):
    mask = titles["count"] >= repeats
    mask &= titles.title.str.count(" ") < words
    mask &= ~titles.title.str.contains(",")
    return mask

def get_good_titles(parts, col="stitle"):
    titles = parts[col].value_counts().reset_index()
    titles.columns = ["title", "count"]
    titles = titles[select_good_titles(titles)].reset_index(drop=True)

    tid2t = titles.title.to_dict()
    tid2t = {k+1: v for k,v in tid2t.items()}
    t2tid = {v: k for k,v in tid2t.items()}
    return tid2t, t2tid


def create_parts(records):
    # records should be a dataframe with multiindex (record_id, patient_id, record_number)
    # and a column called text
    it = records.text
    tqdm.pandas(desc='Cutting records')
    parts = it.progress_apply(lambda x: pd.Series(cut_record(x))).stack()
    parts.index.names = ['rid', 'pid', 'rord',  'srord']
    parts.name = "text"
    parts = parts.reset_index()
    parts.index.names = ['srid']

    tqdm.pandas(desc='Anonymizing texts')
    parts["stext"] = parts.text.progress_apply(anonymize_text)

    tqdm.pandas(desc='Extracting titles')
    parts["title"] = parts.text.progress_apply(extract_title)

    tqdm.pandas(desc='Normalizing titles')
    parts["stitle"] = parts.title.progress_apply(normalize_title)

    print("--> Filtering titles")
    tid2t, t2tid = get_good_titles(parts)
    titles = pd.DataFrame({
        "title": tid2t.values(),
        "freq": parts.label.value_counts()[1:].sort_index()
    })

    print("--> Adding labels")
    parts["label"] = parts["stitle"].map(defaultdict(int, t2tid))-1

    return parts, titles


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=512,
        truncation=True,
    )


def load_records():
    raise NotImplemented

# records should be a dataframe with multiindex (record_id, patient_id, record_number)
# and a column called text
records = load_records()
parts, titles = create_parts(records)

parts.reset_index(drop=True).to_feather("dataset/parts.feather")
titles.to_feather("dataset/titles.feather")
