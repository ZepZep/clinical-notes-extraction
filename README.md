# Unsupervised clinical notes extraction
Implementation of methods used in the "Unsupervised extraction, labelling and clustering of segments from clinical notes" paper ([preprint](https://arxiv.org/abs/2211.11799)). Published at IEEE BIBM 2022.


## How to run
1. The `make_segments.py` script segments the clinical notes
    * inputs: as we cannot share our dataset, you need to implement your own loading function. The script excepts a pandas dataframe with multiindex `("record_id", "patient_id", "record_number")` and a single column called `"text"`
    * outputs: it outputs two files into the `dataset` folder:
        * `dataset/parts.feather` which contains the individual note segments
        * `dataset/titles.feather` which contains normalized titles and their frequencies
2. You can now train the vector based methods (lsa and doc2vec) using the `train_vectors.py` script. It should automatically read the files from the dataset folder. it outputs predictions into the `pedictions` folder.
3. In order to be able to train Bi-LSTM and RobeCzech models, we need to create a Huggingface dataset using the `make_hf_dataset.py` script. It should automatically read the files from the dataset folder. It creates two folders in the dataset folder: `train.hf` and `test.hf`
4. You can now train Bi-LSTM and RobeCzech models (`train_bilstm.py` and `train_robeczech.py`). They should automatically load the HF dataset. They outputs predictions into the `pedictions` folder.


## Adapting for different datasets / languages
* the segmentation function probably needs to be tweaked to fit your dataset formating (`cut_record` function inside `make_segments.py`)
* title normalization function might need to be tweaked for some languages (`normalize_title` function inside `make_segments.py`)
* you may want to use different tokenization function for the vector methods (`tokenize_doc` function in `train_vectors.py`)
* you may want to use different Huggingface transformer model and tokenizer. Make sure that the model is compatible with the tokenizer.
    * change the `AutoTokenizer` in `make_hf_dataset.py`
    * change the `AutoModelForSequenceClassification` in `train_robeczech.py`

## Interactive visualisation
We use the Tensorboard embedding projector to visualise the vector space of the 2078 extracted titles. It is available [here](https://zepzep.github.io/clinical-notes-extraction/pages/projector/).

The bookmarks (bottom right) contain 3 presets:
* results from clustering
* neighbours of the comorbidities title
* neighbours of the medication title

All of them use 2D T-SNE dimensionality reduction.
