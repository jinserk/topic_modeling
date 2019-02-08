# topic_modeling

Topic modeling using LDA model.

## Prerequisites
```
$ pip install -r requirements.txt
$ python -c "import nltk; nltk.download('stopwords')"
$ python -m spacy download en
```
to use Mallet LDA model,
```
$ git submodule update --init mallet
$ cd mallet
$ sudo apt install ant
$ ant
```

## Training
check you have no stored data and models.
If you have `data/train` and `models/mylda` directories, the code do not try to rebuild the corpus and retrain the model.
If you need to rebuild/retrain them, rename or remove the directories.
```
$ python main.py
```

## Inference
TBD

