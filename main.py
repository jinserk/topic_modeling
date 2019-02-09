from pathlib import Path
import logging
import pickle

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import gensim
from gensim.models import KeyedVectors
from gensim.summarization.textcleaner import clean_text_by_sentences, clean_text_by_word

from lda_model import LDAModelObject, LDAMalletModelObject
from utils import check_dir, download_file


class TextObject:

    def __init__(self, data_dir='./data', dataset='train'):
        self.data_path = Path(data_dir, dataset)
        self.text_file = self.data_path.joinpath(f"20newsgroups_{dataset}.pkl")
        self.id2word_file = self.data_path.joinpath(f"20newsgroups_{dataset}.txt")
        self.corpus_file = self.data_path.joinpath(f"20newsgroups_{dataset}.mm")

        if self.text_file.exists() and self.id2word_file.exists() and self.corpus_file.exists():
            self.load()
        else:
            self.docs = fetch_20newsgroups(subset=dataset, remove=('headers', 'footers', 'quotes')).data
            self.words = self.get_preprocessed_words0(self.docs)
            #self.words = self.get_preprocessed_words1(self.docs)
            #self.words = self.get_preprocessed_words2(self.docs)
            self.id2word, self.corpus = self.build_corpus(self.words)
            self.save()

    def load(self):
        with self.text_file.open('rb') as f:
            texts = pickle.load(f)
            self.docs  = texts['docs']
            self.words = texts['words']
        self.id2word = gensim.corpora.Dictionary.load_from_text(str(self.id2word_file))
        self.corpus = gensim.corpora.MmCorpus(str(self.corpus_file))

    def save(self):
        check_dir(self.data_path)
        texts = { 'docs': self.docs, 'words': self.words }
        with self.text_file.open('wb') as f:
            pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.id2word.save_as_text(str(self.id2word_file))
        gensim.corpora.MmCorpus.serialize(str(self.corpus_file), self.corpus, metadata=True)

    def build_corpus(self, words):
        logging.info("building id2word and corpus ...")
        id2word = gensim.corpora.Dictionary(words)
        corpus = [id2word.doc2bow(text) for text in words]
        return id2word, corpus

    def get_preprocessed_words0(self, docs):
        logging.info(f"original text:\n{docs[0]}")

        # clean up text
        import re
        docs = [re.sub('\s+', ' ', doc) for doc in docs]                # Remove new line characters
        docs = [re.sub('\S*@\S*\s?', '', doc) for doc in docs]          # remove Emails
        docs = [re.sub("\'", "", doc) for doc in docs]                  # Remove distracting single quotes

        # apply gensim's simple_preprocess
        def clean_up_and_tokenize(docs):
            for doc in docs:
                yield gensim.utils.simple_preprocess(doc, deacc=True)   # deacc=True removes punctuations
        words = list(clean_up_and_tokenize(docs))
        logging.info(f"words list after simple preprocesing:\n{docs[0]}")

        self.prepare_stopwords()
        self.prepare_ngram(words)

        words = self.remove_stopwords(words)
        words = self.make_bigrams(words)
        words = self.lemmatize(words)
        logging.info(f"words list after overall cleanup:\n{docs[0]}")

        return words

    def get_preprocessed_words1(self, docs):
        logging.info("cleaning up texts ...")
        # clean up text
        #import re
        # remove Emails
        #docs = [re.sub('\S*@\S*\s?', '', doc) for doc in docs]
        # Remove new line characters
        #docs = [re.sub('\s+', ' ', doc) for doc in docs]
        # Remove distracting single quotes
        #data = [re.sub("\'", "", doc) for doc in data]
        print(docs[0])

        #self.prepare_stopwords()
        #words = self.remove_stopwords(words)

        # strip html tags
        docs = [gensim.parsing.preprocessing.strip_tags(doc) for doc in docs]
        # strip whitespaces
        docs = [gensim.parsing.preprocessing.strip_multiple_whitespaces(doc) for doc in docs]
        # stem
        #docs = [gensim.parsing.preprocessing.stem(doc) for doc in docs]
        # remove stopwords
        docs = [gensim.parsing.preprocessing.remove_stopwords(doc) for doc in docs]

        # apply gensim's simple_preprocess
        def clean_up_and_tokenize(docs):
            for doc in docs:
                yield gensim.utils.simple_preprocess(doc, deacc=True)   # deacc=True removes punctuations
        words = list(clean_up_and_tokenize(docs))

        # apply default filters
        #words = gensim.parsing.preprocessing.preprocess_documents(docs)

        self.prepare_ngram(words)
        words = self.make_bigrams(words)

        # lemmatize docs
        words = self.lemmatize(docs)

        print(words[0])
        return words

    def get_preprocessed_words2(self, docs):
        # tokenize by sentences
        docs = [[s.token for s in clean_text_by_sentences(doc)] for doc in docs]
        #print(docs[0])
        # tokenize by words
        #words = [clean_text_by_word(doc, deacc=True) for doc in docs]

        # lemmatize docs
        words = self.lemmatize(docs)

    def prepare_stopwords(self):
        from nltk.corpus import stopwords
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    def remove_stopwords(self, texts):
        logging.info("removing stopwords ...")
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in self.stop_words]
                for doc in texts]

    def prepare_ngram(self, words):
        logging.info("generating bigram phraser ...")
        bigram  = gensim.models.Phrases(words, min_count=5, threshold=100)
        logging.info("generating trigram phraser ...")
        trigram = gensim.models.Phrases(bigram[words], threshold=100)

        self.bg = gensim.models.phrases.Phraser(bigram)
        self.tg = gensim.models.phrases.Phraser(trigram)
        #print(self.tg[self.bg[words[0]]])

    def make_bigrams(self, texts):
        logging.info("generating bigram words ...")
        return [self.bg[doc] for doc in texts]

    def make_trigrams(self, texts):
        logging.info("generating trigram words ...")
        return [self.tg[self.bg[doc]] for doc in texts]

    def lemmatize(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        logging.info("lemmatizing ...")
        import spacy
        nlp = spacy.load('en', disable=['parser', 'ner'])

        out = []
        for sentences in texts:
            doc = nlp(" ".join(sentences))
            out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return out


class MedicalWord2Vec:

    base_url = "http://evexdb.org/pmresources/vec-space-models/"
    w2v_files = [
        'PMC-w2v.bin',
        'PubMed-and-PMC-ri.tar.gz',
        'PubMed-and-PMC-w2v.bin',
        'PubMed-w2v.bin',
        'wikipedia-pubmed-and-PMC-w2v.bin',
    ]

    def __init__(self, data_dir = './data/w2v'):
        self.w2v_file = self.w2v_files[4]
        self.w2v_path = Path(data_dir, self.w2v_file)
        if not self.w2v_path.exists():
            download_file(self.base_url, self.w2v_file, data_dir)

        self.w2v_model = KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)


class EDWDataset(TextObject):

    src_dir = "/media/samba_share/data/pds/"

    def __init__(self, data_dir="./data/edw_sample"):
        self.data_path = Path(data_dir)
        self.text_file = self.data_path.joinpath(f"edw_sample.pkl")
        self.id2word_file = self.data_path.joinpath(f"edw_sample.txt")
        self.corpus_file = self.data_path.joinpath(f"edw_sample.mm")

        if self.text_file.exists() and self.id2word_file.exists() and self.corpus_file.exists():
            self.load()
        else:
            self.docs = self.fetch_visit_note()
            self.words = self.get_preprocessed_words(self.docs)
            self.id2word, self.corpus = self.build_corpus(self.words)
            self.save()

    def fetch_visit_note(self):
        src_path = Path(self.src_dir)
        if not src_path.exists():
            logging.error(f"error to open {self.src_dir}. check the mount status.")
            sys.exit(1)

        df = pd.DataFrame()
        for xlsx_file in src_path.rglob("sample*.xlsx"):
            tmp = pd.read_excel(xlsx_file, index_col=2)
            df = df.append(tmp, sort=False)

        return df["contents"].dropna().tolist()

    def get_preprocessed_words(self, docs):
        logging.info(f"original text:\n{docs[0]}")

        # clean up text
        import re
        tmp_docs = []
        for doc in docs:
            try:
                doc = gensim.parsing.preprocessing.strip_tags(doc)
                doc = gensim.parsing.preprocessing.strip_multiple_whitespaces(doc)
                doc = gensim.parsing.preprocessing.strip_non_alphanum(doc)
                doc = gensim.parsing.preprocessing.strip_numeric(doc)
                doc = gensim.parsing.preprocessing.strip_short(doc)
                #doc = re.sub('\s+', ' ', doc)           # Remove new line characters
                doc = re.sub('\S*@\S*\s?', '', doc)     # remove Emails
                doc = re.sub("\'", "", doc)             # Remove distracting single quotes
                tmp_docs.append(doc)
            except:
                print(doc)
                breakpoint()
        logging.info(f"cleanup text:\n{tmp_docs[0]}")

        # apply gensim's simple_preprocess
        def clean_up_and_tokenize(docs):
            for doc in docs:
                yield gensim.utils.simple_preprocess(doc, deacc=True)   # deacc=True removes punctuations
        words = list(clean_up_and_tokenize(tmp_docs))
        logging.info(f"words list after simple preprocesing:\n{words[0]}")

        self.prepare_stopwords()
        self.prepare_ngram(words)

        words = self.remove_stopwords(words)
        words = self.make_bigrams(words)
        words = self.lemmatize(words)

        logging.info(f"words list after overall cleanup:\n{words[0]}")
        return words



def visualize(model_obj, txt_obj):
    import pyLDAvis
    import pyLDAvis.gensim
    import matplotlib.pyplot as plt
    #matplotlib inline
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    try:
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(model_obj.model, txt_obj.corpus, txt_obj.id2word)
        vis
    except:
        vis = pyLDAvis.gensim.prepare(model_obj.model, txt_obj.corpus, txt_obj.id2word)
        pyLDAvis.save_html(vis, 'lda_result.html')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Topic modeling using LDA")
    parser.add_argument('--data-dir', default='data', type=str, help="data directory where preprocessed text files are stored")
    parser.add_argument('--model-dir', default='models', type=str, help="model directory where pretrained LDA models are stored")
    parser.add_argument('--model', default='my_lda', type=str, help="model file name in model directory")
    args = parser.parse_args()

    if True:
        t = TextObject(args.data_dir)  # 20_newsgroup dataset in default
        m = LDAMalletModelObject(args.model, t)

        #visualize(m, t)
        result = m.query_topic(t.corpus[:10], t.docs[:10])

        from pprint import pprint
        pprint(result)
    else:
        #c = MedicalWord2Vec()
        v = EDWDataset()

