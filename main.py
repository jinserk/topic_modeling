from pathlib import Path
import logging
import pickle

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import gensim
from gensim.summarization.textcleaner import clean_text_by_sentences, clean_text_by_word

from utils import check_dir

logging.basicConfig(format="%(asctime)s [%(levelname)-5s] %(message)s", level=logging.INFO)


class TextObject:

    def __init__(self, data_dir='./data', dataset='train'):
        self.data_path = Path(data_dir, dataset)
        self.words_file = self.data_path.joinpath(f"20newsgroups_{dataset}.pkl")
        self.id2word_file = self.data_path.joinpath(f"20newsgroups_{dataset}.txt")
        self.corpus_file = self.data_path.joinpath(f"20newsgroups_{dataset}.mm")

        if self.words_file.exists() and self.id2word_file.exists() and self.corpus_file.exists():
            self.load()
        else:
            self.docs = fetch_20newsgroups(subset=dataset, remove=('headers', 'footers', 'quotes')).data
            logging.info("cleaning up texts ...")
            self.words = self.get_preprocessed_words0(self.docs)
            #self.words = self.get_preprocessed_words1(self.docs)
            #self.words = self.get_preprocessed_words2(self.docs)
            logging.info("building id2word and corpus ...")
            self.id2word, self.corpus = self.build_corpus(self.words)
            self.save()

    def load(self):
        with self.words_file.open('rb') as f:
            self.words = pickle.load(f)
        self.id2word = gensim.corpora.Dictionary.load_from_text(str(self.id2word_file))
        self.corpus = gensim.corpora.MmCorpus(str(self.corpus_file))

    def save(self):
        check_dir(self.data_path)
        with self.words_file.open('wb') as f:
            pickle.dump(self.words, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.id2word.save_as_text(str(self.id2word_file))
        gensim.corpora.MmCorpus.serialize(str(self.corpus_file), self.corpus, metadata=True)

    def build_corpus(self, words):
        id2word = gensim.corpora.Dictionary(words)
        corpus = [id2word.doc2bow(text) for text in words]
        return id2word, corpus

    def get_preprocessed_words0(self, docs):
        logging.info(f"original text:\n{docs[0]}")

        # clean up text
        import re
        docs = [re.sub('\S*@\S*\s?', '', doc) for doc in docs]  # remove Emails
        docs = [re.sub('\s+', ' ', doc) for doc in docs]        # Remove new line characters
        docs = [re.sub("\'", "", doc) for doc in docs]          # Remove distracting single quotes

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


class LDAModelObject:

    def __init__(self, model_name, txt_obj, model_dir="./models"):
        self.model_path = Path(model_dir, model_name)
        self.txt_obj = txt_obj

        if not self.model_path.exists():
            check_dir(model_dir)
            self.model = self.build_model()
            self.save()
        else:
            self.model = self.load()

    def build_model(self, num_topics_range=(5, 40, 5)):
        coherence_list = []
        model_list = []

        num_topics_list = list(range(*num_topics_range))
        logging.info(f"build an optimized LDA model among num_topics = {num_topics_list}")
        logging.disable()
        for num_topics in num_topics_list:
            model = gensim.models.ldamodel.LdaModel(corpus=self.txt_obj.corpus,
                                                    id2word=self.txt_obj.id2word,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
            coherence = self.get_coherence(model, self.txt_obj)
            model_list.append(model)
            coherence_list.append(coherence)
        logging.enable()

        logging.info(f"(num_topic, coherence) = {[i for i in zip(num_topic_list, coherence_list)]}")
        idx = np.argmax(coherence_list)
        logging.info(f"Max coherence score = {coherence_list[idx]} at num_topics = {num_topics_list[idx]}")
        return model_list[idx]

    def show_model_performance(self):
        perplexity = self.get_perplexity(self.model, self.txt_obj)
        logging.info(f"perplexity: {perplexity}")
        coherence_score = self.get_coherence(self.model, self.txt_obj)
        logging.info(f"coherence score: {coherence_score}")

    def get_perplexity(self, model, txt_obj):
        # a measure of how good the model is. lower the better.
        return model.log_perplexity(txt_obj.corpus)

    def get_coherence(self, model, txt_obj):
        # a measure of how good the model is. higher the better.
        coherence_model = gensim.models.CoherenceModel(model=model, texts=txt_obj.words, dictionary=txt_obj.id2word, coherence='c_v')
        return coherence_model.get_coherence()

    def save(self):
        self.model.save(str(self.model_path))

    def load(self):
        return gensim.models.ldamodel.LdaModel.load(str(self.model_path))

    def format_topics_sentences(texts=data):
        data_frame = pd.DataFrame()
        # Get main topic in each document
        for i, row in enumerate(self.model[self.txt_obj.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in self.model.show_topic(topic_num)])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)



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

    t = TextObject(args.data_dir)  # 20_newsgroup dataset in default
    m = LDAModelObject(args.model, t)

    #visualize(m, t)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.head(10)


