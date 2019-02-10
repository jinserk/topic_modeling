import sys
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import gensim

from utils import check_dir


class LDAModelObject:

    def __init__(self, model_name, txt_obj, model_dir="./models"):
        self.model_path = Path(model_dir, model_name)
        self.txt_obj = txt_obj

        if not self.model_path.exists():
            check_dir(self.model_path)
            self.model = self.build_model()
            self.save(model_name)
        else:
            self.model = self.load(model_name)

        #self.data_frame = self.format_topics()

    def build_model(self, num_topics_range=(10, 100, 10)):
        coherence_list = []
        model_list = []

        num_topics_list = list(range(*num_topics_range))
        logging.info(f"build an optimized LDA model among num_topics = {num_topics_list}")
        logging.disable(logging.CRITICAL)
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
        logging.disable(logging.NOTSET)

        logging.info(f"(num_topic, coherence) = {[i for i in zip(num_topics_list, coherence_list)]}")
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

    def save(self, prefix):
        prefix = self.model_path.joinpath(prefix)
        self.model.save(str(prefix))

    def load(self, prefix):
        prefix = self.model_path.joinpath(prefix)
        return gensim.models.ldamodel.LdaModel.load(str(prefix))

    def query_topic(self, corpus, docs):
        # txt_obj.corpus should be a list of word list
        topics = [sorted(i[0], key=lambda x: x[1], reverse=True)[0]
                  for i in self.model[corpus]]
        tmp = []
        for i, (topic_num, prop_topic) in enumerate(topics):
            topic_keywords = self.model.show_topic(topic_num)
            tmp.append((topic_num, prop_topic, topic_keywords, docs[i]))
        return tmp


class LDAMalletModelObject(LDAModelObject):

    def __init__(self, model_name, txt_obj, model_dir="./models"):
        #super().__init__(model_name, txt_obj, model_dir)
        self.mallet_path = Path("./mallet/bin/mallet")
        if not self.mallet_path.exists():
            logging.error("no mallet model exists. please download and install it first")
            sys.exit(1)

        self.model_path = Path(model_dir, model_name)
        self.txt_obj = txt_obj

        if not self.model_path.exists():
            check_dir(self.model_path)
            self.model = self.build_mallet_model()
            self.save(model_name)
        else:
            self.model = self.load(model_name)

        #self.data_frame = self.format_topics()

    def build_mallet_model(self, num_topics_range=(10, 100, 10)):
        """ using LDA MALLET model
            http://mallet.cs.umass.edu/download.php
            git clone https://github.com/mimno/Mallet.git
        """
        coherence_list = []
        model_list = []

        num_topics_list = list(range(*num_topics_range))
        logging.info(f"build an optimized LDA model among num_topics = {num_topics_list}")
        logging.disable(logging.CRITICAL)
        for num_topics in num_topics_list:
            model = gensim.models.wrappers.LdaMallet(str(self.mallet_path),
                                                     corpus=self.txt_obj.corpus,
                                                     id2word=self.txt_obj.id2word,
                                                     num_topics=num_topics)
            coherence = self.get_coherence(model, self.txt_obj)
            model_list.append(model)
            coherence_list.append(coherence)
        logging.disable(logging.NOTSET)

        #import matplotlib.pyplot as plt
        #plt.plot(num_topics_list, coherence_list)
        #plt.xlabel("Num Topics")
        #plt.ylabel("Coherence score")
        #plt.savefig("model_optim.jpg")

        logging.info(f"(num_topic, coherence) = {[i for i in zip(num_topics_list, coherence_list)]}")
        idx = np.argmax(coherence_list)
        logging.info(f"Max coherence score = {coherence_list[idx]} at num_topics = {num_topics_list[idx]}")
        return model_list[idx]

    def save(self, prefix):
        prefix = self.model_path.joinpath(prefix)
        self.model.save(str(prefix))

    def load(self, prefix):
        prefix = self.model_path.joinpath(prefix)
        return gensim.models.wrappers.LdaMallet.load(str(prefix))

    def query_topic(self, corpus, docs):
        # txt_obj.corpus should be a list of word list
        topics = [sorted(i, key=lambda x: x[1], reverse=True)[0]
                  for i in self.model[corpus]]
        df = pd.DataFrame()
        for i, (topic_num, prop_topic) in enumerate(topics):
            topic_keywords = self.model.show_topic(topic_num)
            df = df.append(pd.Series([docs[i], topic_num, prop_topic,
                                      ', '.join([k for k, v in topic_keywords])]),
                           ignore_index=True)
        df.columns = ['Docs', 'Topic', 'Topic Prob', 'Keywords']
        return df

    def export_to_excel(self, result, file_name):
        df = pd.DataFrame.from_dict(result)
        df.to_excel(file_name)




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


