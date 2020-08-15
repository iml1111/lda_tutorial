import numpy as np
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim

class LDAEvaluator:
    '''LDA model Evaluator'''
    def __init__(self):
        self.model = None
        self.dictionary = None
        self.corpus = None

    def load(self, model_path="lda_model", dict_path="./lda_dict"):
        '''사전에 학습한 모델을 불러옵니다, 모델 및 딕셔너리 필요'''
        self.model = models.ldamodel.LdaModel.load(model_path)
        self.dictionary = corpora.Dictionary.load(dict_path)

    def evaluate(self):
        '''모델의 성능 지표 2가지를 측정하여 출력합니다'''
        cm = CoherenceModel(model=self.model,
                            corpus=self.corpus,
                            coherence="u_mass")
        coherence = cm.get_coherence()
        perplexity = self.model.log_perplexity(self.corpus)
        print("Coherence:", coherence)
        print('Perplexity:', perplexity)

    def get_topic_list(self, num=5):
        '''
        해당 모델이 표현하고 있는 주제 리스트를 출력합니다.
        각 주제들의 이름을 알 수는 없지만 어떤 단어들이 주제를 이루고 있는지,
        그리고 각각의 단어들의 성분이 어떻게 되는지 간단하게 알아볼 수 있습니다.
        '''
        topics = self.model.print_topics(
                                    num_topics=-1,
                                    num_words=num)
        return topics

    def get_topic(self, tokens):
        corpus = self.dictionary.doc2bow(tokens)
        #corpus = [self.dictionary.doc2bow(token) for token in tokens]
        for temp in self.model[corpus]:
            print(temp)

    def is_in_dict(self, words):
        if type(words) is str: words = [words]
        temp = self.dictionary.doc2idx(words)
        result=[]
        for i in temp:
            if i == -1: result += [False]
            else: result += [True]
        if len(words) == 1: return result[0]
        else: return result

    def visualize(self, name="test"):
        vis = pyLDAvis.gensim.prepare(self.model,
                                      self.corpus,
                                      self.dictionary)
        pyLDAvis.save_html(vis, name + ".html")

