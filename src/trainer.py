import re
from gensim.test.utils import datapath
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim

class LDATrainer:
    """docstring for LDAModelHandler"""
    def __init__(self):
        # 학습에 사용될 코퍼스 데이터
        self.corpus = []
        # 정수 인코딩 딕셔너리
        self.dictionary = None
        # 모델 객체 저장 변수
        self.model = None
        # 전처리를 위한 정규표현식 패턴
        self.preproc_pattern = re.compile('[^ ㄱ-ㅣ가-힣|a-z|0-9|:]+')
        # 이모지 문자 제거를 위한 패턴
        self.emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           "]+", flags=re.UNICODE)

    def make_corpus(self, docs):
        corpus = []
        dictionary = corpora.Dictionary()
        # 문서 전처리 과정
        for doc in docs:
            doc = doc.strip()
            doc = doc.lower()
            doc = self.emoji_pattern.sub(r' ', doc)
            doc = self.preproc_pattern.sub(r' ', doc)
            doc = re.sub(r'\s+', ' ', doc)
            tokens = doc.split()
            dictionary.add_documents([tokens])
            corpus += [tokens]

        # 코퍼스 정수 인코딩
        corpus = [dictionary.doc2bow(doc) for doc in corpus]

        # tfidf transform 적용
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]

        self.corpus = corpus
        self.dictionary = dictionary


    def train(self, num_topics, passes, iterations, workers):
        ldamodel = LdaMulticore(self.corpus,
                                num_topics=num_topics,
                                id2word=self.dictionary,
                                passes=passes,
                                workers=workers,
                                iterations=iterations)
        self.model = ldamodel



