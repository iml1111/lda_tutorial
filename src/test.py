from trainer import LDATrainer
from evaluator import LDAEvaluator
from train_data import docs

# 코퍼스 생성
lda_trainer = LDATrainer()
lda_trainer.make_corpus(docs)

# 학습했던 모델 및 코퍼스 불러오기
evaluator = LDAEvaluator()
evaluator.load()
evaluator.corpus = lda_trainer.corpus

print("\n학습 결과 추정 지표")
evaluator.evaluate()

print("\n토픽 리스트표")
for i in evaluator.get_topic_list():
	print(i)

print("\n 문서 토픽 분석하기")
token = ['코로나', '바이러스', '기자', '기사']
print(token)
print(evaluator.get_topic(token))

print("\n 토픽 분포 시각화...")
evaluator.visualize()
