from trainer import LDATrainer
from train_data import docs

lda_trainer = LDATrainer()
lda_trainer.make_corpus(docs)
lda_trainer.train(num_topics=5,
					  passes=30,
					  iterations=10,
					  workers=4)