import logging
import os
import tempfile
from gensim import corpora
from pprint import pprint  # pretty-printer
from collections import defaultdict
from six import iteritems
from gensim import corpora, models, similarities
import gensim
import numpy as np
import json

from operator import itemgetter, attrgetter, methodcaller

MODEL_FOLDER = "./models"



# dictionary = corpora.Dictionary.load(os.path.join(MODEL_FOLDER, 'dictionary.dict'))
# corpus = corpora.MmCorpus(os.path.join(MODEL_FOLDER, 'corpus.mm'))
# lsi = models.LsiModel.load(os.path.join(MODEL_FOLDER, 'model.lsi'))
# tfidf =  models.TfidfModel.load(os.path.join(MODEL_FOLDER, 'model.tfidf'))
# w2v_model=gensim.models.Word2Vec.load(os.path.join(MODEL_FOLDER, 'w2v_model.pkl'))
# d2v_model=gensim.models.Doc2Vec.load(os.path.join(MODEL_FOLDER, 'doc2vecI.model'))


dictionary = corpora.Dictionary.load(os.path.join(MODEL_FOLDER, 'dictionary-.dict'))
corpus = corpora.MmCorpus(os.path.join(MODEL_FOLDER, 'corpus-.mm'))
lsi = models.LsiModel.load(os.path.join(MODEL_FOLDER, 'model-.lsi'))
tfidf =  models.TfidfModel.load(os.path.join(MODEL_FOLDER, 'model-.tfidf'))
w2v_model=gensim.models.Word2Vec.load(os.path.join(MODEL_FOLDER, 'w2v_model-.pkl'))
d2v_model=gensim.models.Doc2Vec.load(os.path.join(MODEL_FOLDER, 'doc2vec-.model'))

###################input:
doc = "kim kardashian"

print ("Input string:")
print(doc)
print("\n")
query=doc.lower().split();

vec_bow = dictionary.doc2bow(query)

# print (len(w2v_model.wv.vocab))
# vec_lsi = lsi[vec_bow] # convert the query to LSI space
# print(vec_lsi)
# print(vec_bow)

# print("tfidf scores:")
ts=tfidf[vec_bow]

print("TF-IDF word-score values:")
print(ts)
print("\n")
# print(dictionary[ts[0][0]])


# wordvec=w2v_model.wv[query]
# print("wvector")
# print(wordvec)
sen_vec=[]
for i in range(0,100):
	sen_vec.append(0.0)
# print(sen_vec)
lenq=len(query)

for i in range(0,lenq):

	tfscore=ts[i][1]
	# print(tfscore)

	word=dictionary[ts[i][0]]
	# print(word)

	wordvec=d2v_model.wv[word]
	# print(wordvec)
	td_x_wv=tfscore*wordvec
	# print(td_x_wv)
	for j in range(0,100):
		sen_vec[j]=sen_vec[j]+td_x_wv[j]
		sen_vec[j]=sen_vec[j]/lenq


# pprint(sen_vec)
# pprint(d2v_model.wv[query])


print("Similarity with our sentence vector")
pprint(d2v_model.wv.most_similar(np.array([sen_vec],dtype='float32'), topn=10))

print("\nSimilarity with model query")
pprint(d2v_model.wv.most_similar(query, topn=10))