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

# dictionary = corpora.Dictionary.load(os.path.join(MODEL_FOLDER, 'dictionary-.dict'))
# corpus = corpora.MmCorpus(os.path.join(MODEL_FOLDER, 'corpus-.mm'))
# lsi = models.LsiModel.load(os.path.join(MODEL_FOLDER, 'model-.lsi'))
# tfidf =  models.TfidfModel.load(os.path.join(MODEL_FOLDER, 'model-.tfidf'))
# w2v_model=gensim.models.Word2Vec.load(os.path.join(MODEL_FOLDER, 'w2v_model-.pkl'))
# d2v_model=gensim.models.Doc2Vec.load(os.path.join(MODEL_FOLDER, 'doc2vec-.model'))



dictionary = corpora.Dictionary.load(os.path.join(MODEL_FOLDER, 'dictionary.dict'))
corpus = corpora.MmCorpus(os.path.join(MODEL_FOLDER, 'corpus.mm'))
lsi = models.LsiModel.load(os.path.join(MODEL_FOLDER, 'model.lsi'))
tfidf =  models.TfidfModel.load(os.path.join(MODEL_FOLDER, 'model.tfidf'))
w2v_model=gensim.models.Word2Vec.load(os.path.join(MODEL_FOLDER, 'w2v_model.pkl'))
d2v_model=gensim.models.Doc2Vec.load(os.path.join(MODEL_FOLDER, 'doc2vecI.model'))

###################input:
doc = "crime"



print ("Input string:")
print(doc)
query=doc.lower().split();

res_vec=d2v_model.infer_vector(doc_words=query, steps=20, alpha=0.025)
print("Inferred vector from doc2vec model:")
print (res_vec[0:12])


with open('namelist2.txt', 'r') as f:
	filenamelist = f.readlines()
filenamelist = [x.strip() for x in filenamelist] 

print("\nQuerying resultant vector in doc2vec as-")
k=d2v_model.wv.most_similar(positive=[res_vec],topn=10)

print("\tWord vector:")
pprint(k)

k=d2v_model.docvecs.most_similar(positive=[res_vec],topn=10)

print("\n\tDocument vector:")
pprint(k)

print("\nDOCUMENT RESULTS:")
i=0
for kk in k:
	i=i+1
	if kk[1]>0.5:
		pprint(kk[0])
		print("----------------------------------------------------------------------------")
		with open("./text/"+filenamelist[kk[0]-1][0:12]+".txt", "r") as f:
			print((f.read()[0:500])+". . .")
	# pprint()



docI=[]
docS=[]
# i=0
# for sim in doc_list:
# 	if sim > 0.05:
# 		docI.append(i)
# 		docS.append(sim)
# 		print("File number: ")
# 		print(i)
# 		print("---------------------------------")
# 		with open("./text/"+filenamelist[i+1][0:12]+".txt", "r") as f:
# 			print(f.read())
# 	i=i+1

# docIsorted = [x for _,x in sorted(zip(docS,docI),reverse=True)]
# docSsorted = sorted(docS,reverse=True)
# i=0
# for d in docIsorted:
# 	print(d)
# 	print(docSsorted[i])
# 	i=i+1
# 	with open("./text/"+filenamelist[d][0:12]+".txt", "r") as f:
# 		pprint(f.read())

