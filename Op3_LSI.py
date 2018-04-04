import logging
import os
import tempfile
from gensim import corpora
from pprint import pprint  # pretty-printer
from collections import defaultdict
from six import iteritems
from gensim import corpora, models, similarities
import numpy as np
import gensim
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
d2v_model=gensim.models.Doc2Vec.load(os.path.join(MODEL_FOLDER, 'doc2vec.model'))

###################input:
doc = "president donald trump"


print ("Input string:")
print(doc)
query=doc.lower().split();

# vec_bow = dictionary.doc2bow(query)
# vec_lsi = lsi[vec_bow] # convert the query to LSI space
# vec_tfidf=tfidf[vec_bow]


doc_vec=d2v_model.wv[query]
# wor_vec=w2v_model.wv[query]

# print("dddddddddddd")
# pprint(doc_vec)
# print("wwwwwwwwwwww")
# pprint(wor_vec)

sim_docs=d2v_model.wv.most_similar(doc_vec, topn=10)
# sim_words=w2v_model.wv.most_similar(positive=wor_vec, topn=10)
print("Similar words: ")
pprint(sim_docs)


for sim_words in sim_docs:
    if sim_words[1] > 0.5:
        query.append(sim_words[0])


vec_bow = dictionary.doc2bow(query)
vec_lsi = lsi[vec_bow] # convert the query to LSI space
# print(vec_lsi)

index = similarities.MatrixSimilarity.load(os.path.join(MODEL_FOLDER, 'lsi.index'))

sim_in = index[vec_lsi] # perform a similarity query against the corpus


docI=[]
docS=[]

with open('namelist2.txt', 'r') as f:
	filenamelist = f.readlines()
filenamelist = [x.strip() for x in filenamelist] 



i=0
for sim in sim_in:
	if sim > 0.46:
		docI.append(i)
		docS.append(sim)
		# print(sim)
		# print("---------------------------------")
		# with open("./text/"+filenamelist[i+1][0:12]+".txt", "r") as f:
		# 	print(f.read())
	i=i+1

docIsorted = [x for _,x in sorted(zip(docS,docI),reverse=True)]
docSsorted = sorted(docS,reverse=False)
i=0
for d in docIsorted:
	# print(d)
	# print(docSsorted[i])
	i=i+1
	if i>5:
		break;
	print("------------------------------------------------------------------------------------")
	# print("------------------------------------------------------------------------------------")
	with open("./text/"+filenamelist[d][0:12]+".txt", "r") as f:
		print(f.read()[0:500]+"...")
		# print(docSsorted[i])



	# print(documents[l[0]]) # print (document_number, document_similarity) 2-tuples

