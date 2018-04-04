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

dictionary = corpora.Dictionary.load(os.path.join(MODEL_FOLDER, 'dictionary.dict'))
corpus = corpora.MmCorpus(os.path.join(MODEL_FOLDER, 'corpus.mm'))
lsi = models.LsiModel.load(os.path.join(MODEL_FOLDER, 'model.lsi'))

tfidf =  models.TfidfModel.load(os.path.join(MODEL_FOLDER, 'model.tfidf'))
w2v_model=gensim.models.Word2Vec.load(os.path.join(MODEL_FOLDER, 'w2v_model.pkl'))

###################input:
doc = "earthquake near city"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
# print(vec_lsi)
# print(vec_bow)

# print("tfidf scores:")
ts=tfidf[vec_bow]
# print(dictionary[ts[0][0]])

query=doc.lower().split();

wordvec=w2v_model.wv[query]
# print("wvector")
# print(wordvec)
sen_vec=[0,0,0,0,0,0,0,0,0,0]
# for i in range(0,3):
# 	tfscore=ts[i][1]
# 	print(tfscore)
# 	word=dictionary[ts[i][0]]
# 	print(word)
# 	wordvec=w2v_model.wv[word]
# 	print(wordvec)

# 	td_x_wv=tfscore*wordvec
# 	print(td_x_wv)

# 	for j in range(0,10):
# 		sen_vec[j]=sen_vec[j]+td_x_wv
# 	for j in range(0,10):
# 		sen_vec[j]=sen_vec[j]/3

d2v_model=gensim.models.Doc2Vec.load('./models/doc2vec.model')

s="water transport"

vec_bow = dictionary.doc2bow(s.lower().split())
ts=tfidf[vec_bow]

d=d2v_model.wv[s.split()]
w=w2v_model.wv[s.split()]*ts[0][1]

print("dddddddddddd")
pprint(d)
sumd=d[0]+d[1]
print("wwwwwwwwwwww")
pprint(sumd)

pprint(d2v_model.wv.similar_by_vector(np.array(sumd, dtype='float32') , topn=10, restrict_vocab=None))
pprint(w2v_model.wv.most_similar(positive=np.array(w[0], dtype='float32'), topn=10))

print("reversed:-----------")

pprint(d2v_model.wv.similar_by_vector(np.array(w[0], dtype='float32') , topn=10, restrict_vocab=None))
pprint(w2v_model.wv.most_similar(positive=np.array([d], dtype='float32'), topn=10))


# print(d2v_model.infer_vector(['only', 'you', 'can', 'prevent', 'forrest', 'fires']))
# print(d2v_model.docvecs['only', 'you', 'can', 'prevent', 'forrest', 'fires'])
# doc_vec=d2v_model["career"]



# for sim_words in sim:
#     if sim_words[1] > 0.0:
#         query.append(sim_words[0])


# vec_bow = dictionary.doc2bow(query)
# vec_lsi = lsi[vec_bow] # convert the query to LSI space
# # print(vec_lsi)

# index = similarities.MatrixSimilarity.load(os.path.join(MODEL_FOLDER, 'lsi.index'))



# sims = index[vec_lsi] # perform a similarity query against the corpus
# i=0
# # for l in sims:

# # 	print(l) # print (document_number, document_similarity) 2-tuples
# # 	print(i)
# # 	i=i+1
# # 	# print(documents[l[0]]) # print (document_number, document_similarity) 2-tuples

