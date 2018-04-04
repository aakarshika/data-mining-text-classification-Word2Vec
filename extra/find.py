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
d2v_model=gensim.models.Doc2Vec.load(os.path.join(MODEL_FOLDER, 'doc2vec.model'))

###################input:
doc = "president news"
query=doc.lower().split();

vec_bow = dictionary.doc2bow(query)
vec_lsi = lsi[vec_bow] # convert the query to LSI space
vec_tfidf=tfidf[vec_bow]


doc_vec=d2v_model.wv[query]
wor_vec=w2v_model.wv[query]

# print("dddddddddddd")
# pprint(doc_vec)
# print("wwwwwwwwwwww")
# pprint(wor_vec)

sim_docs=d2v_model.wv.most_similar(doc_vec, topn=10)
sim_words=w2v_model.wv.most_similar(positive=wor_vec, topn=10)

pprint(sim_docs)
pprint(sim_words)

# pprint(d2v_model.wv.similar_by_vector(np.array(sumd, dtype='float32') , topn=10, restrict_vocab=None))
# pprint(w2v_model.wv.most_similar(positive=np.array(w[0], dtype='float32'), topn=10))

# print("reversed:-----------")

# pprint(d2v_model.wv.similar_by_vector(np.array(w[0], dtype='float32') , topn=10, restrict_vocab=None))
# pprint(w2v_model.wv.most_similar(positive=np.array([d], dtype='float32'), topn=10))


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

