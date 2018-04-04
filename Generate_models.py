import zipfile
import json
import re
import logging
import os
import gensim

from gensim import corpora
from pprint import pprint  # pretty-printer
from collections import defaultdict
from six import iteritems
from gensim import corpora, models, similarities
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec


cachedStopWords = stopwords.words("english")
MODEL_FOLDER = "./models"


class MySentences(object):
	def __init__(self, filenamelist):
		self.filenamelist = filenamelist

	def __iter__(self):
		for i in range(1,10):
			docj = json.load(open("./json/"+filenamelist[i]))
			doctext=docj["text"]
			print("\nUnprocessed text:\n")
			print(doctext)
			print("-----------")

			print("\nProcessed text:\n")
			cleandoctext = self.clean_text(doctext)
			s=[]
			s=cleandoctext.lower().split()
			print(cleandoctext.lower())
			yield s #array of each word
			
			# print("**********")
			# print(cleantext)
			# print("-----------------------------------------------------------------")
			# [[['ggf'],['jhjgj']],[],[]]
	def clean_text(self,text):

		text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE) #removing urls in text
		text=text.lower()
		text=text.replace('\n', " ")
		text=text.replace('\xc2\xa0', " ")

		soup = BeautifulSoup(text,"lxml") #remove html
		text = soup.text

		text = re.sub(r'[^a-z]', ' ', text) #remove all symbols and numbers
		
		return text

def remove_f1_and_stop_words(texts):
	# remove words that appear only once
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1

	return [[token for token in text if (frequency[token] > 1 and token not in cachedStopWords)] for text in texts]

def model_w2v(texts):
	bigramer = gensim.models.Phrases(texts)
	# trigramer = gensim.models.Phrases(bigramer[texts])
	# quadgramer = gensim.models.Phrases(trigramer[texts])
	w2v_model = gensim.models.Word2Vec(bigramer[texts], size=100, window=8, min_count=3, workers=4)

	w2v_model.save(os.path.join(MODEL_FOLDER, 'w2v_model-.pkl'))


def model_lsi(texts):

	dictionary = corpora.Dictionary(texts)
	dictionary.save(os.path.join(MODEL_FOLDER, 'dictionary-.dict'))  # store the dictionary, for future reference

	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize(os.path.join(MODEL_FOLDER, 'corpus-.mm'), corpus)  # store to disk, for later use

	tfidf = models.TfidfModel(corpus) 
	corpus_tfidf = tfidf[corpus]
	tfidf.save(os.path.join(MODEL_FOLDER, 'model-.tfidf'))
	
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
	corpus_lsi = lsi[corpus_tfidf] #  bow->tfidf->fold-in-lsi

	lsi.save(os.path.join(MODEL_FOLDER, 'model-.lsi')) 

	index = similarities.MatrixSimilarity(lsi[corpus]) # transforming corpus to LSI space and indexing

	index.save(os.path.join(MODEL_FOLDER, 'lsi-.index'))


	# print("\n\nDictionary:\n")
	# pprint(dictionary.token2id)

	# print("\n\nCorpus:\n")
	# print(corpus)


with open('namelist2.txt', 'r') as f:
	filenamelist = f.readlines()
filenamelist = [x.strip() for x in filenamelist] 
sentences=MySentences(filenamelist)
sentences=remove_f1_and_stop_words(sentences)


# print("\n\nSentences:\n")
# print("\n\nSentences:\n")
# print(sentences[2])

model_w2v(sentences)
# sentences = word2vec.Text8Corpus('text8')
model_lsi(sentences)


print("\n\n\nProcessed text:\n")
print(" ".join(sentences[2]))

# myfile = open("text8","r") 
# data = []
# lin= myfile.readlines()tence s


# # print(lines)print("\n\n\nProcessed text:\n")

#sfor line in lines:
# 	for word in lines[0].split(): 
# 		# data.append([line.strip()])
# 		data.append([word])
# print(data)


# print("------------")
# model_w2v(data)
# sentences = word2vec.Text8Corpus('text8')
# model_lsi(sentences)