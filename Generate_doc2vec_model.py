import zipfile
import json
import re
import logging
import os
import gensim
import gensim

from gensim import corpora
from pprint import pprint  # pretty-printer
from collections import defaultdict
from six import iteritems
from gensim import corpora, models, similarities
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")
MODEL_FOLDER = "./models"


# def model_d2v(texts):
# 	d2v_model = gensim.models.Doc2Vec(texts, size=300, window=8, min_count=5, workers=4)
# 	d2v_model.build_vocab(texts)
# 	d2v_model.save(os.path.join(MODEL_FOLDER, 'd2v_model.pkl'))


class MySentences(object):
	def __init__(self, filenamelist):
		# print("hgfjhgj")
		self.filenamelist = filenamelist
	def __iter__(self):
		for i in range(1,1000):
			print(i)
			docj = json.load(open("./json/"+filenamelist[i]))
			doctext=docj["text"]
			title=docj["title"]
			# print(doctext)
			cleandoctext = self.clean_text(doctext)
			# cleantitle = self.clean_text(title[:35])
			# cleantitle="_".join(cleantitle.split())
			# if cleantitle == "":
			# 	# print("########################################################3problem")
			# 	cleantitle=cleandoctext[:35]
			# cleantitle="_".join(cleantitle.split())
			# # print(cleandoctext)
			# # print("                                ")
			# t=[]
			# t.append(cleantitle)
			# print(t)
			l=gensim.models.doc2vec.LabeledSentence(words=cleandoctext.lower().split() , tags=[i])
			# print(l)
			# print("************")
			yield l
			# print("**********")
			# print(cleantext)
			# print("-----------------------------------------------------------------")
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
	frequency = defaultdict(int)
	for text in texts:
		for token in text[words]:
			frequency[token] += 1

	return [[token for token in text if (frequency[token] > 1 and token not in cachedStopWords)] for text in texts]


with open('namelist2.txt', 'r') as f:
	filenamelist = f.readlines()
filenamelist = [x.strip() for x in filenamelist]
it=MySentences(filenamelist)


model = gensim.models.Doc2Vec(size=100, window=8, min_count=3, workers=4, alpha=0.025, min_alpha=0.025) # use fixed learning rate

model.build_vocab(it)

model.train(it, total_examples=model.corpus_count, epochs=model.iter)

model.save("./models/doc2vec-.model")


