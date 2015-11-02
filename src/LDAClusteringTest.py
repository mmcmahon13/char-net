
# coding: utf-8

# In[74]:

from __future__ import division
from collections import defaultdict

import os
import random
import json

json_filename = "C:\\Users\\Molly\\Google Drive\\senior classes\\nlp\\term_project\\book-nlp-master\\book-nlp-master\\data\\output\\rowling2\\book.id.book"
#json_filename = "C:\\Users\\Molly\\Google Drive\\senior classes\\nlp\\term_project\\book-nlp-master\\book-nlp-master\\data\\output\\dickens\\book.id.book"
#json_filename = "C:\\Users\\Molly\\Google Drive\\senior classes\\nlp\\term_project\\book-nlp-master\\book-nlp-master\\data\\output\\twain\\book.id.book"


with open(json_filename) as json_file:
	char_json = json.load(json_file)
print "successfully loaded json"


# In[75]:

characters = {}

for character in char_json["characters"]:
	char_name = character["names"][0]['n']
	characters[char_name] = []
	for word_dict in character["patient"]:
		characters[char_name].append(word_dict['w'] + "_PATIENT")
	for word_dict in character["agent"]:
		characters[char_name].append(word_dict['w'] + "_AGENT")
	for word_dict in character["mod"]:
		characters[char_name].append(word_dict['w'] + "_MOD")
	for word_dict in character["poss"]:
		characters[char_name].append(word_dict['w'] + "_POSS")

texts = [None for i in range(len(characters))]
names = defaultdict(float)
i = 0
for key, value in characters.iteritems():
	print str(key) + ": " + str(value)
	texts[i] = value
	names[key] = i
	i += 1
	print


# In[105]:

from gensim import corpora, models, similarities 
 
# create Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

# create bag of words model to perform analysis
corpus = [dictionary.doc2bow(text) for text in texts]
numtopics = 9

lda = models.LdaModel(corpus, num_topics=numtopics, 
                            id2word=dictionary, 
                            update_every=5, 
                            chunksize=100, 
                            passes=100)


# In[106]:

print(lda.show_topics())


# In[107]:

import numpy as np

# topics_matrix = lda.show_topics(num_topics=20, formatted=False, num_words=50)
# topics_matrix = np.array(topics_matrix)

# topic_words = topics_matrix[:,:,1]
# for i in topic_words:
#     print count
#     print([str(word) for word in i])
#     print()

for i in range(1, numtopics):
    print i
    print lda.show_topic(i, 20)
    print


# In[108]:
char_topic_matrix = [[] for name in names]
for namekey in names:
    print str(namekey) + " cluster: "
    print lda[corpus[names[namekey]]]


# In[108]:

from sklearn.cluster import KMeans

num_clusters = 5

kmeans = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

