#!/usr/bin/env python
# coding: utf-8

# In[38]:


import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import numpy as np
import matplotlib
import random


# In[39]:


df = pd.read_csv('train.csv',sep=',',header=None)
#print(df.head())
word_tokenize = TweetTokenizer()
df = df.fillna('none')
X = df.values
stop_words = set(stopwords.words("english"))
stop_words.add(':')
stop_words.add(',')
stop_words.add('|')
X=X[1:,1:]
#data = [(word_tokenize(x[0]),word_tokenize(x[1]),word_tokenize(x[2]),int(x[3])) for x in X]
data = []
for x in X:
    temp=[]
    for i in range(1,3):
        t = [w.lower() for w in word_tokenize.tokenize(x[i]) if w.casefold() not in stop_words]
        temp.append(t)
    
    data.append(([x[0]],temp[0],temp[1],int(x[3])))

print(data[50])
print(len(data))
random.shuffle(data)


# In[40]:


all_words = []
all_words.append([])
all_words.append([])
all_words.append([])
for x in data:
    for i in range(0,3):
        all_words[i] = all_words[i] + x[i]

keyword_words = nltk.FreqDist(all_words[0])
location_words = nltk.FreqDist(all_words[1])
text_words = nltk.FreqDist(all_words[2])


# In[41]:


#print(keyword_words.B())  222
#print(location_words.B()) 3250
#print(text_words.B()) 23027

keyword_features = list(keyword_words.keys())[0:150]
location_features = list(location_words.keys())[0:2000]
text_features = list(text_words.keys())[0:5000]

def feature_set(keyword,location,text):
    features = {}
    i=0
    for w in keyword_features:
        features[i] = ((w.casefold() in keyword))
        i=i+1
    for w in location_features:
        features[i] = ((w.casefold() in location))
        i=i+1
    for w in text_features:
        features[i] = ((w.casefold() in text))
        i=i+1
    return features
        
    


# In[42]:


featureset = [(feature_set(x[0],x[1],x[2]),x[3]) for x in data]
training_set = featureset[0:6000]
test_set = featureset[6000:]


# In[43]:


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Accuracy : ", nltk.classify.accuracy(classifier,test_set)*100 )
classifier.show_most_informative_features()

