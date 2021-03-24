# -*- coding: utf-8 -*-

#%%
## Import python packages
import pandas as pd
import numpy as np

import gensim 
import gensim.downloader
from nltk.corpus import stopwords
from gensim import corpora
from gensim import models
from gensim import similarities

from collections import defaultdict
#%%
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')

en_stops = set(stopwords.words('english'))

#%%
## Setup: initialize some constants

GENERAL_RULES = ['Wear a mask',
    'Stay 6 feet from others',
    'Avoid crowds',
    'Avoid poorly ventilated spaces',
    'Wash your hands often',
    'Cover coughs and sneezes',
    'Clean and disinfect frequently touched surfaces daily',
    'Monitor your health daily',
    'Get vaccinated']

# chose to modify the situation keywords slightly to make them single words
CATEGORIES = ['sick', 'older', 'asthma', 'newborns']

#%%
# Parse the CDC guidelines text file into a python dictionary, where the keys are the title 
# headers and the values are a list of sentences that fall under that header

""" Takes in input text, splits by delimiter, returns as dictionary with headings as keys 
    output example: {'Things to know about the COVID-19 Pandemic':'Three Important Ways to Slow the Spread', 'Wear a mask to protect yourself and others and stop the spread of COVID-19.', ''}
"""
def splitText(text, delimiter):
    textArray = text.split(delimiter)
    textDictionary = {}
    for line in textArray:
        # finds the first line in the section and uses that as the heading
        firstNewlineIndex = line.find("\n")
        header = line[0:firstNewlineIndex]
        # adds remaining lines to dictionary, replace("\xa0", " ") added in to get rid of weird symbol
        textDictionary[header] = (line[firstNewlineIndex + 1:]).replace("\xa0", " ").split("\n")
        
    return textDictionary

""" Takes in input text, splits by delimiter, returns as pandas dataframe with headings as keys e.g. [index, heade, text]
"""
def textToDataFrame(text, delimiter):
    textArray = text.split(delimiter)
    df = pd.DataFrame(columns=["header", "text"])
    for line in textArray:
        # finds the first line in the section and uses that as the heading
        firstNewlineIndex = line.find("\n")
        header = line[0:firstNewlineIndex]
        # puts the remaining text into dataframe
        df2 = pd.DataFrame({'header': header, 'text':(line[firstNewlineIndex + 1:]).replace("\xa0", " ").split("\n")})
        # combines new dataframe with the return dataframe
        df = df.append(df2)
    return df

filename = './data/CDCGuidelines.txt'

with open(filename, encoding="utf8") as myFile:
    data = myFile.read()

df = textToDataFrame(data, "***")
print(df.sample(10))

headers = df['header'].unique()
#%%

"""
Next step is to see if we can automatically sort through all the titles 
and find ones that are closely related to one of our categories
to do this, we need to do some preprocessing on the titles to:
 - remove stop words, such as: the, of, to, etc.
 - remove punctuation (i.e., commas, periods, colons, parens)
 - lower case everything
 
"""
def removeStopWords(text):
    returnText = ''
    for w in text.split(' '):
        if w not in en_stops and len(w)>0:
            returnText += ' ' + w
    returnText = returnText.strip()
    # print(returnText)
    return returnText

# remove words that appear only once
frequency = defaultdict(int)
combined = list(df['text']) + list(df['header'])

for text in combined:
    for token in text.split():
        frequency[token] += 1
        
def removeSingleOccurances(text):
    returnText = ''
    for w in text.split(' '):
        if frequency[token] > 1:
            returnText += ' ' + w
    returnText = returnText.strip()
    # print(returnText)
    return returnText
    

def nlpCleanup(df, columnName):
    df[columnName] = df[columnName].str.replace('\d+', '',regex=True) # for digits
    df[columnName] = df[columnName].str.replace(r'(\b\w{1,2}\b)', '',regex=True) # for word length lt 2
    df[columnName] = df[columnName].str.replace('[^\w\s]', '',regex=True) # for punctuation 
    df[columnName] = df[columnName].apply(removeStopWords)
    df[columnName] = df[columnName].apply(removeSingleOccurances)
    df[columnName] = df[columnName].str.lower()
    return df


df = nlpCleanup(df, columnName='header')
df = nlpCleanup(df, columnName='text')

#%%
headers = list(pd.Series(df['header'].unique()).str.split())
texts = list(df['text'].str.split()) 
allWords = texts + headers
dictionary = corpora.Dictionary(allWords)

corpus = [dictionary.doc2bow(text) for text in texts]
# corpus_headers = [headerDictionary.doc2bow(text) for text in headers]

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
# lsi_headers = models.LsiModel(corpus_headers, id2word=dictionary, num_topics=2)

#%%

doc = "Avoid crowds"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] 
print(vec_lsi)

#%%
index = similarities.MatrixSimilarity(lsi[corpus])
#%%
#perform a query
sims = index[vec_lsi]
print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])
orderedResults = []
for doc_position, doc_score in sims:
    orderedResults.append((doc_score, df["text"].iloc[doc_position]))

print(orderedResults[0:20])



#%%
"""
steps:
    1. make corpus from titles and find similar titles to categories
    2. use pos to limit to sentences that start with verbs
    3.  use lsi method to find union of general rules with specific rules
    
"""