# -*- coding: utf-8 -*-

#%%
## Import python packages
import pandas as pd
import numpy as np

import gensim 
import gensim.downloader
from nltk.corpus import stopwords
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
CATEGORIES = ['sick', 'elderly', 'asthma', 'newborn']

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
print(df)
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
    

def nlpCleanup(df, columnName):
    df[columnName] = df[columnName].str.replace('\d+', '',regex=True) # for digits
    df[columnName] = df[columnName].str.replace(r'(\b\w{1,2}\b)', '',regex=True) # for word length lt 2
    df[columnName] = df[columnName].str.replace('[^\w\s]', '',regex=True) # for punctuation 
    df[columnName] = df[columnName].apply(removeStopWords)
    df[columnName] = df[columnName].str.lower()
    return df


df = nlpCleanup(df, columnName='header')
df = nlpCleanup(df, columnName='text')

#%%
"""
Next step is to vectorize the situation keywords (aka categories)
I am using the glove vectors, which is a word2vec model trained on wikipedia
"""
cat_vects = []
for c in CATEGORIES:
    v = glove_vectors.wv[c]
    cat_vects.append(v)
    
catDf = pd.DataFrame(data = cat_vects, index = CATEGORIES)

#%%
"""
Next step is to compare each word in the title with each of our category vectors. 
Return the minimum distance between all words in sentence with closest situation keyword
If distance is less than 10, classify as applicable to that title
"""

def closestVect(text):
    minDist = 10 #TODO - this is hardcoded for now... this might need to be tuned
    closestKeyWord = None
    
    for w in text.split(" "):
        try:
            if len(w.strip())>0:
                v = glove_vectors.wv[w]
                dist = np.sum(np.square(catDf - v), axis = 1)
                idx = np.argmin(dist)
                d = dist[idx]
                kw = catDf.index[idx]
                if d < minDist:
                    closestKeyWord = kw
                    minDist = d
        except Exception as e:
            # print(e)
            pass
    return closestKeyWord
    


df['category'] = df['header'].apply(closestVect)
