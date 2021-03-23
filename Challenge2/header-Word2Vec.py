# -*- coding: utf-8 -*-

## Import python packages
import pandas as pd

import gensim 
import gensim.downloader
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')


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


# Parse the CDC guidelines text file into a python dictionary, where the keys are the title 
# headers and the values are a list of sentences that fall under that header

def splitText(text, delimiter):
    textArray = text.split(delimiter)
    textDictionary = {}
    for line in textArray:
        # finds the first line in the section and uses that as the heading
        firstNewlineIndex = line.find("\n")
        # adds remaining lines to dictionary, replace("\xa0", " ") added in to get rid of weird symbol
        textDictionary[line[0:firstNewlineIndex]] = (line[firstNewlineIndex + 1:]).replace("\xa0", " ").split("\n")
    return textDictionary

filename = './data/CDCGuidelines.txt'

with open(filename, encoding="utf8") as myFile:
    data = myFile.read()

docArray = splitText(data, "***")

## show an example of one of the titles
print(docArray["Symptoms of Coronavirus"])

"""
Now convert this to a pandas dataframe
"""


"""
Next step is to see if we can automatically sort through all the titles 
and find ones that are closely related to one of our categories
to do this, we need to do some preprocessing on the titles to:
 - remove stop words, such as: the, of, to, etc.
 - remove punctuation (i.e., commas, periods, colons, parens)
 - lower case everything
 
"""

def nlpCleanup(df, columnName):
    df[columnName] = df[columnName].str.replace('\d+', '',regex=True) # for digits
    df[columnName] = df[columnName].str.replace(r'(\b\w{1,2}\b)', '',regex=True) # for words
    df[columnName] = df[columnName].str.replace('[^\w\s]', '',regex=True) # for punctuation 
    return df
    


"""
Next step is to vectorize the situation keywords (aka categories)
"""
cat_vects = []
for c in CATEGORIES:
    v = glove_vectors.wv[c]
    cat_vects.append(v)
    
catDf = pd.DataFrame(data = cat_vects, index = CATEGORIES)


