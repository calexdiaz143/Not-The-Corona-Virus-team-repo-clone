# -*- coding: utf-8 -*-

#%%
## Import python packages
import pandas as pd
import numpy as np

import gensim
import gensim.downloader
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from gensim import corpora
from gensim import models
from gensim import similarities


from collections import defaultdict
#%%
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
#nltk.download('averaged_perceptron_tagger')
en_stops = set(stopwords.words('english'))
if 'you' in en_stops:
    en_stops.remove('you')
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

rule_shortNames = ['wear_mask',
    'social_distance',
    'avoid_crowds',
    'poor_ventilation',
    'wash_hands',
    'cover_coughs',
    'disinfect_surfaces',
    'monitor_health',
    'vaccine']

PROMPTS = ["If you are sick with COVID-19",
           "If you are an older adult",
           "If you have asthma",
           "If you are at home caring for a newborn and are diagnosed with or test positive for COVID-19"]


# chose to modify the situation keywords slightly to make them single words
CATEGORIES = ['sick', 'older', 'asthma', 'newborns']
PROMPTS = ['If you are sick with COVID-19', 'If you are an older adult', 
           'If you have asthma', 
           'If you are at home caring for a newborn and are diagnosed with or test positive for COVID-19']
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
    df = pd.DataFrame(columns=["headerIndex","header", "text"])
    headerIndex = 0
    for line in textArray:
        if len(line) > 0:
            # print(headerIndex)
            # finds the first line in the section and uses that as the heading
            firstNewlineIndex = line.find("\n")
            header = line[0:firstNewlineIndex]
            # print(header)
            # puts the remaining text into dataframe
            df2 = pd.DataFrame({'headerIndex':headerIndex, 'header': header, 'text':(line[firstNewlineIndex + 1:]).replace("\xa0", " ").split("\n")})
            # combines new dataframe with the return dataframe
            df = df.append(df2, ignore_index=True)
            headerIndex += 1
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
combined = list(df['text']) + list(df['header']) + GENERAL_RULES

for text in combined:
    for token in text.split(' '):
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

# first make copy of unaltered text, this will be needed later
df['header_orig'] = df['header']
df['text_orig'] = df['text']

df = nlpCleanup(df, columnName='header')
df = nlpCleanup(df, columnName='text')

#%%
"""
pre-process general rules and prompts
"""
genRulesDf = pd.DataFrame(GENERAL_RULES, columns=['rule'])

genRulesDf = nlpCleanup(genRulesDf, columnName='rule')

print(genRulesDf.head())

promptDf = pd.DataFrame(PROMPTS, columns=['prompt'])
promptDf = nlpCleanup(promptDf, columnName='prompt')

print(genRulesDf.head())

#%%
"""
pre-process prompts rules
"""
promptsDf = pd.DataFrame(PROMPTS, columns=['header'])

promptsDf = nlpCleanup(promptsDf, columnName='header')

print(promptsDf.head())
#%%
"""
Next step is to vectorize the situation keywords (aka categories)
I am using the glove vectors, which is a word2vec model trained on wikipedia
"""
cat_vects = []
for c in CATEGORIES:
    v = glove_vectors.word_vec(c)
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
                v = glove_vectors.word_vec(w)
                # dist = glove_vectors.cosine_similarities(v, catDf.values)
                # found that euclidean dist worked better than cosine similarity...
                dist = np.sum(np.square(catDf - v), axis = 1)
                idx = np.argmin(dist)
                d = dist[idx]
                kw = catDf.index[idx]
                if d < minDist:
                    closestKeyWord = kw
                    minDist = d
        except Exception as e:
            # sometimes the words are not in glove_vectors, so just ignore those errors
            # print(e)
            pass
    return closestKeyWord


df['category'] = df['header'].apply(closestVect)

#%%
"""
Next step is to limit to rows that match one of the 4 categories
"""
df = df[np.logical_not(df['category'].isnull())]

headerDf = df[['headerIndex', 'header', 'category']]
headerDf = headerDf.drop_duplicates()

# print out those rows
print(headerDf)

#%%
"""
Next we want to limit these texts to just ones that start with a verb
"""

def startsWithVerb(s):
    # returns if the first word in the sentence is a verb = this is a
    # shortcut way to check for actionable instructions
    ret = False
    if len(s)>0:
        s=s.lower()
        tag_pos_string = pos_tag(word_tokenize(s))
        firstWordPartOfSpeech = tag_pos_string[0][1]
        ret = firstWordPartOfSpeech in ('VB', 'VBP')

    return ret

df['text_orig'] = df['text_orig'].str.replace('@', '', regex=False)
df['text_orig'] = df['text_orig'].str.replace('*', '', regex=False)

df['actionable'] = df['text_orig'].apply(startsWithVerb)

#%%
# limit to those that start with verb to give actionable instruction
df = df[df['actionable']==True]

#%%
# this shows the number of instructions by category
# note that we cannot have more than 18 instructions per category
# TODO: we have too many instructions for each category right now...
# this is probably because there are some duplicated instructions within each category
# need to do a self-similarity comparison to remove dups
print(df.groupby('category').count())

print(df)


#%%
"""
Next we want to compare the general rules against the rules that passed all
previous processing steps and find the union.  Using the LSI model to find phrase
similarity.  When a phrase is similar enough drop that general rule,
else add that general rule for that category.
We'll also use phrase similarity to remove duplicate instructions within each category
"""

# this creates a dictionary and bag of words encoding based on all text
headers = list(pd.Series(df['header'].unique()).str.split())
texts = list(df['text'].str.split())
generalRules = list(genRulesDf['rule'].str.split())
allWords = texts + headers + generalRules
dictionary = corpora.Dictionary(allWords)
corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
index = similarities.MatrixSimilarity(lsi[corpus])

#%%
# TODO
"""
Next we need to remove duplicate/highly overlapping instructions within this list
For example, "Keep track of your symptoms." and "Monitor your symptoms." are
very similar instructions given for sick people.
We need to remove the dups within each category.
"""
# for cat in CATEGORIES:
#     df_temp = df[df['category'] == cat]
#     txts = df_temp['text']
#     for i, text in txts:
#         vec_bow = dictionary.doc2bow(doc.lower().split())
#         vec_lsi = lsi[vec_bow]


#%%

"""
Here's an example to show which phrases in the text are close to one of the general rules
"""

doc = "Avoid crowds"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
print(vec_lsi)


#perform a query
sims = index[vec_lsi]
# print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])
orderedResults = []
for doc_position, doc_score in sims:
    orderedResults.append((doc_score, df["text"].iloc[doc_position]))

for i, (sim, text) in enumerate(orderedResults):
    if i > 20:
        break
    print(sim, text)



#%%
"""
add a column to show similarity of each row to the general rules
"""
for i, gen in enumerate(GENERAL_RULES):
    vec_bow = dictionary.doc2bow(gen.lower().split())
    vec_lsi = lsi[vec_bow]
    sims = index[vec_lsi]
    df[rule_shortNames[i]] = sims

#%%
"""
prompt code
"""
promptsTexts = [
    [word for word in document.lower().split() if word not in en_stops]
    for document in PROMPTS
]

promptDict = corpora.Dictionary(promptsTexts)
promptCorpus = [promptDict.doc2bow(promptText) for promptText in promptsTexts]
promptLsi = models.LsiModel(promptCorpus, id2word=promptDict,  num_topics=2)
promptDoc = "if you are old and have asthma"
prompt_vec_bow = promptDict.doc2bow(promptDoc.lower().split())
prompt_vec_lsi = promptLsi[prompt_vec_bow]
print(prompt_vec_lsi)
promptIndex = similarities.MatrixSimilarity(promptLsi[promptCorpus])

promptSims = promptIndex[prompt_vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(promptSims)))

promptSims = sorted(enumerate(promptSims), key=lambda item: -item[1])
for doc_position, doc_score in promptSims:
    print(doc_score, PROMPTS[doc_position])
    
#%%
# a threshold to id very similar phrases
THRESHOLD = 0.999  # TODO: this probably should be tuned

dfs = []

for cat in CATEGORIES:
    print('--------')
    print(cat)
    tempDf = df[df['category'] == cat]
    text = tempDf['text']

    for i, rule in enumerate(rule_shortNames):
        dupRule = np.any(tempDf[rule] > THRESHOLD)

        if not dupRule:
            newRule = pd.DataFrame([[cat, GENERAL_RULES[i]]], columns=['category', 'text_orig'])
            tempDf = pd.concat([tempDf,newRule])
        else:
            print('found dup:', rule)

    dfs.append(tempDf[['category','text_orig']])

finalDf = pd.concat(dfs)

#%%
"""
format final output per HTM spec and save to csv
"""

finalDf['category'] = finalDf['category'].str.replace('older', 'older_adult', regex=False)
finalDf['category'] = finalDf['category'].str.replace('newborns', 'covid_with_newborn', regex=False)

finalDf.columns = ['situation', 'rules']

print(finalDf)

fileName = './submission/Challenge2_submission.csv'

finalDf = finalDf.drop_duplicates()
finalDf.to_csv(fileName, index=False)


# """ Comparison function need to fill in """
# def matchRules(genRule, ruleShort, category, df, threshold):
#     returnDf = df.loc[(df['category'] == category) & (df[ruleShort] > threshold)]
#     # rules = returnDf.
#     #
#     #return returnDf['text_orig'].values.tolist()
#     return returnDf

# """ Compares both sets, returns the union """
# def union(genRules, rulesShort, category, df, threshold):
#     returnSet = pd.DataFrame()
#     for index in range(0, len(genRules)):

#         match = matchRules(genRules[index], rulesShort[index], category, df, threshold)
#         #if len(match) == 0:
#         #    match = pd.DataFrame(genRules[index])
#         returnSet = returnSet.append(match)

#     return returnSet.drop_duplicates(subset=['text_orig'])

    # return list(dict.fromkeys(returnSet))
    #return matchRules(genRules[0], rulesShort[0], category, df, threshold)

# print("testing Union")
# returnSick = union(GENERAL_RULES, rule_shortNames, CATEGORIES[0], df, 0.99)

