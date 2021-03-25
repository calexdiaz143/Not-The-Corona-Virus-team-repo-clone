from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
""" Takes in input text, splits by delimiter, returns as dictionary with headings as keys 
    output example: {'Things to know about the COVID-19 Pandemic':'Three Important Ways to Slow the Spread', 'Wear a mask to protect yourself and others and stop the spread of COVID-19.', ''}
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


""" Comparison function need to fill in """
def matchRules(genRule, ruleShort, category, df, threshold):
    returnDf = df.loc[(df['category'] == category) & (df[ruleShort] > threshold)]
    
    return returnDf

""" Compares both sets, returns the union """
def union(genRules, rulesShort, category, df, threshold):
    returnSet = []
    for index in range(0, len(genRules)):
        rules = matchRules(genRules[index], rulesShort[index], category, df, threshold)
        if len(rules) == 0:
            rules.Append(genRules[index])
    #print(returnSet)
    return returnSet

# with open("./Challenge2/data/CDCGuidelines.txt", encoding="utf8") as myFile:
# # filename = askopenfilename()
# # with open(filename, encoding="utf8") as myFile:
#     data = myFile.read()

#     dfRes = textToDataFrame(data, "***")
#     print(dfRes.head(50))
