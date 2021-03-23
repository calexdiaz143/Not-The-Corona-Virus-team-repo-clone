from tkinter.filedialog import askopenfilename

""" Takes in input text, splits by delimiter, returns as dictionary with headings as keys 
    output example: {'Things to know about the COVID-19 Pandemic':'Three Important Ways to Slow the Spread', 'Wear a mask to protect yourself and others and stop the spread of COVID-19.', ''}
"""
def splitText(text, delimiter):
    textArray = text.split(delimiter)
    textDictionary = {}
    for line in textArray:
        # finds the first line in the section and uses that as the heading
        firstNewlineIndex = line.find("\n")
        # adds remaining lines to dictionary, replace("\xa0", " ") added in to get rid of weird symbol
        textDictionary[line[0:firstNewlineIndex]] = (line[firstNewlineIndex + 1:]).replace("\xa0", " ").split("\n")
    return textDictionary


filename = askopenfilename()
# with open("C:/Users/Timmy/Documents/work/hackathon/Not-The-Corona-Virus-team-repo/Challenge2/data/CDCGuidelines.txt", encoding="utf8") as myFile:
with open(filename, encoding="utf8") as myFile:
    data = myFile.read()
    docArray = splitText(data, "***")
    
    print(docArray["Symptoms of Coronavirus"])

