''' < Import and load the spacy model > '''
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example
from TrainingDataFinal import TRAIN_DATA
import matplotlib.pyplot as plt

''' < reading data > '''
with open('data\jobs_crawl_filter_jobsuche_en.csv', encoding="utf8") as f:
    contents = f.readlines()

''' < Empty Lists to save data > '''
JobTitle = list()
JobID = list()
separedted_Skills = list()
true_Skills = list()

''' < Extracting Title of the Job >'''
for jobs in contents:
    for jobTitle in jobs:
        index = jobs.index(jobTitle)
        if jobTitle.isdecimal() and (jobTitle+jobs[index+1]==jobTitle+'|'):
            jobTitle = jobs.split('|')[2]
            jobId = jobs.split('|')[0]
            if jobTitle not in JobTitle:
                JobTitle.append(jobTitle.lower())
            if jobId not in JobID:
                JobID.append(jobId)
            break

''' < Extracting Job Descriptions of the Job >'''
JobDescription=list()

description = ''
for jobs in contents:
    for jobTitle in jobs:
        index = jobs.index(jobTitle)
        if jobTitle.isdecimal() and (jobTitle+jobs[index+1]==jobTitle+'|'):
            if len(description)>0 and description not in JobDescription:
                # description = description.replace("\n", "")
                JobDescription.append(description)
            description = ''
            break
        else:
            jobTitle = jobTitle.replace("\n", "")
            description=description+''+jobTitle.lower()
    description=description+' '

''' < Train a New NER > '''
nlp=spacy.load('en_core_web_sm')

# Getting the pipeline component
ner=nlp.get_pipe("ner")

# New label
LABEL = "SKILL"

# Add the new label to ner
ner.add_label(LABEL)

# Resume training
optimizer = nlp.resume_training()
move_names = list(ner.move_names)

# List of pipes you want to train
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

# List of pipes which should remain unaffected in training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Begin training by disabling other pipeline components
with nlp.disable_pipes(*other_pipes) :

  sizes = compounding(1.0, 4.0, 1.001)
  # Training for 30 iterations     
  for itn in range(60):
    # shuffle examples before training
    random.shuffle(TRAIN_DATA)
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=sizes)
    # ictionary to store losses
    losses = {}

    for batch in batches:
        texts, annotations = zip(*batch)
        
        example = []
        # Update the model with iterating each text
        for i in range(len(texts)):
            doc = nlp.make_doc(texts[i])
            example.append(Example.from_dict(doc, annotations[i]))
        
        # Update the model
        nlp.update(example, drop=0.5, losses=losses)
n=0
# Testing the NER for first 2000 Job Descriptions
for jobdes in JobDescription:
    n+=1
    doc = nlp(jobdes)
    
    for ent in doc.ents:
        # print(ent)
        print(f'Text:{ent.text} | Label: {ent.label_}')
        # if ent.text not in separedted_Skills:
        separedted_Skills.append(ent.text)
        
    if n==2000:
        break

ignore_List= ['edge', 'r.', 'e.g', 'work']
# Separated True Skills
# for skill in separedted_Skills:
#     for train_val in TRAIN_DATA:
#         if skill in train_val[0] and skill not in ignore_List:
#             # if skill not in true_Skills:
#             true_Skills.append(skill)


for skill in separedted_Skills:
    for train_val in TRAIN_DATA:
        if skill.lower() in train_val[0].lower() and skill not in ignore_List:
            true_Skills.append(skill)


# Visualization 
import pandas as pd
df = pd.DataFrame({'Skills':true_Skills})


# Find Most Common Skills
from collections import Counter
most_common_Skills= Counter(true_Skills)

dic = {}
x=list()
y=list()

for skill in most_common_Skills:
    if most_common_Skills[skill] > 105:
        print(f"{skill} : {most_common_Skills[skill]}")
        dic[skill] = most_common_Skills[skill]
        x.append(skill)
        y.append(int(most_common_Skills[skill]))

# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]//2,y[i], horizontalalignment='left', rotation=90)

x= range(len(dic))
y=list(dic.values())
colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
plt.figure(figsize=(15, 5))
plt.bar(range(len(dic)), list(dic.values()), color=colors[3], align='edge', width=0.5)
plt.xticks(range(len(dic)), list(dic.keys()), rotation=90, fontsize=15)
# giving title to the plot
addlabels(x,y)
plt.title("Most Common Skills", fontsize=14)    
# giving X and Y labels
plt.xlabel("Skill Name", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()



''' < Connect to Databank: Sqlite > '''

import sqlite3

conn = sqlite3.connect('myDatabank.db')
cursor = conn.cursor()

# Creating table
table = """ CREATE TABLE Skills (
            JobID INT,
            JobTitle TEXT, 
            JobDescription TEXT, 
            Skills TEXT
        ); """
cursor.execute(table)

n=0
# Testing the NER
for i,  jobdes in enumerate(JobDescription[1:]): 
    n+=1
    doc = nlp(jobdes)
    title = JobTitle[n]
    entitiyText = ""
    for ent in doc.ents:
        entitiyText = entitiyText+ ent.text+  ' | '
    print(entitiyText)
    print()
    cursor.execute('''INSERT INTO JobAdverticements VALUES (?, ?, ?, ?)''', (i, title, jobdes, entitiyText))
    if n>2000:
        break

conn.commit()
conn.close()