# Import and load the spacy model
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example
from spacy.matcher import PhraseMatcher

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

''' < reading data > '''
with open('data\jobs_crawl_filter_jobsuche_en.csv', encoding="utf8") as f:
    contents = f.readlines()

JobTitle = list()
JobID = list()

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
n=0
# Testing the NER
for jobdes in JobDescription:
    n+=1
    doc = nlp(jobdes)
    # print("Entities in \n")
    # print(jobdes)
    print('\n')
    # for ent in doc.ents:
    #     # print(ent)
    #     print(f'Text:{ent.text} | Label: {ent.label_}')
    annotations = skill_extractor.annotate(jobdes)
    print(annotations)
    print()
    # print(n)
    if n==50:
        break

