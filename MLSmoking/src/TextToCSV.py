import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import re
import csv
import fileinput
import sys
#Version - just writing 
#Write our training txt file into two column, [classification type, post]
textnames = ['training_bad_vape.txt', 'training_cold_turkey.txt', 'training_vape_ex.txt']
csvnames = ['training_bad_vape.csv', 'training_cold_turkey.csv', 'training_vape_ex.csv']
for i in range(0,3):
    with open(textnames[i],mode='r', encoding="utf-8") as tf:
        with open(csvnames[i], mode='w', encoding="utf-8") as csv_file:
            line = tf.readline()
            fieldnames = ["classification", "posts"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            while line:
                writer.writerow({'classification': '0', 'posts': line.strip()})
                line = tf.readline()



#Version - writing and cleaning 
# 
# Tim - I write this to test if I can clean the data then write the file
# Result : I cannot. Error:
# Traceback (most recent call last):
#   File "TextToCSV.py", line 54, in <module>
#     if(arrayOfWords[len(arrayOfWords)-1].find('.') == -1):
# IndexError: list index out of range
# Idk how to fix that. 
# textnames = ['training_bad_vape.txt', 'training_cold_turkey.txt', 'training_vape_ex.txt']
# csvnames = ['training_bad_vape2.csv', 'training_cold_turkey2.csv', 'training_vape_ex2.csv']
# for i in range(0,3):
#     with open(textnames[i],mode='r', encoding="utf-8") as tf:
#         with open(csvnames[i], mode='w', encoding="utf-8") as csv_file:
#             line = tf.readline()
#             fieldnames = ["classification", "posts"]
#             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#             writer.writeheader()
#             while(line):
#                 temp  = line.replace('\n','')
#                 temp = re.sub(r'[^a-zA-Z ,.,\']', ' ', temp)
#                 arrayOfWords = temp.split()
#                 corpus = ""
#                 for i in range(0,len(arrayOfWords)):
#                     if(arrayOfWords[i].count('.') >= 2):
#                         arrayOfWords[i] = arrayOfWords[i].replace('.', ' ')
#                 for i in range(0, len(arrayOfWords)):
#                     if(arrayOfWords[i].count(' ') > 1):
#                         count = arrayOfWords[i].count(' ')
#                         arrayOfWords[i] = arrayOfWords[i].replace(' ', '', count - 1)
#                 # if(re.search("\W",arrayOfWords[i]) != None):
#                 # 	x = re.sub("\W,", " ", arrayOfWords[i])
#                 # 	print("Not word character, ", i)
#                 # 	#print(x)
#                 if(arrayOfWords[len(arrayOfWords)-1].find('.') == -1):
#                     arrayOfWords[len(arrayOfWords)-1] = arrayOfWords[len(arrayOfWords)-1]+"."
#                 for word in arrayOfWords:
#                     corpus = corpus + " " + word
#                 corpus = corpus.strip()
#                 writer.writerow({'classification': '0', 'posts': corpus })
#                 line = tf.readline()