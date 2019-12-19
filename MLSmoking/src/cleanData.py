import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import re
#vape = pd.read_csv('coldTurkeyData.csv',error_bad_lines=False)
# no_vape = pd.read_csv('seleniumData.csv')
# #cold_turkey = pd.read_csv('coldTurkeyData.csv',error_bad_lines=False)
# no_vape.columns = no_vape.columns.str.strip()

# # no_vape['post'] =( line.split('\t') for line in no_vape['post'])
# # print(no_vape)
# # cold_turkey = cold_turkey.dropna()
# # print(cold_turkey.head())
# no_vape = no_vape.drop(columns =['index'])
# no_vape['post'].str.replace("\t", " ")
# # no_vape['post'].str.replace(":)", " ")
# no_vape['post'] = no_vape['post'][0:no_vape['post'].find(".")]
# no_vape = no_vape[no_vape.post.str.len() > 5]
# # no_vape = no_vape[0:no_vape.post.str.find(".")]
# no_vape.to_csv('./clean/seleniumCleaned.csv')
def load_corpus():
	corpus = ""
	#this loads the data from sample_corpus.txt
	with open('testing.txt','r') as f:
	# 	temp  = f.read().replace('\n','')
	# 	temp = re.sub(r'[^a-zA-Z ,.,\']', ' ', temp)
	# 	arrayOfWords = temp.split()
	# 	corpus = ""
	# 	for i in range(0,len(arrayOfWords)):
	# 		if(arrayOfWords[i].count('.') >= 2):
	# 			arrayOfWords[i] = arrayOfWords[i].replace('.', ' ')
	# 	for i in range(0, len(arrayOfWords)):
	# 		if(arrayOfWords[i].count(' ') > 1):
	# 			count = arrayOfWords[i].count(' ')
	# 			arrayOfWords[i] = arrayOfWords[i].replace(' ', '', count - 1)
	# 		# if(re.search("\W",arrayOfWords[i]) != None):
	# 		# 	x = re.sub("\W,", " ", arrayOfWords[i])
	# 		# 	print("Not word character, ", i)
	# 		# 	#print(x)
	# 	if(arrayOfWords[len(arrayOfWords)-1].find('.') == -1):
	# 		arrayOfWords[len(arrayOfWords)-1] = arrayOfWords[len(arrayOfWords)-1]+"."
	# 	for word in arrayOfWords:
	# 		corpus = corpus + " " + word
	# 	corpus = corpus.strip()
	# 	# print(corpus)
	# return corpus
		temp  = f.read().replace('\n','')
		temp = re.sub(r'[^a-zA-Z ,.,\']', ' ', temp)
		arrayOfWords = temp.split()
		corpus = ""
		for i in range(0,len(arrayOfWords)):
			if(arrayOfWords[i].count('.') >= 2):
				arrayOfWords[i] = arrayOfWords[i].replace('.', ' ')
		for i in range(0, len(arrayOfWords)):
			if(arrayOfWords[i].count(' ') > 1):
				count = arrayOfWords[i].count(' ')
				arrayOfWords[i] = arrayOfWords[i].replace(' ', '', count - 1)
			# if(re.search("\W",arrayOfWords[i]) != None):
			# 	x = re.sub("\W,", " ", arrayOfWords[i])
			# 	print("Not word character, ", i)
			# 	#print(x)
		if(arrayOfWords[len(arrayOfWords)-1].find('.') == -1):
			arrayOfWords[len(arrayOfWords)-1] = arrayOfWords[len(arrayOfWords)-1]+"."
		for word in arrayOfWords:
			corpus = corpus + " " + word
		corpus = corpus.strip()

		#taking out sentences less than 5 words
		
		# print(corpus)
	return corpus

def segment_and_tokenize(corpus):
	#make sure to run: 
	#conda install -c conda-forge spacy 
	#python -m spacy download en
	#in the command line before using this!

	#corpus is assumed to be a string, containing the entire corpus
	nlp = spacy.load('en')
	tokens = nlp(corpus)
	# for word in tokens:
	# 	print(word.text)
	sents = [[t.text for t in s] for s in tokens.sents if len([t.text for t in s])>5]
	for a in sents:
		print(a)
def main():
	sentence = load_corpus()
	segment_and_tokenize(sentence)

if __name__ == "__main__":
    main()	