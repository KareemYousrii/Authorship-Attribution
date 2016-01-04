#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import os
from readability import Readability

in_dir = '/Users/ahmedk/Documents/TUM/Courses/Semester 1/Machine Learning/Project/Data/Authors/Additional Test Data'
out_dir = '/Users/ahmedk/Documents/TUM/Courses/Semester 1/Machine Learning/Project/Data/Additional Test Output/Full_Feat'

pos_tags = ['NN', 'IN', 'PRP', 'DT', ',', '.', 'JJ', 'VBD', 'RB', 'CC', 'VB', 'NNS', 'PRP$', '\'\'', 'VBN', '\`\`', 'VBP', 'TO', 'MD', 'VBZ', 'VBG', 'CD', 'WDT', 'WRB', 'RP', 'NNP', 'WP', 'POS', ':', 'EX', 'JJR', 'JJS', '-NONE-', 'PDT', 'RBR', 'RBS', 'WP$']

func_words = ['the', 'and', 'to', 'of', 'a', 'i', 'in', 'was', 'it', 'that', 'he', 'you', 'his', 'her', 'had', 'with', 'as', 'for', 'she', 'but', 'my', 'not', 'at', 'is', 'me', 'have', 'be', 'on', 'him', 'this', 'which', 'all', 'so', 'there', 'by', 'no', 'from', 'do', 'if', 'we', 'were', 'they', 'what', 'an', 'when', 'been', 'out', 'or', 'up', 'are']

for subdir, _, files in os.walk(in_dir):
	for file in files:
		if not file.startswith('.'):

			# Create a corresponding feature file
			out_file = open(os.path.join(out_dir, file), 'w')

			# Read the file
			in_file = open(os.path.join(subdir, file))
			raw = in_file.read().decode('UTF-8')
			raw = raw.lower()
			tokens = nltk.word_tokenize(raw)

			# Create a frequency distribution for the text
			text = nltk.Text(tokens)
			fdist = nltk.FreqDist(text)

			# Calculate the type-token ratio
			vocab_richness = len(set(tokens)) / len(tokens)
			out_file.write(str(vocab_richness) + '\n')			

			# Calculate average word length:
			avg_word_len = fdist.N() / len(fdist)
			out_file.write(str(avg_word_len) + '\n')
			
			# Compute Readability
                        rd = Readability(raw)
                        out_file.write(str(rd.FleschKincaidGradeLevel()) + '\n')			

			# Calculate the distribution of parts-of-speech
			tagged_text = nltk.pos_tag(text)
			tag_fd = nltk.FreqDist(tag for (word,tag) in tagged_text)
			
			for tag in pos_tags:
				out_file.write(str(tag_fd[tag]) + '\n')

			# Calculate the frequency of the 50 most frequenct function words
			stopwords = nltk.corpus.stopwords.words('english')
			txt_stopwords = [w for w in tokens if w in stopwords]
			functionWrd_freq = nltk.FreqDist(txt_stopwords)
			
			for func_word in func_words:
				out_file.write(str(functionWrd_freq[func_word]) + '\n') 
			
			# Close files
			in_file.close()
			out_file.close()
