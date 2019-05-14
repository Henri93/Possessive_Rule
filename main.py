import re
import nltk
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
import enchant

real_word_checker = enchant.Dict("en_US")
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
irregular_roots = ["arise","awake","be","were","bear","beat","become","begin","bend","bet","bind","bite","bleed","blow","break","breed","bring","broadcast","build","burn","burst","buy","can","catch","choose","cling","come","cost","creep","cut","deal","dig","do","draw","dream","drink","drive","eat","fall","feed","feel","fight","find","fly","forbid","forget","forgive","freeze","get","give","go","grind","grow","hang","have","hear","hide","hit","hold","hurt","keep","kneel","know","lay","lead","lean","learn","leave","lent","lie","light","lose","make","may…","mean","meet","mow","overtake","pay","put","read","ride","ring","rise","run","saw","say","see","sell","send","set","sew","shake","shall","shed","shine","shoot","shrink","shut","sing","sink","sit","sleep","slide","smell","sow","speak","spell","spend","spill","spit","spread","stand","steal","stick","sting","stink","strike","swear","sweep","swell","swim","swing","take","teach","tear","tell","think","throw","understand","wake","wear","weep","will","win","wind","write"]
irregulars = ["arose","awoke","was","were","bore","beat","became","began","bent","bet","bound","bit","bled","blew","broke","bred","brought","broadcast","built","burnt","burst","bought","could","caught","chose","clung","came","cost","crept","cut","dealt","dug","did","drew","dreamt","drank","drove","ate","fell","fed","felt","fought","found","flew","forbade","forgot","forgave","froze","got","gave","went","ground","grew","hung","had","heard","hid","hit","held","hurt","kept","knelt","knew","laid","led","leant","learnt","left","lent","lay","lit","lost","made","meant","met","mowed","had to","overtook","paid","put","read","rode","rang","rose","ran","saw","said","sold","sent","set","sewed","shook","should","shed","shone","shot","shrank","shut","sang","sank","sat","slept","slid","smelt","sowed","spoke","spelt","spilt","spat","spread","stood","stole","stuck","stung","stank","struck","swore","swept","swam","swung","took","taught","tore","told","thought","threw","understood","woke","wore","wept","would","won","wound","wrote"]

def sortFiles(child):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(os.listdir(child), key = alphanum_key)

def isVerb(word):
	pos_l = {}
	for tmp in wn.synsets(word):
		if tmp.name().split('.')[0] == word:
			if tmp.pos() in pos_l:
				pos_l[tmp.pos()] += 1
			else:
				pos_l[tmp.pos()] = 1
	if pos_l and 'v' in pos_l:
		if (max(pos_l, key=pos_l.get) == 'v' or pos_l['v'] >= 3):
			return True
			
	return False

#set graph styles
sns.set()
sns.set_style("whitegrid")
sns.set_context("paper", rc={"lines.linewidth": 2})

children = ["Eve", "Sarah", "Adam"]
# children = ["Eve"]

for child in children:
	over_reg = 0
	correct_irreg = 0
	error_rates = []

	parent_correct_irregulars = set()
	parent_correct_regulars = set()
	N = 0
	E = 0
	productive = "non-productive"
	prev_porductive = "non-productive"
	productive_cords = []



	correct_irregulars = set()
	correct_regulars = set()
	overregularized = set()
	verbs = set()

	points = {"time":[], "error_rate":[], "productive":[]}
	regular_points = {"time":[], "number":[]}
	
	files = sortFiles(child)


	for filename in files:
		current_file = child + "/" + filename
		
		
		with open(current_file, errors='ignore') as file:
			data = file.readlines()
			
			#reset
			over_reg = 0
			correct_irreg = 0
			line_num = 0

			for line in data:
				#child utterance
				if "*CHI" in line:
					#find part of speech for each word
					tokens = nltk.word_tokenize(line)
					pos = nltk.pos_tag(tokens)

					potential_verbs = set()
					#get verb from MOT pos tag
					for token_verbs in re.finditer("[^ad]v\|(\w*)(&|\s)", data[line_num+1], re.IGNORECASE):
						if token_verbs:
							#get verb stem
							word = token_verbs.group(1)
							# verbs.add(stemmer.stem(word))
							potential_verbs.add(word)
							
					
					for tag in pos:

						#get verb from nltk pos tag
						# if real_word_checker.check(tag[0]) and lemmatizer.lemmatize(tag[0]) in potential_verbs:
						# if "ed" == tag[0][-2:]:
						# 	verbs.add(tag[0])
							# verbs.add(stemmer.stem(tag[0]))
							# if stemmer.stem(tag[0]) in potential_verbs:
							# 	verbs.add(tag[0])
							# else:
							# 	print(tag[0]+": ", potential_verbs)

						#correct irregular
						if tag[0] in irregulars:
							correct_irreg += 1
							correct_irregulars.add(tag[0])
		
						#overregularized
						root = tag[0][:-2]
						if "ed" == tag[0][-2:] and root in irregular_roots:
							over_reg += 1
							overregularized.add(tag[0])
								  	
						#correct regular
						if "ed" == tag[0][-2:] or ('V' in tag[1] and isVerb(tag[0]) and tag[0] not in irregulars):
							#still some overregularized
							correct_regulars.add(tag[0])
							if real_word_checker.check(tag[0]):
								verbs.add(stemmer.stem(tag[0]))

						# #correct regular
						# if "V" in tag[1] and "ed" == tag[0][-2:] and len(tag[0]) >= 3:
						# 	#still some overregularized
						# 	correct_regulars.add(tag[0])
						# 	verbs.add(stemmer.stem(tag[0]))
							
							

				# elif "*" in line:
				# 	#find part of speech for each word
				# 	tokens = nltk.word_tokenize(line)
				# 	pos = nltk.pos_tag(tokens)
				# 	for tag in pos:

				# 		# correct irregular
				# 		if tag[0] in irregulars:
				# 			correct_irreg += 1
				# 			correct_irregulars.add(tag[0])

				# 		#correct regular
				# 		if "ed" == tag[0][-2:] or ('V' in tag[1] and isVerb(tag[0])):
				# 			#still some overregularized
				# 			correct_regulars.add(tag[0])

				line_num += 1
			
			#Tolerance principle
			label = os.path.splitext(filename)[0]
			
			E = len(correct_irregulars)
			N = len(verbs) + len(correct_irregulars) 
			theta = 0
			if N > 0:
				theta = N/np.log(N)
				if(E <= theta):
					productive = "productive"
					if prev_porductive is "non-productive":
						productive_cords.append((label,None))
					prev_porductive = productive
					
				else:
					productive = "non-productive"
					if prev_porductive is "productive":
						productive_cords[-1] = (productive_cords[-1][0], label)
					prev_porductive = productive
					
			#calculate error rate
			error_rate = 0
			if over_reg == 0 and correct_irreg == 0:
				error_rate = 0
			else:
				error_rate = 1-(over_reg/(over_reg+correct_irreg))

			#if error rate is 0 then nothing interesting happened in data period
			if error_rate != 0:
				error_rates.append(error_rate)
				points["time"].append(label)
				points["error_rate"].append(error_rate)
				points["productive"].append(productive)
			
			regular_points['time'].append(label)
			regular_points['number'].append(N)

			stats = str(round(error_rate, 3)) + "\t E:" + str(E) + " | N:" +str(N) + " | θ:" +str(theta)
			print(current_file + ": " + stats + "\t" + productive)
			# break
			
	
	# print("Parent IRR: ",parent_correct_irregulars)
	# print("Parent REG: ",parent_correct_regulars)
	# print("Both IRR: ",parent_correct_irregulars.intersection(correct_irregulars))
	print(child+" IRR: ",correct_irregulars)
	print(child+" OVR: ",overregularized)
	# print(child+" REG: ",correct_regulars)
	print(child+" VERBS: ",verbs)
	
	#plot child's error rate for past tense	
	df = pd.DataFrame(points)
	ax = sns.lineplot(x="time", y="error_rate", data=df, markers=True, color="#2c3e50")
	ax.set(ylim=(0.5, 1.05))
	ax.set_title(child)
	plt.xticks(rotation=-80)
	for coord in productive_cords:
		if coord[1] is None:
			end = points["time"][-1]
		else:
			end = coord[1]
			
		ax.axvspan(coord[0], end, alpha=0.3, color='#3498db')
		
	ax.text(0.5, 1.06, "Productive Rule", horizontalalignment='left', size='medium', color='#3498db')
	plt.show()

	#plot past tense useage
	# df = pd.DataFrame(regular_points)
	# ax = sns.lineplot(x="time", y="number", data=df, markers=True)
	# ax.set_title(child)
	# plt.xticks(rotation=-80)
	# plt.show()
