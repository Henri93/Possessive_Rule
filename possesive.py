import re
import nltk
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


'''
Plot a child's (1-error rate) by creating a dataframe
'''
def plotError(child,points,productive_cords):
	plt.figure(figsize=(12, 10),dpi=144)
	df = pd.DataFrame(points)
	ax = sns.lineplot(x="number_of_points", y="avg_error", data=df, markers=True, marker='o', dashes=False, color="#2c3e50")
	
	#prevent labels from bunching up
	if len(points["number_of_points"]) > 20:
		ax.set_xticks(points["number_of_points"][::3])
		ax.set_xticklabels(points["age"][::3])
	else:
		ax.xaxis.set_ticks(points["number_of_points"])
		ax.xaxis.set_ticklabels(points["age"]) 

	plt.xticks(rotation=-80)
	ax.set(ylim=(0, 1.05))
	ax.set(xlabel="Age")
	ax.set_title(child)
	
	#blue shaded regions for productive periods
	for coord in productive_cords:
		if coord[1] is None:
			end = points["number_of_points"][-1]
		else:
			end = coord[1]
			
		ax.axvspan(coord[0], end, alpha=0.3, color='#3498db')
		
	ax.text(0.5, 1.06, "Productive Rule", horizontalalignment='left', size='medium', color='#3498db')
	fig = ax.get_figure()
	fig.savefig("Test/"+child+"-avg_error_rate"".png")
	plt.show()


'''
Plot a child's unique productions by creating a dataframe
'''
def plotProduction(child,points,productive_cords):
	plt.figure(figsize=(12, 10),dpi=144)
	df = pd.DataFrame(points)
	ax = sns.lineplot(x="number_of_points", y="unique_productions", data=df, markers=True, marker='o', dashes=False, color="#2c3e50")
	
	#prevent labels from bunching up
	if len(points["number_of_points"]) > 20:
		ax.set_xticks(points["number_of_points"][::3])
		ax.set_xticklabels(points["age"][::3])
	else:
		ax.xaxis.set_ticks(points["number_of_points"])
		ax.xaxis.set_ticklabels(points["age"]) 

	plt.xticks(rotation=-80)
	ax.set(xlabel="Age")
	ax.set_title(child)
	
	#blue shaded regions for productive periods
	for coord in productive_cords:
		if coord[1] is None:
			end = points["number_of_points"][-1]
		else:
			end = coord[1]
			
		ax.axvspan(coord[0], end, alpha=0.3, color='#3498db')
		
	ax.text(0.5, np.max(points["unique_productions"]), "Productive Rule", horizontalalignment='left', size='medium', color='#3498db')
	
	fig = ax.get_figure()
	fig.savefig("Test/"+child+"-productions"".png")
	plt.show()

'''
Sort a child's files by age using alphanumeric comparison
'''
def sortFiles(child):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	files = sorted(os.listdir(child), key = alphanum_key)
	for file in files:
		if not(file[0].isdigit()):
			files.remove(file)
	return files

'''
Create a nice age label for plotting i.e. 01;01.00
'''
def convertLabel(label):
	year = label[:2]
	month = label[2:4]
	day = label[4:6]
	return year + ";" + month + "." + day

'''
Double check regular expression match with Stanford's POS tagger
'''
def confirmLacking(line, proper, noun):
	properCheck = False
	nounCheck = False

	tokens = nltk.word_tokenize(line)
	pos = nltk.pos_tag(tokens)
	
	#make sure first is NNP(Proper Noun) and second is N(noun)
	for tag in pos:
		if tag[1] == "NNP" and tag[0] == proper:
			properCheck = True
			
		if "N" in tag[1] and tag[0] == noun:
			nounCheck = True

	return properCheck and nounCheck


def main():

	children = ["Eve","Peter","Naomi"]

	#set graph styles
	sns.set()
	sns.set_style("whitegrid")
	sns.set_context("paper", rc={"lines.linewidth": 3, 'lines.markeredgewidth': 0.2, "font.size":14, "axes.titlesize":22,"axes.labelsize":16,'xtick.labelsize': 12, 'ytick.labelsize': 12})

	contraps = ["everyone's", "he's", "how's", "here's", "it's", "let's", "she's", "somebody's" "someone's", "something's", "that's", "there's", "this's", "what's", "when's", "where's", "which's", "who's", "why's"]

	#perform calculations for each child
	for child in children:
		lacking_possesive = 0
		correct_possessive = 0
		error_rates = []

		N = 0
		E = 0
		avg_error = 0
		
		productive = "non-productive"
		prev_porductive = "non-productive"
		productive_cords = []

		correct_possessive_cases = set()
		unique_possessive_cases = set()
		lack_possesive_cases = set()
		verb_cases = set()
		
		#used to create a dataframe to plot
		points = {"number_of_points":[], "age":[], "1 - error_rate":[], "avg_error":[],"unique_productions":[], "productive":[]}
		
		files = sortFiles(child)
		

		number_of_points = 0
		
		for filename in files:
			
			current_file = child + "/" + filename
			label = os.path.splitext(filename)[0]

			with open(current_file, errors='ignore') as file:
				
				#reset
				lacking = 0
				correct = 0
				line_num = 0

				data = file.readlines()
				for line in data:
					#child utterance
					if "*CHI" in line:
						
						#find cases of missing possesive 's
						for token_verbs in re.finditer("n:prop\|(\w*)\sn\|(\+n\|)?(\w*)(\+n\|)?(\w*)", data[line_num+1], re.IGNORECASE):
							if token_verbs:
								proper = token_verbs.group(1)
								noun = token_verbs.group(3)

								if token_verbs.group(5):
									noun = token_verbs.group(3) + token_verbs.group(5)
								
								if not confirmLacking(line, proper, noun):
									lacking += 1
									lack_possesive_cases.add(proper +":"+ noun)

						#find cases of correct possesive 's
						for token_verbs in re.finditer("dn-POSS\s((adj)*(n)*)+", data[line_num+1], re.IGNORECASE):
							if token_verbs:
								for match in re.finditer("(\w*'s)\s(\w*)", line, re.IGNORECASE):
									if match:
										if match.group(1) not in contraps:
											correct += 1
											correct_possessive_cases.add(line)
											unique_possessive_cases.add(match.group(1)+":"+match.group(2))

						#find exceptions to possesive a.k.a. 's for is
						for token_verbs in re.finditer("\w*'s", line, re.IGNORECASE):
							if token_verbs:
								if token_verbs.group(0) in contraps:
									verb_cases.add(token_verbs.group(0))
									
								

					line_num += 1
				
				#Tolerance principle
				E = len(verb_cases)
				N = len(correct_possessive_cases) + len(verb_cases) 
				theta = 0
				if N > 1 and E > 1:
					theta = N/np.log(N)
					if(E <= theta):
						productive = "productive"
						if prev_porductive is "non-productive":
							productive_cords.append((number_of_points,None))
						prev_porductive = productive
						
					else:
						productive = "non-productive"
						if prev_porductive is "productive":
							productive_cords[-1] = (productive_cords[-1][0], number_of_points)
						prev_porductive = productive
						
				#calculate error rate
				error_rate = 0
				if lacking == 0 and correct == 0:
					error_rate = 1
				else:
					error_rate = 1-(lacking/(lacking+correct))
				avg_error += error_rate

				#add data points for graphing
				error_rates.append(error_rate)
				points["number_of_points"].append(number_of_points)
				number_of_points += 1
				points["age"].append(convertLabel(label))
				points["1 - error_rate"].append(error_rate)
				points["avg_error"].append(avg_error / number_of_points)
				points["unique_productions"].append(len(unique_possessive_cases))
				points["productive"].append(productive)
				
				#print stats for file
				stats = str(round(error_rate, 3)) + "\t E:" + str(E) + " | N:" +str(N) + " | θ:" +str(theta)
				print(current_file + ": " + stats + "\t" + productive)

				
		print(child+" Cor: ",unique_possessive_cases)
		print(child+" Inc: ",lack_possesive_cases)
		print(child+" Veb: ",verb_cases)
		
		#plot child's data
		plotError(child,points,productive_cords)
		plotProduction(child,points,productive_cords)

main()
