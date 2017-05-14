#File for training
from __future__ import print_function
from __future__ import division
from collections import Counter
import sys

M = 0
N = 0
V = 0
labels = []
vocabulary = []
class_count = {}
text = []
training_data = []
classes = []

class_vocabulary={}
class_word_count = {}

def readTrainFile(filename):
	global M
	global V
	global N
	global vocabulary
	global class_count
	global class_vocabulary
	global class_word_count
	class_text = {}
	with open(filename,'r') as f:
		for line in f: 
			doc = line.split()
			classLabel = doc[0]
			labels.append(classLabel)
			if(classLabel in class_count):
				class_count[classLabel] = class_count[classLabel]+1
				class_text[classLabel].extend(doc[1:len(doc)])
				class_word_count[classLabel] = class_word_count[classLabel] + len(doc)
			else: 
				class_count[classLabel] = 1
				class_text[classLabel] = doc[1:len(doc)]
				class_word_count[classLabel] = len(doc)
			vocabulary.extend(doc)
			text.append(list(set(doc))) 
			M = M+1
	vocabulary = list(set(vocabulary))
	for key in class_text:
		class_vocabulary[key] = Counter(class_text[key])
	N = len(class_count)
	V = len(vocabulary)
	for key in class_count:
		classes.append(key)

readTrainFile(sys.argv[1])
print("Trainging file read")
print("M: ",M)
print("N: ",N)
print("V: ",V)

def buildFeatures():
	global M
	global training_data
	global vocabulary
	global classes
	for i in range(0,M):
	 	f_vector = [] 
	 	for word in vocabulary:
	 		if word in text[i]:
	 			f_vector.extend([1])
	 		else:
	 			f_vector.extend([0])
	 	training_data.append(f_vector)
	 	
buildFeatures()
print("Feature Set Created")

def conditionalProbability(j,label):
	denominator = class_count[label] + 2.0
	numerator = 1.0
	for i in range(0,M):
		if(training_data[i][j] == 1 and labels[i] == label):
			numerator = numerator + 1.0
	return numerator/denominator

binomialModel = {}

def buildBinomialModel():
	for j in range(0,V):
		for label in classes:
			binomialModel[j,label]  = conditionalProbability(j,label)

buildBinomialModel()
print("Binomial Model Build Completed")
 
def writeToModelFile(modelfile):
 	with open(modelfile,'w') as f:
 		# TEST1
 		f.write(str(M))
 		f.write("\n")
 		f.write(str(V))
 		f.write("\n")
 		print(vocabulary,file=f)
 		print(class_count,file=f)
 		print(binomialModel,file=f)
 		# TEST2
 		print(class_word_count,file=f)
 		print(class_vocabulary,file=f)

writeToModelFile(sys.argv[2])
print("Written to modelfile")