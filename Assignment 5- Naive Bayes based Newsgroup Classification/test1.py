#Code for Part1
from __future__ import division
import operator
from collections import Counter
import sys
from math import log

M = 0
V = 0
vocabulary = []
class_count = []
model ={}
labels = []

def readModelFile():
	global M
	global V
	global class_count
	global vocabulary
	global model
	with open(sys.argv[2],'r') as f:
		M = int(f.readline())
		V = int(f.readline())
		vocabulary = eval(f.readline())
		class_count = eval(f.readline())
		model = eval(f.readline())
	for key in class_count:
		labels.append(key)

readModelFile()
print "Modelfile read"

classProbability = {}

def findClassProbability():
	for label in class_count:
		classProbability[label] = (class_count[label]+1)/(M+len(labels))

findClassProbability()

def findProbability(f_vector,label,newWords):
	p = log(classProbability[label])
	for j in range(0,V):
		if(f_vector[j] == 1):
			p = p+log(model[(j,label)])
		else:
			p = p+log((1.0-model[(j,label)]))
	p = p + newWords*log(1/(class_count[label]+2))
	return p

def mostLikelyClass(f_vector,newWords):
	probabilities = []
	for label in class_count:
		p = findProbability(f_vector,label,newWords)
		probabilities.append(p)
	ix = probabilities.index(max(probabilities))
	return labels[ix]
	
def createFeatureVector(doc):
	f_vector=[]
	for word in vocabulary:
		if word in doc:
			f_vector.extend([1])
		else:
			f_vector.append([0])
	return f_vector

def numNewWords(doc):
	num = 0
	for word in doc:
		if word not in vocabulary:
			num=num+1
	return num

def readTestFile(filename):
	result = []
	line_number = 0
	with open(filename,'r') as f:
		for line in f: 
			doc = list(set(line.split()))
			f_vector = createFeatureVector(doc)
			newWords = numNewWords(doc)
			ans = mostLikelyClass(f_vector,newWords)
			result.append(ans)
			line_number = line_number + 1 
	return result

testAns = readTestFile(sys.argv[1])
print "Result Calculated"

def writeToFile	(output):
	with open(output,'w') as f:
		for ans in testAns:
			f.write(ans)
			f.write("\n")

writeToFile(sys.argv[3])
print "Output Written"