#Code for Part1
from __future__ import division
import operator
from collections import Counter
import sys
from math import log
M = 0
V = 0 
class_count = []
class_word_count = {}
class_vocabulary = {}
labels = []
cache = {}

def readModelFile():
	global M
	global V
	global class_count
	global class_word_count
	global class_vocabulary
	with open(sys.argv[2],'r') as f:
		M = int(f.readline())
		V = int(f.readline())
		vocabulary = eval(f.readline())
		class_count = eval(f.readline())
		model = eval(f.readline())
		class_word_count = eval(f.readline())
		class_vocabulary = eval(f.readline())
	for key in class_count:
		labels.append(key)

readModelFile()
print "Modelfile read"

classProbability = {}

def findClassProbability():
	for label in class_count:
		classProbability[label] = class_count[label]/M

findClassProbability()

def calculateConditionalProbability(word,classLabel):
	if((word,classLabel) in cache):
		return cache[(word,classLabel)]
	else:
		if(word in class_vocabulary[classLabel]):
			nk = class_vocabulary[classLabel][word]
		else:
			nk = 1
		n = class_word_count[classLabel]
		p = nk/(n+V)
		cache[(word,classLabel)] = p
		return p

def mostLikelyClass(doc):
	probabilities = []
	for classLabel in labels:
		p = log(classProbability[classLabel])
		for word in doc:
			p = p + log(calculateConditionalProbability(word,classLabel))
		probabilities.append(p)
	ix = probabilities.index(max(probabilities))
	return labels[ix]

def readTestFile(filename):
	result = []
	line_number = 0
	with open(filename,'r') as f:
		for line in f: 
			doc = line.split()
			ans = mostLikelyClass(doc)
			result.append(ans)
			print line_number
			line_number = line_number +1 
	return result

testAns = readTestFile(sys.argv[1])
print "Result Calculated"

def writeToFile(output):
	with open(output,'w') as f:
		for ans in testAns:
			f.write(ans)
			f.write("\n")

writeToFile(sys.argv[3])
print "Output Written"