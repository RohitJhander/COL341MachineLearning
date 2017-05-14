#!/usr/bin/python
import sys
import re
import scipy
import numpy
import argparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import diagsvd
from scipy.spatial.distance import cosine
from numpy.linalg import inv

####### ARGUMENTS 
parser = argparse.ArgumentParser()
parser.add_argument('-z', dest='z')
parser.add_argument('-k', dest='k')
parser.add_argument('--dir', dest='dir')
parser.add_argument('--doc_in', dest='doc_in')
parser.add_argument('--doc_out', dest='doc_out')
parser.add_argument('--term_in', dest='term_in')
parser.add_argument('--term_out', dest='term_out')
parser.add_argument('--query_in', dest='query_in')
parser.add_argument('--query_out', dest='query_out')
args = parser.parse_args()
z = args.z
k = args.k
directory = args.dir
doc_in  = args.doc_in
doc_out = args.doc_out
term_in = args.term_in
term_out = args.term_out
query_in = args.query_in
query_out = args.query_out

####### Global 
num_docs = 5000
pattern = re.compile(r'\W+')

######## Documents
docMap = {}
docIndexMap = {} 
documents = [[] for i in range(num_docs)]

####### Terms 
vocabulary = {}
termMap = {}

def readDocuments():
	for i in range(1,num_docs+1):
		words = []
		filename = directory+"/"+str(i)+".txt" 
		print("Reading: "+filename)
		with open(filename,'r') as doc:
			title = doc.readline().strip()
			docMap[title] = i-1
			docIndexMap[i-1] = title
			words = words + [x.lower() for x in pattern.split(title)] 
			for sentence in doc:
				sentence = sentence.rstrip('\n')
				if(len(sentence)!=0): words = words + [x.lower() for x in pattern.split(sentence)]
			documents[i-1] = words

readDocuments()
print("======> Reading, Done!")

######## Sparse Matrix
indptr = [0]
indices = []
data = []
for d in documents:
    for term in d:
        index = vocabulary.setdefault(term, len(vocabulary))
        termMap[index] = term
       	indices.append(index)
        data.append(1)
    indptr.append(len(indices))
X = csc_matrix((data, indices, indptr), dtype=int)
print("======> Sparse Matrix, Done!")


######## Singular Value Decompostion (z - rank approximation)
T, s, Dt = svds(X.astype(float), k=int(z), which = 'LM')
S = diagsvd(s, int(z), int(z))
print "T:", T.shape
print "S:", S.shape
print "Dt:", Dt.shape
print("======> SVD, Done!")


######## DOC INPUT QUERY
inputDocs = []
def readInputDocs():
	global inputDocs
	with open(doc_in,'r') as doc:
		for line in doc:
			inputDocs.append(docMap[line.rstrip('\n')])
	
readInputDocs();
#print "inputdocs: ", str(inputDocs)
print("======> Reading Input DOC, Done!")

DS = numpy.dot(numpy.transpose(Dt),S)
print "DS: ", DS.shape
print("======> Building DS Matrix, Done!")

def topKDocs(di,k):
	global DS
	global docIndexMap
	toReturn = []
	cosineVector = []
	for i in range(0,num_docs):
		cosineVector.append(cosine(DS[di],DS[i]))
	topK= numpy.argsort(numpy.asarray(cosineVector))[0:k]
	for index in topK:
		toReturn.append(docIndexMap[index])
	return toReturn

with open(doc_out,"a") as f:
	for i in inputDocs:
		out1 = topKDocs(i,int(k))
		for ans in out1:
			f.write(ans+";\t")
		f.write("\n")

######## TERM INPUT QUERY
inputTerms = []
def readInputTerms():
	global inputTerms
	with open(term_in,'r') as doc:
		for line in doc:
			inputTerms.append(vocabulary[line.rstrip('\n')])
	
readInputTerms();
#print "inputTerms: ", inputTerms
print("======> Reading Input TERM, Done!")

TS = numpy.dot(T,S)
print "TS: ", TS.shape
print("======> Building TS Matrix, Done!")

def topKTerms(ti,k):
	global TS
	global termMap
	toReturn = []
	cosineVector = []
	num_terms = len(vocabulary)
	for i in range(0,num_terms):
		cosineVector.append(cosine(TS[ti],TS[i]))
	topK = numpy.argsort(numpy.asarray(cosineVector))[0:k]
	for index in topK:
		toReturn.append(termMap[index])
	return toReturn

with open(term_out,"a") as f:
	for i in inputTerms:
		out1 = topKTerms(i,int(k))
		for ans in out1:
			f.write(ans+";\t")
		f.write("\n")

####### QUERY
inputQuery = []
def readInputQuery():
	global inputQuery
	with open(query_in,'r') as doc:
		for line in doc:
			inputQuery.append(line.split())
	
readInputQuery()
#print(inputQuery)

S_inverse = inv(S)
T_transpose = numpy.transpose(T)

def getMiniDocument(query):
	global S_inverse
	global T_transpose
	qtf = []
	num_terms = len(vocabulary)
	for i in range(num_terms):
		qtf.append(0.0)
	for term in query:
		qtf[vocabulary[term]] = 1.0
	miniDocument = numpy.dot(S_inverse,numpy.dot(T_transpose,qtf))
	return miniDocument

def topKDocsQuery(miniDocument,k):
	global DS
	global docIndexMap
	toReturn = []
	cosineVector = []
	for i in range(0,num_docs):
		cosineVector.append(cosine(miniDocument,DS[i]))
	topK= numpy.argsort(numpy.asarray(cosineVector))[0:k]
	for index in topK:
		toReturn.append(docIndexMap[index])
	return toReturn


with open(query_out,"a") as f:
	for query in inputQuery:
		out1 = topKDocsQuery(getMiniDocument(query),int(k))		
		for ans in out1:
			f.write(ans+";\t")
		f.write("\n")

