import re
import numpy as np 
import random
#preprocess
#some of the formula of NB
# P(A|B)= P(A)*P(B|A)/P(B)
#P(a|X)= P(X|a)*P(a)
"""
description:
receive a large string split it into string list
param:
None
returns:
None

"""
def textParse(bigString):
    listOfTokens = re.split(r'\W+',bigString) 
    #use special char as the sign 
    #to split aka non num or alphabet
    return[tok.lower() for tok in listOfTokens if len(tok)>2]
    #except singal char word like "I"

"""
function description 
make the splited word list to the vocab list , unique words

param :
dataset
returns:
vocabSet

"""
def createVocabList(dataset):
    vocabSet = set([])
    for doc in dataset:
        vocabSet=vocabSet|set(doc) # take the union 
    return list(vocabSet)


"""
function description :
based on the vocabList , make the inputSet vectorize , the veactor element is either 1 or 0

param:
vocabList- the thing returned from createVocabList
inputset : splited word list

returns:
returnVec-vectorized model 
"""

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("this word %s is not in my vocablist" %word)
    return returnVec

"""
function description: based on the vocabList to build the model
param: 
vocabList - from the return of the createVocabList
inputSet - splited

returns:
returnVec - the vectorization, word model 
"""
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        return returnVec
    
"""
function description:Naive bayes training function 

param: 
trainMatrix- the returnVec from setOfwords2Vec
trainCategory - train document tag vector, the classVec returns from loadDataSet

returns :
p0Vect : non spam conditional proabbaility P(w|0)
p1Vect: spam conditional prob P(w|1)
pAbusive  : spam prob

"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs= len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num= np.ones(numWords);p1Num=np.ones(numWords) #lapalce smoothing
    p0Denom= 2.0;p1Denom= 2.0 #lapalce smoothing
    for i in range(numTrainDocs):
        if trainCategory[i]==1:#calculate the p1 (w|1)
            p1Num+=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = np.log(p1Num/p1Denom) 
    p0Vect = np.log(p0Num/p0Denom) # use log to prevent overflow at the lower bound
    return p0Vect,p1Vect,pAbusive

"""
function description: classifier of naive bayes
param: 
vec2Classify - word havn't being classify 
p0Vec - non spam conditional prob
p1Vec - spam conditional prob
pClass1 - spam prob 
returns:
0 - non spam
1- spam

"""
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1= sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0 
    
def spamTest():
    docList =[];classList= [];fullText=[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r',encoding='latin-1').read())     
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        #spam tag 1
        wordList = textParse(open('email/ham/%d.txt' % i, 'r',encoding='latin-1').read())      
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) 
    trainingSet = list(range(50)); testSet =[]
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("wrongly classfied testing set :",docList[docIndex])
    print('error rate %.2f%%' %(float(errorCount)/len(testSet)*100))


if __name__ == "__main__":
    spamTest()
                                          
                                    


