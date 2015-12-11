
# coding: utf-8

# In[200]:

import os  # for os.path.basename
from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl



# Convert the annotated character list into a vector for each character
# Vectors have length |number of features|, each index is a binary feature indicating if character is in that group
def getVectors(filename):
    f=open(filename,"r")
    nextIsTopic=False;
    TopicCount=0
    thelist=[] #list of all characters with their position being the position in the vector list
    listcount=0
    dic= defaultdict(lambda: -1) # list of their positions in the vector list
    for line in f: # get number of topics and character positions
        line = line.strip('\n')
        if(dic[line]==-1):
            if(line!="*" and nextIsTopic==False):
                thelist.append(line)
                dic[line]=listcount
                listcount+=1
        if(nextIsTopic):
            nextIsTopic=False
            TopicCount+=1 
        if(line=="*"):
            nextIsTopic=True

    f.close()
    f=open(filename,"r")
    currentTopic=-1
    vectors=[]
    for char in thelist:
        l=[]
        for i in range(TopicCount):
            l.append(0)
        vectors.append(l)
    nextIsTopic=False

   # print TopicCount
    
    for line in f:
        line = line.strip('\n')
        if(nextIsTopic==False and line!="*"):
            vectors[dic[line]][currentTopic]=1
        if(nextIsTopic):
            nextIsTopic=False
            currentTopic+=1
        if(line=="*"):
            nextIsTopic=True
    # return the list of character feature vectors and the list of corresponding character names
    return (vectors,thelist)






