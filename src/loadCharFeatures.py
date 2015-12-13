
# coding: utf-8

# general
from __future__ import division
from collections import defaultdict
from collections import Counter
import os
import random
import re

# to read in JSON character objects 
import json

# for stopwords
import nltk
from nltk.tokenize import RegexpTokenizer

# for lots of math junk and plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# tokenize input, perform stemming using nltk snowball stemming
# used to remove punctuation, etc.
def tokenize_and_stem(text):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(stemmer.stem(token))
    return filtered_tokens

# Read in character JSON, build feature vectors for a BOW-type model
def getCharVectorsFromJson( mention_threshold = 100,
                            use_stopwords = True,
                            book_src_dir = "C:\\Users\\Molly\\Google Drive\\senior classes\\nlp\\term_project\\book-nlp-master\\book-nlp-master\\data\\originalTexts\\full_hp.txt",
                            json_filename = "C:\\Users\\Molly\\Google Drive\\senior classes\\nlp\\term_project\\char-net\\booknlp_output\\book.id.book"):
    
    bookfile = open(book_src_dir, 'r')
    book = bookfile.read()
    bookDict = defaultdict(float)
    for word in book.split():
        bookDict[word] += 1
    counter = Counter(bookDict)
    top_n = counter.most_common(100)
    print "Most common words in book: ", top_n
    print
    bookfile.close()

    # Paste in direct path to the JSON file (because the relative paths aren't working)

    # wk_dir = os.path.dirname(os.path.realpath('__file__'))
    # json_filename = os.path.join(wk_dir, "..", "booknlp_output\\potter.all.book.txt")

    with open(json_filename) as json_file:
    	char_json = json.load(json_file)
    print "successfully loaded json"
    json_file.close()


    # Load stopwords list, combine it with NLTK's English stopwords list (add words to list as needed)
    stopwords = []
    if use_stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
        curDir = os.path.dirname(os.path.realpath('__file__'))
        stop = os.path.join(curDir,'stopwords')     
        stopfile = open(stop, 'r')
        stop = stopfile.read()
        for word in stop.split():
            stopwords.extend(tokenize_and_stem(word))

        # also throw out the top n words in the book
        # take this out if necessary? just seeing if throwing out the top words improves results
        for word in top_n:
            stopwords.extend(tokenize_and_stem(word[0].decode("utf8").lower()))   
        print stopwords
        stopwords = set(stopwords)


    # Create character feature vectors
    # Possible improvements:
    # Fiddle with the mention threshold, dialogue features (to include or not to include?)
    # maybe throw out the most common attributes afterward

    # dictionary of lists of character attributes, indexed by name (i.e. character's "bag of words")
    characters = {}
    # dictionary mapping character's first given "name" to the list of all names referring to them
    character_namelists = {}
    # dictionary holding the counts of every feature we encounter (to determine common/useless features later)
    total_features = defaultdict(float)

    # iterate through JSON character list returned by bookNLP
    for character in char_json["characters"]:
        # Throw out characters that are mentioned less than n times (play with this threshold?)
        if character["NNPcount"] > mention_threshold:
            namelist = [character["names"][i]['n'] for i in range(len(character["names"]))]
            char_name = character["names"][0]['n']
            #store character's full list of names for later use
            character_namelists[char_name] = namelist
            
            #create feature vector for that character
            characters[char_name] = []

            # add patient verbs to the character BOW
            for word_dict in character["patient"]:
                curWord = tokenize_and_stem(word_dict['w'])
                if len(curWord) > 0 and curWord[0].lower() not in stopwords:
                    characters[char_name].append(curWord[0] + "_PATIENT")
                    total_features[curWord[0] + "_PATIENT"] += 1
            # add agent verbs to the character BOW
            for word_dict in character["agent"]:
                curWord = tokenize_and_stem(word_dict['w'])
                if len(curWord) > 0 and curWord[0].lower() not in stopwords:
                    characters[char_name].append(curWord[0] + "_AGENT")
                    total_features[curWord[0] + "_AGENT"] += 1
            # add adjectives to the character BOW
            for word_dict in character["mod"]:
                curWord = tokenize_and_stem(word_dict['w'])
                if len(curWord) > 0 and curWord[0].lower() not in stopwords:
                    characters[char_name].append(curWord[0] + "_MOD")
                    total_features[curWord[0] + "_MOD"] += 1
            # add character's spoken dialogue to character BOW
            for word_dict in character['speaking']:
                curPhrase = word_dict['w']
                spoken_words = tokenize_and_stem(curPhrase)
                for word in spoken_words:
                    if word not in stopwords:
                        characters[char_name].append(word + "_SAY")
                        total_features[word + "_SAY"] += 1
            # add character "possessions" to character BOW
            # are things characters possess indicative of their character? Often not, so maybe exclude
            for word_dict in character["poss"]:
                curWord = tokenize_and_stem(word_dict['w'])
                if len(curWord) > 0 and curWord[0].lower() not in stopwords:
                    characters[char_name].append(curWord[0] + "_POSS")
                    total_features[curWord[0] + "_POSS"] += 1

    # returns character BOW dictionary (characters), list of names for each character, all the feature counts, and the top n book words
    return (characters, character_namelists, total_features, top_n)
