# coding: utf-8

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request

def iter_ngrams(doc, n):
    return (doc[i : i+n] for i in range(len(doc)-n+1))

class HMM:
    def __init__(self, smoothing=0):
        self.smoothing = smoothing

    def fit_transition_probas(self, tags):
        states = []
        counts = defaultdict(lambda: Counter())
        transition_probas = {}
        n = 2
        k = self.smoothing
        c = Counter()
        for tag in tags:
            c.update(tag[:-1])
        noPos = len(c.items())
        d = Counter()
        for tag in tags:
            d.update(tag)
        for p , pcount in d.items():
            states.append(p)
        for tag in tags:
            for ngram in iter_ngrams(tag, 2):
                counts[ngram[:-1][0]].update([ngram[-1]])
        
        for ngram, word_counts in counts.items():
            counts[ngram] = {word: count for word, count in word_counts.items()}
       
        for pos in states:
            dict = defaultdict(lambda: 0)
            dprobs = defaultdict(lambda: 0)
            probs = counts[pos]
            dprobs.update(probs)
            for p in states:
                try:
                    dict[p] = (dprobs[p] + k)/(c[pos] +(k * noPos))
                except ZeroDivisionError:
                    dict[p] = 0
            transition_probas[pos] = dict
        self.transition_probas = transition_probas
        self.states = states

            
    

    def fit_emission_probas(self, sentences, tags):
        emission_probas = {}
        counts = defaultdict(lambda: Counter())
        k = self.smoothing
        states = self.states
        d = Counter()
        for tag in tags:
            d.update(tag)
        c = Counter()
        for sentence in sentences:
            c.update(sentence)
        noPos = len(c.items())
        docnum = 0
        for sentence in sentences:
            tokenNum = 0
            for token in sentence:
                counts[tags[docnum][tokenNum]].update([token])
                tokenNum = tokenNum + 1
            docnum = docnum + 1
        for ngram, word_counts in counts.items():
            counts[ngram] = {word: count for word, count in word_counts.items()}
        for tagpos in states:
            dict = defaultdict()
            dprobs = defaultdict(lambda: 0)
            probs = counts[tagpos]
            dprobs.update(probs)
            for sentancePos,sentanceCount in c.items():
                try:
                    dict[sentancePos] = (dprobs[sentancePos] + k)/(d[tagpos] +(k * noPos))
                except ZeroDivisionError:
                    dict[p] = 0
            emission_probas[tagpos] = dict
        self.emission_probas = emission_probas
       


    def fit_start_probas(self, tags):
        counts = defaultdict(lambda: Counter())
        start_probas = {}
        k = self.smoothing
        states = self.states
        c = Counter()
        d = Counter()
        for tag in tags:
            d.update(tag[:-1])
        noPos = len(d.items())
        newTags = [[] for i in range(len(tags))]
        i = 0
        for tag in tags:
            newTags[i] = [tag[j] for j in range(len(tag))]
            i = i + 1
        for tag in newTags:
            tag.insert(0,'start')
        for tag in newTags:
            c.update(tag[:-1])
        for tag in newTags:
            counts['start'].update([tag[1]])
        for ngram, word_counts in counts.items():
            counts[ngram] = {word: count for word, count in word_counts.items()}
        tempDict = defaultdict(lambda:0)
        tempDict.update(counts['start'])
        for pos in states:
            start_probas[pos] = (tempDict[pos] +k) / (c['start'] + (noPos * k))
        self.start_probas = start_probas
    
    

    def fit(self, sentences, tags):
        self.fit_transition_probas(tags)
        self.fit_emission_probas(sentences, tags)
        self.fit_start_probas(tags)


    def viterbi(self, sentence):
        path = []
        proba = 0.0
        h = len(self.emission_probas.keys())
        
        vTrellis = np.zeros(shape=(h,len(sentence)))
        backtrk = np.zeros(shape=(h,len(sentence)))
        posDict = { }
        counter = 0
        for p in self.emission_probas.keys():
            posDict[counter] = p
            counter = counter + 1
        for i, token in enumerate(sentence):
            if i > 0 :
                for m2 in posDict.keys():
                    valuearray = [0] * len(posDict.keys())
                    m2val = posDict[m2]
                    for m1 in posDict.keys():
                        m1val = posDict[m1]
                        btj = self.emission_probas[m2val][token]
                        y = vTrellis[m1,i-1]
                        aij = self.transition_probas[m1val][m2val]
                        vtj = aij * btj * y
                        valuearray[m1] = vtj
                    rowIndex = np.argmax(valuearray)
                    colval = np.amax(valuearray)
                    backtrk[m2,i] = rowIndex
                    vTrellis[m2,i] = colval
            else:
                for m in posDict.keys():
                    val = posDict[m]
                    btj = self.emission_probas[val][token]
                    aij = self.start_probas[val]
                    vtj = aij * btj
                    backtrk[m,i] = -1
                    vTrellis[m,i] = vtj
        rowIndex = np.argmax(vTrellis,axis=0)
        rowIndex = rowIndex[len(sentence)- 1]
        proba = np.amax(vTrellis, axis=0)[len(sentence)- 1]
        path.append(posDict[rowIndex])
        for j in reversed(range(len(sentence))):
            rowIndex = int(backtrk[rowIndex,j])
            if rowIndex > -1:
                path.append(posDict[rowIndex])
        return list(reversed(path)),proba


def read_labeled_data(filename):
    with open(filename, "r") as ins:
        finalArray = []
        array = []
        for line in ins:
            fline = line.rstrip('\n')
            if fline == '':
                finalArray.append(array)
                array = []
            else:
                array.append(fline)
    ins.closed
    finalSentanceArray = []
    finalTagArray = []
    for sentance in finalArray:
        tempSentance = []
        tempTag = []
        for token in sentance:
            sp = token.split()
            if len(sp) > 1:
                tempSentance.append(token.split()[0])
                tempTag.append(token.split()[1])
        finalSentanceArray.append(tempSentance)
        finalTagArray.append(tempTag)
    return finalSentanceArray, finalTagArray




def download_data():
    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')

if __name__ == '__main__':
    fname = 'data.txt'
    if not os.path.isfile(fname):
        download_data()
    sentences, tags = read_labeled_data(fname)

    model = HMM(.001)
    model.fit(sentences, tags)
    print('model has %d states' % len(model.states))
    print(model.states)
    sentence = ['Look', 'at', 'what', 'happened']
    print('predicted parts of speech for the sentence %s' % str(sentence))
    print(model.viterbi(sentence))
