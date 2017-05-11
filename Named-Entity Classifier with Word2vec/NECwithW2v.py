# coding: utf-8

from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
import gensim
import nltk
from nltk.corpus import brown


def download_data():
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'    
    urllib.request.urlretrieve(url, 'test.txt')
    
    
def read_data(filename):
    ins = open(filename, "r")
    finalList = []
    array = []
    for line in ins:
        if line.strip():
            templine = line.rstrip('\n')
            tempArray = templine.split()
            if(tempArray[0] != '-DOCSTART-'):
                array.append((tempArray[0],tempArray[1],tempArray[2],tempArray[3]))
        else:
            if(array):
                finalList.append(array)
                array =[]
    finalList.append(array)
    ins.close()
    return finalList

def make_feature_dicts(data,w2v_model,token=True,caps=True,pos=True,chunk=True,context=True,w2v=True):
    listOfDicts = []
    nerList = []
    word_vectors = w2v_model.wv
    for tokenList in data:
        sentanceList = []
        for tokenTupple in tokenList:
            featDict = {}
            if(w2v):
                if (tokenTupple[0] in word_vectors.vocab):
                    wordVec = w2v_model.wv[tokenTupple[0]]
                    for i,val in enumerate(wordVec):
                        tokw2v = 'w2v_' + str(i)
                        featDict[tokw2v] = val
            
            if(token):
                tokfeat = 'tok=' + tokenTupple[0].lower()
                featDict[tokfeat] = 1
            
            if(caps):
                if(tokenTupple[0][0].isupper()):
                    capfeat = 'is_caps'
                    featDict[capfeat] = 1
            if(pos):
                posfeat = 'pos='+tokenTupple[1]
                featDict[posfeat] = 1
            if(chunk):
                chunkfeat  = 'chunk=' + tokenTupple[2]
                featDict[chunkfeat] = 1
            if(context and sentanceList):
                    prevDict = sentanceList[-1]
                    curDict = {'prev_' +k: v for k, v in prevDict.items() if not (k.startswith('prev_') or k.startswith('next_'))}
                    nextDict = {'next_' +k: v for k, v in featDict.items() if not (k.startswith('prev_') or k.startswith('next_'))}
                    featDict.update(curDict)
                    prevDict.update(nextDict)
                    sentanceList[-1] = prevDict
            sentanceList.append(featDict)
            nerList.append(tokenTupple[3])
        listOfDicts.extend(sentanceList)
        sentanceList.clear()
    numpya = np.array(nerList)
    return listOfDicts,numpya


def confusion(true_labels, pred_labels):
    labels = list(set(true_labels))
    labels.sort()
    d = pd.DataFrame(0, index=labels, columns=labels)
    for i in range(len(true_labels)):
        val = d.get_value(true_labels[i], pred_labels[i])
        val = val + 1
        d.set_value(true_labels[i], pred_labels[i], val)
    return d


def evaluate(confusion_matrix):
    labels = list(confusion_matrix.columns.values)
    rows = ['precision','recall','f1']
    d = pd.DataFrame(index=rows, columns=labels)
    for label in labels:
        tp = confusion_matrix.get_value(label, label)
        fp = 0
        fn = 0
        for l in labels:
            if(label != l):
                fp = fp + confusion_matrix.get_value(l, label)
                fn = fn + confusion_matrix.get_value(label, l)
        p = tp /(tp+fp)
        r = tp /(tp+fn)
        f1 = (2*p*r) / (p+r)
        d.set_value('recall', label, r)
        d.set_value('precision', label, p)
        d.set_value('f1', label, f1)
    return d


def average_f1s(evaluation_matrix):
    labels = list(evaluation_matrix.columns.values)
    ser = evaluation_matrix.loc[['f1']]
    val = 0
    for label in labels:
        if label!= 'O':
            val = val + evaluation_matrix.get_value('f1', label)
    avg = val / (len(labels) - 1)
    return avg


def evaluate_combinations(train_data, test_data,w2v_model):
    df = pd.DataFrame(columns=['f1','n_params','caps', 'pos', 'chunk', 'context','w2v'])
    feature_fns = ['caps', 'pos', 'chunk', 'context','w2v']
    combi = list(product([False,True],repeat = 5))
    for y , c in enumerate(combi):
        dicts, labels = make_feature_dicts(train_data,w2v_model,token=True,caps=c[0],pos=c[1],chunk=c[2],context=c[3],w2v=c[4])
        vec = DictVectorizer()
        X = vec.fit_transform(dicts)
        clf = LogisticRegression()
        clf.fit(X, labels)
        test_dicts, test_labels = make_feature_dicts(test_data,w2v_model,token=True,caps=c[0],pos=c[1],chunk=c[2],context=c[3],w2v=c[4])
        X_test = vec.transform(test_dicts)
        preds = clf.predict(X_test)
        confusion_matrix = confusion(test_labels, preds)
        evaluation_matrix = evaluate(confusion_matrix)
        avg = average_f1s(evaluation_matrix)
        clen = len(clf.coef_)
        count = 0
        for i in range(clen):
            count  = count + len(clf.coef_[i])
        df.loc[y] = pd.Series({'f1':avg,'n_params':int(count),'caps':c[0], 'pos':c[1], 'chunk':c[2], 'context':c[3],'w2v':c[4]})
    df[['n_params']] = df[['n_params']].astype(int)
    dataf = df.sort_values(['f1'],ascending=False)
    return dataf



if __name__ == '__main__':
    sentences = brown.sents()
    model = gensim.models.Word2Vec(sentences,size=50,window=5,min_count=5)
    model.save('brownModel')
    download_data()
    f = open('output.txt', 'w')
    train_data = read_data('train.txt')
    w2v_model = gensim.models.Word2Vec.load('brownModel')
    dicts, labels = make_feature_dicts(train_data,w2v_model,token=True,caps=True,pos=True,chunk=True,context=True,w2v=True)
    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    outputString = 'training data shape:' + str(X.shape)+ '\n\n'
    f.write(outputString)
    clf = LogisticRegression()
    clf.fit(X, labels)
    
    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data,w2v_model,token=True,caps=True,pos=True,chunk=True,context=True,w2v=True)
    X_test = vec.transform(test_dicts)
    outputString = 'testing data shape: ' + str(X_test.shape) + '\n\n'
    f.write(outputString)
    
    preds = clf.predict(X_test)
    
    confusion_matrix = confusion(test_labels, preds)
    outputString = 'confusion matrix:\n'+ str(confusion_matrix) +'\n\n'
    f.write(outputString)
    
    evaluation_matrix = evaluate(confusion_matrix)
    outputString = 'evaluation matrix:\n'+ str(evaluation_matrix) + '\n\n'
    f.write(outputString)
    
    outputString = 'average f1s: ' + str(average_f1s(evaluation_matrix))+ '\n\n'
    f.write(outputString)
    
    combo_results = evaluate_combinations(train_data, test_data,w2v_model)
    outputString = 'combination results:\n' + str(combo_results)+ '\n'
    f.write(outputString)
    f.close()
