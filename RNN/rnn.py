
import urllib.request
import tensorflow as tf
from collections import Counter
import gensim
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score


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

def getWords(train_doc):
    data =[]
    output = []
    words = []
    for sentence in train_doc:
        tempData = []
        tempOutput =[]
        for word in sentence:
            tempData.append(word[0])
            output.append(word[3])
            words.append(word[0])
        data.append(tempData)
    return data,output,words

def getWordIds(data):
    uniqueWords = list(set(data))
    char2int = dict()
    int2char = dict()
    for i, c in enumerate(uniqueWords):
        char2int[c.lower()] = i
        int2char[i] = c.lower()

    return char2int,int2char

def make_feature_dict(data,w2v_model):
    vecList = []
    word_vectors = w2v_model.wv
    for tokenList in data:
        for tokenTupple in tokenList:
            if (tokenTupple in word_vectors.vocab):
                wordVec = w2v_model.wv[tokenTupple]
                new_temp_list = []
                for j in wordVec:
                    new_temp_list.append([j])
                vecList.append(new_temp_list)
    return vecList

def evaluate(confusion_matrix,int2char):
    dict ={}
    for val,char in int2char.items():
        temp_list = ([0]*len(int2char.items()))
        temp_list[val]=1
        dict[char] =temp_list

    labels = list(int2char.keys())
    rows = ['precision','recall','f1']
    d = pd.DataFrame(index=rows, columns=dict.keys())
    for label in labels:
        tp = confusion_matrix[label][label]
        fp = 0
        fn = 0
        for l in labels:
            if(label != l):
                fp = fp + confusion_matrix[l][label]
                fn = fn + confusion_matrix[label][l]
        p = tp /(tp+fp)
        r = tp /(tp+fn)
        f1 = (2*p*r) / (p+r)
        d.set_value('recall', int2char.get(label), r)
        d.set_value('precision', int2char.get(label), p)
        d.set_value('f1', int2char.get(label), f1)
    return d



if __name__ == '__main__':
    f = open('output.txt', 'w')
    train_doc = read_data('train.txt')
    train_words,train_output,words = getWords(train_doc)
    train_uniqueWords = set(words)
    
    test_doc = read_data('test.txt')
    test_words,test_output,testwords = getWords(test_doc)
    test_uniqueWords = set(testwords)
    
    unique_set = train_uniqueWords.union(test_uniqueWords)
    uniqueWords = list(unique_set)
    
    word_to_id,id_to_word = getWordIds(uniqueWords)
    
    train_output_unique = set(train_output)
    test_output_unique = set(test_output)
    unique_ouput_set = train_output_unique.union(test_output_unique)
    unique_output = list(unique_ouput_set)
    full_data = []
    full_data.extend(train_words)
    full_data.extend(test_words)
    
    model = gensim.models.Word2Vec(full_data,size=50,window=5,min_count=5)
    train_word_vecList = make_feature_dict(train_words,model)
    
    char2int,int2char = getWordIds(unique_output)
    train_output_vecList = []
    for to in train_output:
        temp_list = ([0]*len(unique_output))
        v = char2int.get(to.lower())
        temp_list[v]=1
        train_output_vecList.append(temp_list)
    test_word_vecList = make_feature_dict(test_words,model)


    test_output_vecList = []
    for to in test_output:
        temp_list = ([0]*len(unique_output))
        v = char2int.get(to.lower())
        if isinstance( v, int ):
            temp_list[v]=1
        test_output_vecList.append(temp_list)
    
    
    data = tf.placeholder(tf.float32, [None, 50,1])
    target = tf.placeholder(tf.float32, [None, 5])

    num_hidden = 20
    cell = tf.contrib.rnn.BasicLSTMCell(num_hidden,state_is_tuple=True,activation=tf.sigmoid)
    val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
    yprime = tf.matmul(last, weight) + bias
    prediction = tf.nn.softmax(yprime)
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    minimize = optimizer.minimize(cross_entropy)
    mistakes = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    batch_size = 1000
    no_of_batches = int(len(train_word_vecList) / batch_size)
    epoch = 100
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_word_vecList[ptr:ptr+batch_size], train_output_vecList[ptr:ptr+batch_size]
            ptr+=batch_size
            sess.run(minimize,{data: inp, target: out})

    y_p = tf.argmax(target, 1)
    y_pred = sess.run( y_p, feed_dict={data:test_word_vecList, target:test_output_vecList})
    y_true = np.argmax(test_output_vecList,1)
    f.write('confusion_matrix:'+ '\n\n')
    confusion_mat = confusion_matrix(y_true, y_pred)
    f.write(str(confusion_mat)+ '\n\n')
    evaluation_matrix = evaluate(confusion_mat,int2char)
    f.write('evaluation matrix:'+ '\n\n')
    f.write(str(evaluation_matrix))
    sess.close()
    f.close()












