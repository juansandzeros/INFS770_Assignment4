__author__ = 'jharrington'

import os
import json
import pandas as pd
import numpy as np
import nltk
from re import sub, compile
from sklearn import metrics, model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
import gensim # please intall gensim according to the course lecture
from gensim.models import LdaModel, LsiModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, cross_validation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import sys
# This time, I don't use main and methods. I notice that some of you had trouble understanding how to define and use a method.
#import data
data_file = "amazon_review_texts.csv" # you need to change the filepath
dataset = pd.read_csv(data_file, sep=",")
categories = ['watch', 'automotive','electronic','software']
#examine data
print dataset.head()

'''
Your code for Q1
'''
print "Q1:"
print dataset['score'].value_counts()

# below is the code for preprocessing, you don't need to change the code
# list of stopwords. In order to use nltk stopwords, you need to download NLTK. Please watch the course video
nltk.download("stopwords")
stopwords = set(nltk.corpus.stopwords.words("english"))
#Create a regex tokenizer that only consider alphabets as token and remove numbers
tokenizer = nltk.tokenize.RegexpTokenizer(r"[a-z]+")
#Initialize a stemmer
stemmer = nltk.stem.PorterStemmer()

#Define a function that preprocess a single text and returns a list of tokens
def preprocess(text):
    tokens = []
    for token in tokenizer.tokenize(text.lower()):
        if len(token)>3 and token not in stopwords:
            tokens.append(stemmer.stem(token))
    return tokens
#Process all texts/reviews
processed = map(preprocess, dataset.text) # in your dataset, you have a column "text". dataset.text refers to the column
#Examine first processed text
print processed[0]
'''
 Your code for Q2
'''
print "Q2:"
# calculate the token frequency
# the FreqDist function takes in a list of tokens and return a dict containg unique tokens and frequency
fdist = nltk.FreqDist([token for doc in processed for token in doc])
print fdist.tabulate(10)

processed_doc = map(" ".join, processed)
'''
Your code for Q3
When you run KMeans for 10 times, please try to use for loop
if you are using for loop, you should have something like the following:
nmis = [] # you store the NMI values you obtain in each iteration in this list
for i in range(10):
    # your code for doing kmeans. Please use i as the random_state. Please print the NMI for each iteration and add the NMI value to the list using nmis.append()
    # please also note in the "newsgroup", the dependent variable in the dataset is "target", so we used dataset.target to refer to the dependent variable.
    # In our dataset, the dependent variable is "category", so you need to use dataset.category
    i+=1
import numpy as np
print "the average NMI of 10 iterations: ", np.mean(nmis) # print the average NMI values
'''
print "Q3:"
# vectorize
vectorizer = TfidfVectorizer(stop_words="english")
corpus_vect = vectorizer.fit_transform(processed_doc)
nmis = []
for i in range(10):
    km = KMeans(n_clusters=len(categories), max_iter=100, random_state=i)
    km.fit(corpus_vect)
    score = metrics.normalized_mutual_info_score(dataset.category, km.labels_)
    print score
    nmis.append(score)
    i+=1
print "the average NMI of 10 iterations: ", np.mean(nmis) # print the average NMI values
'''
 Your code for Q4
'''
print "Q4:"
vectorizer = TfidfVectorizer(stop_words="english", min_df=0.01)
corpus_vect = vectorizer.fit_transform(processed_doc)
nmis = []
for i in range(10):
    km = KMeans(n_clusters=len(categories), max_iter=100, random_state=i)
    km.fit(corpus_vect)
    score = metrics.normalized_mutual_info_score(dataset.category, km.labels_)
    print score
    nmis.append(score)
    i+=1
print "the average NMI of 10 iterations: ", np.mean(nmis) # print the average NMI values
'''
 Your code for Q5.
'''
print "Q5:"
# examine the representative words for each cluster
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(len(categories)):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print
'''
 Your code for Q6
'''
print "Q6:"
#df_vect = pd.DataFrame(corpus_vect.toarray(), columns=vectorizer.get_feature_names())
#df_vect
vectorizer = TfidfVectorizer(stop_words="english")
corpus_vect = vectorizer.fit_transform(processed_doc)
# convert the vectorized data to a gensim corpus object
corpus = gensim.matutils.Sparse2Corpus(corpus_vect, documents_columns=False)
# maintain a dictionary for index-word mapping
id2word = dict((v, k) for k, v in vectorizer.vocabulary_.iteritems())
# build the lda model
lda = LdaModel(corpus, num_topics=4,id2word=id2word, passes=10)
print lda.print_topics()
lda_docs = lda[corpus]
#for row in lda_docs:
#    print row
# extract the scores and round them to 3 decimal places
scores = np.round([[doc[1] for doc in row] for row in lda_docs], 3)
# convert the documents scores into a data frame
df_lda = pd.DataFrame(scores, columns=["topic 1", "topic 2", "topic 3", "topic 4"])
print df_lda
'''
 Your code for Q7. Please remember in our dataset, the dependent variable is dataset.score rather than dataset.target
'''
print "Q7:"
skf = cross_validation.StratifiedKFold(dataset.category, n_folds=5)
fold = 0
for train_index, test_index in skf:
    fold += 1
    print "Fold %d" % fold
    # partition
    train_x, test_x = np.array(processed_doc)[train_index], np.array(processed_doc)[test_index]
    train_y, test_y = dataset.score[train_index], dataset.score[test_index]
    # vectorize
    vectorizer = TfidfVectorizer(min_df=0.002, stop_words='english')
    X = vectorizer.fit_transform(train_x)
    print "Number of features: %d" % len(vectorizer.vocabulary_)
    X_test = vectorizer.transform(test_x)
    # train model
    clf = SVC(kernel="linear")
    clf.fit(X, train_y)
    # predict
    pred = clf.predict(X_test)
    # classification results
    for line in metrics.classification_report(test_y, pred).split("\n"):
        print line


satisfaction_map = {5: 1, 4: 0, 3 : 0, 2 : 0, 1 :0}
dataset['satisfaction'] = dataset.score.map(satisfaction_map)
'''
 The above two lines of code create the dependent variable "satisfaction". In pandas, you can use either dataset['satisfaction'] or dataset.satisfaction to refer to a column.
 please try to understand how to use the map function in python. It can be used to do a lot of interesting things
 Your code for Q8.
'''
print "Q8:"
skf = cross_validation.StratifiedKFold(dataset.category, n_folds=5)
fold = 0
for train_index, test_index in skf:
    fold += 1
    print "Fold %d" % fold
    # partition
    train_x, test_x = np.array(processed_doc)[train_index], np.array(processed_doc)[test_index]
    train_y, test_y = dataset.satisfaction[train_index], dataset.satisfaction[test_index]
    # vectorize
    vectorizer = TfidfVectorizer(min_df=0.002, stop_words='english')
    X = vectorizer.fit_transform(train_x)
    print "Number of features: %d" % len(vectorizer.vocabulary_)
    X_test = vectorizer.transform(test_x)
    # train model
    clf = SVC(kernel="linear")
    clf.fit(X, train_y)
    # predict
    pred = clf.predict(X_test)
    # classification results
    for line in metrics.classification_report(test_y, pred).split("\n"):
        print line
'''
 Your code for Q9
'''
# read the lexicon
print "Q9:"
lexicon = dict()

# read postive words
with open("negative-words.txt", "r") as in_file:
    for line in in_file.readlines():
        if not line.startswith(";") and line != "\n":
            lexicon[line.strip()] = -1

# read negative words
with open("positive-words.txt", "r") as in_file:
    for line in in_file.readlines():
        if not line.startswith(";") and line != "\n":
            lexicon[line.strip()] = 1

# define a custom tokenizer
def tokenization(text):
    # replace mention 
    text = sub("@[^ ]+", " ", text)
    # replace hashtags with space
    text = sub("#[^ ]+", " ", text)
    # replace RT (retweet) with space
    text = sub("RT", " ", text)
    # replace URL with space
    text = sub("http[^ ]+", " ", text)

    p = compile("[^a-z]")
    # conver the text to lower case and split by non-alphabetic characters
    # also remove "" due to tokenizing multple spaces
    return [token for token in p.split(text.lower()) if token != ""]

vocab = lexicon.keys()
vectorizer = TfidfVectorizer(tokenizer=tokenization, max_df=0.8, stop_words='english', vocabulary=vocab)
X = vectorizer.fit_transform(dataset["text"])
train_x, test_x, train_y, test_y = train_test_split(X, dataset["score"], test_size=0.2, stratify=dataset["score"], random_state=123)
#test_x3 = keep_sentiment_terms(test_x)
clf = SGDClassifier()
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
# classification results
for line in metrics.classification_report(test_y, pred_y).split("\n"):
    print line

print "END"
