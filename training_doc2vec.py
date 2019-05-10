import argparse
import os  
import re
import codecs
import preprocessor as p
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
import pickle
import numpy as np
import gensim
from sklearn.preprocessing import scale
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from sklearn import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to prepared dataset")
    parser.add_argument('-o', '--output', help="path to output directory", default='pre-trained_models')
    args = parser.parse_args()
    if args.input is None:
        parser.print_usage()
        exit()
    return args

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def loading_datasets(female , male):
    with codecs.open( female, 'r' ,encoding='utf-8' ) as f:
        female_tweets = f.read().split('\n')
    with codecs.open( male , 'r' ,encoding='utf-8' ) as f:
        male_tweets = f.read().split('\n')
    return female_tweets , male_tweets

def Preprocessing(tweets , _stopwords):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG
                  , p.OPT.SMILEY,p.OPT.NUMBER,p.OPT.RESERVED)
    lemmatizer = WordNetLemmatizer()
    cleaned_tweets = p.clean(tweets)
    word_list = cleaned_tweets.split()    
    mod_tweet = []
    for word in word_list:
        if word.lower() in _stopwords:
            continue
        mod_tweet.append(lemmatizer.lemmatize(word.lower()))      
    return ' '.join(mod_tweet)

def training():
    stopwords_list = {'en': set(stopwords.words('english')) , 'es': set(stopwords.words('spanish')) }
    args = get_args()
    input_dir = os.path.normpath(args.input)
    out = os.path.normpath(args.output)
    mkdir(out)
    for dir in os.listdir(input_dir):
        print("Working on Language: " , dir )
        out_dir = os.path.join(out , dir)
        mkdir(out_dir)
        female , male = loading_datasets(os.path.join(input_dir,dir,'human_female.txt'), 
                                    os.path.join(input_dir,dir,'human_male.txt') )
        print("Human-male train-set size: ", len(male))
        print("Human-female train-set size: ", len(female))

        #training for male and female
        print("prepare train ...")
        train_set = male + female
        train_set = [Preprocessing(text , stopwords_list[dir]) for text in male] + \
                    [Preprocessing(text , stopwords_list[dir]) for text in female]
        train_labels = ['male' for _ in male] + ['female' for _ in female]
        
        doc2vec_train_set=[TaggedDocument(words=train_set[i].split(),tags=[train_labels[i]]) for i in range(0,len(train_set))]
        print("Build vocab and train doc2vec .... ")
        n_dim = 300
        doc2vec_model = Doc2Vec(min_count=1,size=n_dim)
        doc2vec_model.build_vocab([x for x in tqdm(doc2vec_train_set)])
        for epoch in range(20):
            print("Training epoch ", epoch)
            doc2vec_model.train(doc2vec_train_set, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.iter,)
        doc2vec_model.save(os.path.join( out_dir , 'doc2vec_male_female.d2v'))
        print('------------------------------------')

training()
