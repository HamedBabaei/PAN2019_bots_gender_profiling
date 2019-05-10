import argparse
import os  
import re
import codecs
import preprocessor as p
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pickle
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to prepared dataset")
    parser.add_argument('-o', '--output', help="path to output directory", default='pre-trained_models')
    parser.add_argument('-ft', type=int, default=5, help='frequency threshold (default=5)')
    parser.add_argument('-n', type=int, default=4, help="n-gram order (default=4)" )
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
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG
                  ,p.OPT.EMOJI, p.OPT.SMILEY,p.OPT.NUMBER, p.OPT.RESERVED)
    lemmatizer = WordNetLemmatizer()
    cleaned_tweets = p.clean(tweets)
    word_list = cleaned_tweets.split()  
    mod_tweet = []
    for word in word_list:
        if word.lower() in _stopwords:
            continue
        mod_tweet.append(lemmatizer.lemmatize(word.lower())) 
    return ' '.join(mod_tweet)

def ngram_represent_text(text,n):
    if n>0:
        tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def ngram_extract_vocabulary(texts,n,ft):
    occurrences=defaultdict(int)
    for text in texts:
        text_occurrences=ngram_represent_text(text,n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=text_occurrences[ngram]
            else:
                occurrences[ngram]=text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary

def training():

    stopwords_list = {'en': set(stopwords.words('english')) , 'es': set(stopwords.words('spanish')) }
    args = get_args()
    n = args.n
    ft = args.ft
    input_dir = os.path.normpath(args.input)
    out = os.path.normpath(args.output)
    mkdir(out)
    for dir in os.listdir(input_dir):
        print("Working on Language: " , dir )
        out_dir = os.path.join(out , dir)
        mkdir(out_dir)
        female , male = loading_datasets(os.path.join(input_dir,dir,'human_female.txt'), 
                                    os.path.join(input_dir,dir,'human_male.txt') )
        print("\t Human-male train-set size: ", len(male))
        print("\t Human-female train-set size: ", len(female))

        #training for male and female
        print("\t Preparing Male&Female dataset for N-Gram vectorizer ...")
        train_set = male + female
        train_set = [Preprocessing(text , stopwords_list[dir]) for text in male] + \
                    [Preprocessing(text , stopwords_list[dir]) for text in female]
        train_labels = ['male' for _ in male] + ['female' for _ in female]
        
        print("\t Extracting Male&Female Vocabluary .... ")
        ngram_vocabulary = ngram_extract_vocabulary(train_set , n , ft)
        print("\t Train N-Gram for Male&Female Profiling ....")
        ngram_vectorizer = CountVectorizer(strip_accents=False, analyzer='char',ngram_range=(n,n),lowercase=False,vocabulary=ngram_vocabulary)
        ngram_train_data = ngram_vectorizer.fit_transform(train_set)
        pickle.dump(ngram_vectorizer.vocabulary_,open( os.path.join( out_dir , "ngram_male_female.pkl"),"wb"))
        print("\t Trained N-gram Vectorizer vocabulary saved to :", str(os.path.join( out_dir , "ngram_male_female.pkl")))
        print('\t ------------------------------------')

training()
