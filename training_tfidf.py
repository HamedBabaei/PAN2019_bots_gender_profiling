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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to prepared dataset")
    parser.add_argument('-o', '--output', help="path to output directory", default='pre-trained_models')
    parser.add_argument('-ft', type=int, help="frequency threshold (default=5)", default=5)
    args = parser.parse_args()
    if args.input is None:
        parser.print_usage()
        exit()
    return args

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def loading_datasets( bot , female , male):
    with codecs.open( bot , 'r' ,encoding='utf-8' ) as f:
        bot_tweets = f.read().split('\n')
    with codecs.open( female, 'r' ,encoding='utf-8' ) as f:
        female_tweets = f.read().split('\n')
    with codecs.open( male , 'r' ,encoding='utf-8' ) as f:
        male_tweets = f.read().split('\n')
    return bot_tweets , female_tweets , male_tweets

def Preprocessing(tweets , _stopwords):
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER,p.OPT.RESERVED)
    lemmatizer = WordNetLemmatizer()
    cleaned_tweets = p.clean(tweets)
    word_list = cleaned_tweets.split()    
    mod_tweet = []
    for word in word_list:
        if word.lower() in _stopwords:
            continue
        mod_tweet.append(lemmatizer.lemmatize(word.lower()))      
    return ' '.join(mod_tweet)

def represent_tweet(tweets ):
    tokens = TweetTokenizer().tokenize(tweets)
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def extract_vocabulary(tweets , ft):
    occurrences=defaultdict(int)
    for tweet in tweets:
        tweet_occurrences = represent_tweet(tweet)
        for ngram in tweet_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=tweet_occurrences[ngram]
            else:
                occurrences[ngram]=tweet_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary

def training():
    stopwords_list = {'en': set(stopwords.words('english')) , 'es': set(stopwords.words('spanish')) }
    args = get_args()
    ft = args.ft
    input_dir = os.path.normpath(args.input)
    out = os.path.normpath(args.output)
    mkdir(out)
    for dir in os.listdir(input_dir):
        print("Working on Language: " , dir )
        out_dir = os.path.join(out , dir)
        mkdir(out_dir)
        bot , female , male  = loading_datasets( os.path.join(input_dir,dir,'bot.txt') , 
                                    os.path.join(input_dir,dir,'human_female.txt'), 
                                    os.path.join(input_dir,dir,'human_male.txt') )
        print("\t Bot train-set size: ", len(bot))
        print("\t Human-male train-set size: ", len(male))
        print("\t Human-female train-set size: ", len(female))

        print("\t Preparing Human&Bot dataset for tfidf vectorizer .... ")
        train_set = male + female + bot
        train_labels = ['human' for _ in male] + ['human' for _ in female] + ['bot' for _ in bot]
        tfidf_train = [Preprocessing(text , stopwords_list[dir]) for text in train_set]
     
        print("\t Extracting Human&Bot vocabulary .... ")
        vocab = extract_vocabulary(tfidf_train , ft )
        print("\t Train TF-IDF for Human&Bot Profiling .... ")
        vectorizer  = TfidfVectorizer(vocabulary=vocab,norm='l2', strip_accents=False,sublinear_tf=True)
        tfidf_train_data = vectorizer.fit_transform(tfidf_train)
        pickle.dump(vectorizer.vocabulary_,open( os.path.join( out_dir , "tfidf_human_bot.pkl"),"wb"))
        print("\tTrained TF-IDF vocabulary saved to :", str(os.path.join( out_dir , "tfidf_human_bot.pkl")))

        print("\t Preparing Male&Female dataset for tfidf vectorizer .... ")
        train_set = male + female
        train_labels = ['male' for _ in male] + ['female' for _ in female]
        tfidf_train = [Preprocessing(text , stopwords_list[dir]) for text in train_set]
        print("\t Extracting Male&Female vocabulary .... ")
        vocab = extract_vocabulary(tfidf_train , ft )
        print("\t Train TF-IDF for Male&Female Profiling .... ")
        vectorizer  = TfidfVectorizer(vocabulary=vocab,norm='l2', strip_accents=False,sublinear_tf=True)
        tfidf_train_data = vectorizer.fit_transform(tfidf_train)
        pickle.dump(vectorizer.vocabulary_,open( os.path.join( out_dir , "tfidf_male_female.pkl"),"wb"))
        print("\t Trained TF-IDF vocabulary saved to :", str(os.path.join( out_dir , "tfidf_male_female.pkl")))
        print('\t ------------------------------------')

training()
