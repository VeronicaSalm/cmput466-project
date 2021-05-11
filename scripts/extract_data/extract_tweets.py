# ----------------------------------------------------------------------------
# extract_tweets.py
#
# Extracts a week worth of tweets from the dataset, filtered to only include
# tweets in English that are not retweets/replies and contain no media.
# ----------------------------------------------------------------------------

import argparse
import json
import os
import sys
import csv
import nltk, string
import spacy
import kenlm
from collections import defaultdict
from hashtag_master.word_breaker.main import segment_word
from hashtag_master.neural_ranker.main import create_neural_ranking_model
from hashtag_master.neural_ranker.rerank import rerank
from textblob import TextBlob

# For Hashtag Master
model_type = "mse_multi"
print("Loading language model.")
language_model = kenlm.LanguageModel("hashtag_master/data/small_gt.bin")
print("Done.")

print("Training Neural Ranking Model.")
neural_ranking_model, feature_extractor = create_neural_ranking_model(model_type)
print("Done")

# Alternate language identification methods:
# TextBlob works, but unfortunately the scale of our project exceeds API limits
# from textblob import TextBlob

# Argparse for input jsonl file, must not be gzipped!
parser = argparse.ArgumentParser(description="Generates tweet files for tweets matching the given criteria.")

# Note: it's expected that all tweets from start date to end date are in the given input folder
# Otherwise, the script will need to be run multiple times to capture the tweets in each
# separate folder.
parser.add_argument("input_path",
                    help="The input folder containing the unzipped .jsonl tweet files",
                    type = str)

parser.add_argument("start_date",
                    help="Start date, in the format YYYY-MM-DD-HH , HH is the hour from 00 to 23",
                    type = str)

parser.add_argument("end_date",
                    help="End date for the tweet range, in the format YYYY-MM-DD-HH , HH is the hour from 00 to 23. Expected to be larger than start date.",
                    type = str)

parser.add_argument("dest",
                    help="The destination folder where the result should be stored",
                    type = str)

# Variables for index fields
RT_STATUS = "retweeted_status"
ENTITIES = "entities"
CREATED_AT = "created_at"
FULL_TEXT = "full_text"
MEDIA = "media"
URLS = "urls"
REPLY = "in_reply_to_user_id"
ID = "id"

def get_features(candidates):
    """
    Extracts the feature vector for the candidate segmentations.
    """
    global feature_extractor
    best_cand = candidates[0]
    feats = []
    for seg in candidates:
        fv = feature_extractor._get_features_for_segmentation(seg, best_cand)
        feats.append(fv)
    return feats

def segment(t):
    """
    Attempts to segment the hashtag represented by "t" using the HashtagMaster tool.
    """
    if t.startswith("#") and len(t) > 1:
        global language_model
        global neural_ranking_model
        top_k = 5
        candidates = segment_word(t.lstrip("#"), 5, language_model)
        if len(candidates) > 1:
            # there are multiple candidates, run the reranking algorithm
            feats = get_features(candidates)
            # print(feats)
            reranked_segs = rerank(candidates, feats, neural_ranking_model, model_type)
            best = reranked_segs[0].split()
        elif len(candidates) == 1:
            best = candidates[0].split()
        else:
            print(t, candidates)
            raise Exception("Error, not enough candidates for hashtag '{}'!".format(t))

        return best
    else:
        return t

def tokenize(text):
    " Removes URLs, segments hashtags, and produces a list of tokens for language detection."
    text = text.replace(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b','')
    text = text.replace(r'www\.\S+\.com','')
    tokens = text.split()
    
    output = []
    for t in tokens:
        if t.startswith("#"):
            output.extend(segment(t))
        else:
            output.append(t)

    return output

# initialize the set of english words
with open("words_dictionary.json", "r") as fobj:
    english_words = json.loads(fobj.read())

def detect_english(tokens, threshold = 0.65):
    '''
    Determines whether or not an input text is in English.

    Arguments:
        - tokens (list): list of tokens
        - threshold (float): the percentage of words that must be English in order
                             for the text to be considered English (defaults to 70%)

    Returns:
        - boolean: True if the text is classified as English, False otherwise
    '''
    global english_words

    normalized = []
    for w in tokens:
        w = w.lower()

        # remove punctuation
        w = w.translate(str.maketrans('', '', string.punctuation))
        if w:
            normalized.append(w)

    N = len(normalized)
    E = 0

    # this case avoids division by 0 if all words are removed
    if N == 0:
        return False

    # count the number of English words
    for n in normalized:
        if n in english_words:
            E += 1
    
    print("Detect English:", E, N, E/N)
    # Return True if the number of English words is above the threshold
    if (E/N) > threshold:
        return True
    else:
        return False

def process_tweet_file(fpath, fdest):
    '''
    Extracts English tweets which are original content only (i.e., not
    retweets or replies) which contains no media (images/videos) or urls.

    This function also counts the number of valid tweets found and prints
    this information to the terminal during execution.

    Arguments:
        - fpath (string): path to .jsonl file to process
        - fdest (string): path to the folder where the result should be stored

    Returns:
        - None, but writes each tweet to a separate to a file
    '''
    cnt = 0
    bad = 0
    dest_path = os.path.join(fdest, fpath.split('/')[-1])
    fobj = open(dest_path, "w")
    
    nlp = spacy.load("en_core_web_sm")
    with open(fpath, "r") as json_file:
        print("Processing '{}'...".format(fpath))
        line = json_file.readline()

        cnt = 0
        while line:
            d = json.loads(line)

            tweet_ID = d[ID]

            text = d[FULL_TEXT]
            #text = "C'est une longue phrase avec de nombreux jetons français, que j'inclus pour voir ce que fait le segmenteur avec de longues phrases dans une autre langue"
            # text = "Bueno cuando le dan la corona a Raven???"
            doc = tokenize(text)
            print(doc)
            if detect_english(doc) or len(doc) < 10:
                # this tweet is classified as English, so store it to the output
                print("English!")
                print(json.dumps(d), file=fobj)
                cnt += 1
            else:
                bad += 1
                print("Not English!")
            """
            if len(doc) < 10:
                print(text)
                lang = TextBlob(text)
                if lang.detect_language() == "en":
                    cnt += 1
                    print("English!")
                    print(json.dumps(d), file=fobj)
                else:
                    bad += 1
                    print("Not English!")
            else:
                if detect_english(doc):
                    # this tweet is classified as English, so store it to the output
                    print("English!")
                    print(json.dumps(d), file=fobj)
                    cnt += 1
                else:
                    print(text)
                    lang = TextBlob(text)
                    if lang.detect_language() == "en":
                        cnt += 1
                        print("English!")
                        print(json.dumps(d), file=fobj)
                    else:
                        bad += 1
                        print("Not English!")
            
            """
            """
            doc = nlp(text)
            print(doc)
            propn = 0
            total = 0
            pos = defaultdict(int)
            dep = defaultdict(int)
            for token in doc:
                print(token.text, token.pos_, token.dep_)

                if token.pos_ == "PROPN":
                    propn += 1
                total += 1
                pos[token.pos_] += 1
                dep[token.dep_] += 1
            print(d[FULL_TEXT])
            print(propn, total, propn / total)
            print(pos)
            print(doc.has_annotation("DEP"))
            print(dep)
            """
        
            """
            # Check that this is not a retweet / reply and that it contains no
            # no media elements or urls.
            if d[REPLY] == None and len(d[ENTITIES][URLS]) == 0 \
                    and MEDIA not in d[ENTITIES] and RT_STATUS not in d:
                # this is not a retweet or reply and there are no media elements
                if detect_english(d[FULL_TEXT]):
                    # this tweet is classified as English, so store it to the output
                    cnt += 1
                    tweet = {ID: tweet_ID,
                             CREATED_AT: d[CREATED_AT],
                             FULL_TEXT: d[FULL_TEXT]}
                    print(json.dumps(tweet), file=fobj)
                else:
                    # this is an invalid tweet
                    bad += 1

            else:
                # this is an invalid tweet
                bad += 1
            """ 
            # read the next line of file
            line = json_file.readline()

    # Inform the user of the results from this file
    print(f"Found {cnt} valid and {bad} invalid tweets.")
    fobj.close()

if __name__ == "__main__":
    args = parser.parse_args()

    # Using the start and end dates, construct the file names.
    # This relies on the input filenames to be structured a particular way
    # (as they are in the Twitter dataset from Chen et. al.)
    start = "coronavirus-tweet-id-" + args.start_date + ".jsonl"
    end = "coronavirus-tweet-id-" + args.end_date + ".jsonl"

    # Iterate over all files in the input directory, and extract valid English
    # tweets from each one.
    if os.path.isdir(args.input_path):
        for f in sorted(os.listdir(args.input_path)):
            fpath = os.path.join(args.input_path, f)
            if f >= start and f <= end:
                process_tweet_file(fpath, args.dest)
    else:
        print("The input path argument must be a valid directory.")

    print("Done processing all provided tweets.")

