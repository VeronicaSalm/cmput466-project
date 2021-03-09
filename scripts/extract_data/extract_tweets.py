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

# initialize the set of english words
with open("words_dictionary.json", "r") as fobj:
    english_words = json.loads(fobj.read())

def detect_english(text, threshold = 0.70):
    '''
    Determines whether or not an input text is in English.

    Arguments:
        - text (string): the text to classify as English or Not English
        - threshold (float): the percentage of words that must be English in order
                             for the text to be considered English (defaults to 70%)

    Returns:
        - boolean: True if the text is classified as English, False otherwise
    '''
    global english_words

    words = text.split()
    normalized = []
    for w in words:
        w = w.lower()

        # ignore hashtags, which will rarely be a single English word
        # hashtags like "#WearAMask" are very common, but the dictionary won't
        # include them.
        if w.startswith("#"):
            continue

        # remove punctuation
        w = w.translate(str.maketrans('', '', string.punctuation))
        if w:
            normalized.append(w)

    N = len(normalized)
    E = 0

    # this case avoids division by 0 if all words are removed
    # (e.g., because the tweet was entirely hashtags)
    if N == 0:
        return False

    # count the number of English words
    for n in normalized:
        if n in english_words:
            E += 1

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

    with open(fpath, "r") as json_file:
        print("Processing '{}'...".format(fpath))
        line = json_file.readline()

        while line:
            d = json.loads(line)

            tweet_ID = d["id"]

            # Check for replies
            if d["in_reply_to_user_id"] == None and len(d[ENTITIES]["urls"]) == 0 \
                    and "media" not in d[ENTITIES] and RT_STATUS not in d:
                # this is not a retweet or reply and there are no media elements
                if detect_english(d["full_text"]):
                    # this tweet is classified as English, so store it to the output
                    cnt += 1
                    tweet = {"id": tweet_ID,
                              "created_at": d["created_at"],
                              "full_text": d["full_text"]}
                    print(json.dumps(tweet), file=fobj)

            else:
                # this is an invalid tweet
                bad += 1

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
