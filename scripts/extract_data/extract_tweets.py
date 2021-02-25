# Extracts a week worth of tweets from the dataset, filtered to only include
# tweets in English that are not retweets/replies and contain no media.

import argparse
import json
import os
import sys
import csv
# TextBlob works, but unfortunately I'm running into API limits
#from textblob import TextBlob
import nltk, string

# Argparse for input json file, must be unzipped!
parser = argparse.ArgumentParser(description="Generates tweet files for tweets matching the given criteria.")

# Note: it's expected that all tweets from start date to end date are in the given input folder
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

with open("words_dictionary.json", "r") as fobj:
    english_words = json.loads(fobj.read())

def detect_english(text, threshold = 0.70):
    global english_words

    words = text.split()
    normalized = []
    for w in words:
        w = w.lower()
        if w.startswith("#"):
            continue
        w = w.translate(str.maketrans('', '', string.punctuation))
        if w:
            normalized.append(w)

    N = len(normalized)
    E = 0
    if N == 0:
        return False
    
    for n in normalized:
        if n in english_words:
            E += 1

    if (E/N) > threshold:
        return True
    else:
        return False


def process_tweet_file(fpath, fdest):
    """
    Arguments: 
        fpath: path to .jsonl file to process
        fdest: path to the folder where the result should be stored

    Returns:
        None, but writes each tweet to a separate to a file
    """
    cnt = 0
    bad = 0
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
                    cnt += 1
                    dest_path = os.path.join(fdest, str(tweet_ID)+".json")
                    with open(dest_path, "w") as fobj:
                        tweet = {"id": tweet_ID, 
                                  "created_at": d["created_at"], 
                                  "full_text": d["full_text"]}
                        print(json.dumps(tweet, indent=4), file=fobj)

                """
                lang = TextBlob(text)
                if lang.detect_language() == "en":
                    print(d["full_text"], lang.detect_language())
                    #print(d["in_reply_to_user_id"])
                    #print(RT_STATUS in d)
                    #print(json.dumps(d[ENTITIES], indent=4))
                    cnt += 1
                else:
                    print(d["full_text"], lang.detect_language())
                """
            else:
                bad += 1
            
            # read the next line of file
            line = json_file.readline()

    print(f"Found {cnt} valid and {bad} invalid tweets.")

if __name__ == "__main__":
    args = parser.parse_args()
    start = "coronavirus-tweet-id-" + args.start_date + ".jsonl"
    end = "coronavirus-tweet-id-" + args.end_date + ".jsonl"
    print(start)
    print(end)

    if os.path.isdir(args.input_path):
        for f in sorted(os.listdir(args.input_path)):
            fpath = os.path.join(args.input_path, f)
            if f >= start and f <= end:
                process_tweet_file(fpath, args.dest)
    else:
        print("The input path argument must be a valid directory.")

    print("Done processing all provided tweets.")

    #indexed.close()
