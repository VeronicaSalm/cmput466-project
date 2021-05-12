# ----------------------------------------------------------------------------
# partition_twitter_data.py
#
# Iterates over a folder, which is expected to contain subfolders in the form:
#   2020-01/, 2020-02/, 2020-03/, ... 
# containing tweet files. This file copies all data to a new folder and
# partitions it into two-week (14 day) chunks. The copied tweets are reduced
# to contain only necessary information for the topic modelling (to save space).
# The new folders are of the form yyyy-dd_yyyy-dd which indicates the start and
# end date, inclusive.
# Also runs VADER on the tweets.
# ----------------------------------------------------------------------------

import argparse
import json
import os
import sys
import csv
from glob import glob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Argparse for input jsonl file, must not be gzipped!
parser = argparse.ArgumentParser(description="Generates tweet files for tweets matching the given criteria.")

parser.add_argument("input_path",
                    help="The path to the input folder",
                    type = str)

parser.add_argument("dest",
                    help="The name of the destination folder where the result should be stored, expected to not exist",
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

def process_tweet_file(fpath, fdest):
    '''
    Arguments:
        - fpath (string): path to .jsonl file to process
        - fdest (string): path to the folder where the result should be stored

    Returns:
        - None, but writes each tweet to a separate to a file
    '''
    analyser = SentimentIntensityAnalyzer()
    dest_path = os.path.join(fdest, fpath.split('/')[-1])
    fobj = open(dest_path, "w")
    global cnt 
    with open(fpath, "r") as json_file:
        #print("Processing '{}'...".format(fpath))
        line = json_file.readline()

        while line:
            d = json.loads(line)

            tweet_ID = d[ID]
            
            if RT_STATUS in d:
                tweet = {ID: tweet_ID,
                     CREATED_AT: d[CREATED_AT],
                     FULL_TEXT: d[RT_STATUS][FULL_TEXT],
                     RT_STATUS: True}
            else:
                tweet = {ID: tweet_ID,
                     CREATED_AT: d[CREATED_AT],
                     FULL_TEXT: d[FULL_TEXT],
                     RT_STATUS: False}
            score = analyser.polarity_scores(d["full_text"])
            tweet["vader_score"] = score
            print(json.dumps(tweet), file=fobj)
            # read the next line of file
            line = json_file.readline()
            cnt += 1

    fobj.close()

if __name__ == "__main__":
    args = parser.parse_args()
    
    if os.path.isdir(args.dest):
        print(f"The directory '{args.dest}' already exists! Remove it and try again.")
        sys.exit()
    else:
        os.system(f"mkdir {args.dest}")

    MAX_DAYS = 14
    START = None
    CURR = None
    END = None
    DAYS = 0
    PREV = None
    PATHS = []

    directories = sorted(glob(os.path.join(args.input_path, "*/")))
    for d in directories:
        files = sorted(os.listdir(d))
        for f in files:
            fpath = os.path.join(d, f)
            day = f.split("-")[5]  # extract day number
            

            if START == None:
                DAYS = 0
                START = f
                CURR = day
            elif day != CURR:
                # day has increased
                DAYS += 1
                CURR = day
                #print(day,DAYS, START, CURR, END)
            
            if DAYS == MAX_DAYS:
                cnt = 0
                end = PREV.split("-")
                END = end[3] + "-" + end[4] + "-" + end[5]
                start = START.split("-")
                START = start[3] + "-" + start[4] + "-" + start[5]
                
                dest = os.path.join(args.dest, START + "_" + END)
                os.system(f"mkdir {dest}")

                print(f"Processing Tweets From {START} to {END}...")
                print(PATHS[0], "to", PATHS[-1])

                for path in PATHS:
                    process_tweet_file(path, dest)
                print(f"Processed {cnt} total Canadian tweets in the given period.")

                START = None
                PATHS = []


            PATHS.append(fpath)

            # keep track of the previous file, for setting the end date
            PREV = f 

    # This means there is a partial week still to process
    if START != None:
        cnt = 0
        end = f.split("-")
        END = end[3] + "-" + end[4] + "-" + end[5]
        start = START.split("-")
        START = start[3] + "-" + start[4] + "-" + start[5]
        
        dest = os.path.join(args.dest, START + "_" + END)
        os.system(f"mkdir {dest}")

        print(f"Processing Tweets From {START} to {END}...")
        print(PATHS[0], "to", PATHS[-1])
        for path in PATHS:
            process_tweet_file(path, dest)
        print(f"Processed {cnt} total Canadian tweets in the given period.")

    print("Done processing all provided tweets.")

