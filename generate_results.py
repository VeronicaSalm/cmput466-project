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

if __name__ == "__main__":
    args = parser.parse_args()
    
    if os.path.isdir(args.dest):
        print(f"The directory '{args.dest}' already exists! Removing...")
        os.system(f"rm -r {args.dest}")
        os.system(f"mkdir {args.dest}")
    else:
        os.system(f"mkdir {args.dest}")

    directories = sorted(glob(os.path.join(args.input_path, "*/")))
    for d in directories:
        dir_name = d.split("/")[1]
        print(dir_name)
        result = os.system(f"python3 tune_topics_nmf.py --train_path {d} --dest {args.dest}/{dir_name}") 
        if result:
            print(f"Encountered an error when running directory '{d}'!")
            sys.exit()

    print("Done processing all provided tweets.")

