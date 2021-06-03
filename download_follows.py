"""
file: download_follows.py

For every user in the input file, downloads their followers and
friends (followed users). The output is stored as a jsonl file,
where each line maps a user id to that user's followers and
friends.
"""

import os
import tweepy as tw
import csv
import argparse
import json

parser = argparse.ArgumentParser(
    description='Performs document classification using Naive Bayes and prints the resulting accuracy.'
    )

parser.add_argument('--user_path', type=str, nargs='?', default = "canadian_locations_22k.csv",
                    help='the path to the input user csv file, defaults to canadian_locations_22k.csv')
parser.add_argument('--output_path', type=str, nargs='?', default ="follow_results.json",
                    help='the path to the output file, defaults to follow_results.json')

consumer_key = "Le0nf1fQaVJC5ZIPX2gyp8NdO"
consumer_secret = "oFGdG2xyo3WSmaBez2PmAjscKqtjChmWjUVkpKdu2ftaPhNw0N"
access_token = "1353814267573624833-jMQTXcJoFyYiKf3qgU7YebPu2mvcOL"
access_token_secret = "nhhxMDzuXqhjiLw163b9ySPAP0OLTBLTKTCPl5fR5BGco"


ID = "id"
FRIENDS = "friends"
FOLLOWERS = "followers"
FOLLOWERS_COUNT = "followers_count"
FRIENDS_COUNT = "friends_count"
ERROR = "encountered_error"

if __name__ == "__main__":
    args = parser.parse_args()
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    users = []
    with open(args.user_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            uid = int(row[0])
            users.append(uid)

    # avoid querying users whose data we have already extracted
    already_processed = set()
    errors = 0
    if os.path.exists(args.output_path):
        print(f"The output path {args.output_path} already exists. Reading already processed users...")
        with open(args.output_path, "r") as f:
            data = f.readlines()
            for line in data:
                user = json.loads(line)
                already_processed.add(int(user[ID]))
                if user[ERROR]:
                    errors += 1

        print(f"Found {len(already_processed)} existing users.")


    with open(args.output_path, "a") as f:
        cnt = 0
        for user in users:
            error = False
            if user in already_processed:
                print(f"Skipping {user}...")
                continue
            
            try:
                followers = api.followers_ids(user)
            except tw.TweepError:
                error = True
                followers = []
            
            try:
                friends = api.friends_ids(user)
            except tw.TweepError:
                error = True
                friends = []


            obj = {
                    ID : user,
                    FOLLOWERS_COUNT : len(followers),
                    FRIENDS_COUNT : len(friends),
                    FOLLOWERS : followers,
                    FRIENDS : friends,
                    ERROR : error
                  }

            print(json.dumps(obj), file=f)

            cnt += 1
            if error:
                errors += 1
            

    print(f"Processed {cnt} users. Out of all users processed, encountered {errors} errors.")
    print(f"The output is stored in {args.output_path}")
