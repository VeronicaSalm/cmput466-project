#!/bin/bash
python3 extract_tweets.py ../../../2020-01/ 2020-01-27-00 2020-02-02-23 ../../../TwitterDataset/data/Jan27-Feb02/
python3 extract_tweets.py ../../../2020-02/ 2020-02-01-00 2020-02-02-23 ../../../TwitterDataset/data/Jan27-Feb02/
python3 extract_tweets.py ../../../2020-03/ 2020-03-16-00 2020-03-22-23 ../../../TwitterDataset/data/Mar16-Mar22/
python3 extract_tweets.py ../../../2020-04/ 2020-04-27-00 2020-05-03-23 ../../../TwitterDataset/data/Apr27-May03/
python3 extract_tweets.py ../../../2020-05/ 2020-05-01-00 2020-05-03-23 ../../../TwitterDataset/data/Apr27-May03/
python3 extract_tweets.py ../../../2020-07/ 2020-07-06-00 2020-07-12-23 ../../../TwitterDataset/data/Jul06-Jul12/

