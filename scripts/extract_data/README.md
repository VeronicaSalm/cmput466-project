# Extract Tweets

This folder contains a script `extract_tweets.py`, which iterates over all tweet files in the original Twitter dataset. It extracts only tweets which are likely to be in English and contain original content.    

For each tweet in the time period, `extract_tweets.py` first confirms that the tweet:
1. Is not a retweet of another tweet or a reply to another tweet. **Justification:** This reduces the overall number of tweets in the corpus (over 500 million!) to a more manageable number by eliminating duplicates and removing tweets that may require context to analyze.
2. Contains no media such as images or URLs. **Justification:** This was largely done to reduce the sample size. Also, URLs and media elements may be difficult to analyze effectively using LDA/NMF, which is primarily text-based, so it makes sense to remove them. Also, the topic of a tweet may not be ascertainable without considering the media elements it contains (e.g., a picture of lots of people wearing masks with a caption "Love this! Stay safe!". The topic of this tweet is masks/mask-wearing, but our algorithms would have no way to determine this information).
3. Is in English. **Justification**: We would like to analyze only English tweets. If non-English language tweets are included, our topic-modelling algorithms may perform poorly and/or be difficult to analyze.

We first attempted to use existing language-detection libraries (textblob, spacy-langdetect) to determine if a tweet is in English, but ran into API limits: these libraries only allow a few thousand queries each day, where we needed to check over 1 million tweets over the weeks of interest.

## extract_tweets.py

#### Running Instructions

```
python3 extract_tweets.py input_path start_date end_date dest
```
* `input_path`: The input folder containing the unzipped .jsonl tweet files containing the original tweets. It is expected that all files in this folder have the name format `"coronavirus-tweet-id-YYYY-MM-DD-HH.jsonl"`, which allows the files to be ordered by date. This is the native file name format from the original Twitter dataset from Chen et. al. (https://github.com/echen102/COVID-19-TweetIDs), so this is a reasonable assumption.
* `start_date`: Start date, in the format YYYY-MM-DD-HH (HH is the hour from 00 to 23). The start date is inclusive.
* `end_date`: End date for the tweet range, in the format YYYY-MM-DD-HH. The end date must be larger than start date and is also inclusive.
* `dest`: The destination folder where the output files should be stored. One file will be produced for every input hour (or, every input tweet file produces exactly one output file).

The resulting `.jsonl` files contain selected fields from the input tweets in the following format:
```
 tweet = {"id": tweet_ID,
          "created_at": creation_date,
          "full_text": tweet_text}
```
Each tweet is stored on a separate line of the output file(s).


## extract_tweets.sh

This script runs `extract_tweets.py` multiple times to extract all tweets for our dataset, storing them in folders in the TwitterDataset repository (https://github.com/VeronicaSalm/TwitterDataset). This script uses relative paths which would need to be changed if the script were to be run again.

#### Running Instructions
```
chmod u+x extract_tweets.sh
./extract_tweets.sh
```
