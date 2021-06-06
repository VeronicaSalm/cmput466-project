# tune_topics_nmf.py
#
# Tunes the number of nmf topics for a given dataset based
# on coherence. Uses gensim's nmf implementation for this, since 
# sklearn's version doesn't have a way to run coherence.
#
# Final topics, coherence, and word clouds are saved 
# to the results path.
# -------------------------------------------------------------
from coherence import Coherence
from collections import defaultdict
import csv, json
import time
import argparse
import os, sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
import numpy as np
from operator import itemgetter
import pickle


from DataManager import DataManager

# Project-wide constants, file paths, etc.
import settings

parser = argparse.ArgumentParser(
    description='Runs NMF on the twitter data, reports the resulting coherence, and prints the results to the output path.'
    )

parser.add_argument('--train_path', type=str, nargs='?', default = "../TwitterDataset/data/Jan27-Feb02/",
                    help='the path to the twitter dir, defaults to ../TwitterDataset/data/Jan27-Feb02/')


parser.add_argument('--dest', type=str, nargs='?', default = "canadian_results",
                    help='the path to the results file, defaults to canadian_results')

parser.add_argument('--num_words', type=int, nargs='?', default = 15,
                    help='the number of words to generate for each topic, defaults to 15')

def plot_top_words(model, feature_names, n_topics, n_top_words, title, dest):
    # fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    
    # Compute the number of rows using the provided n_topics
    if n_topics % 5 == 0:
        n_rows = (n_topics//5) 
    else:
        n_rows = (n_topics//5) + 1
    
    print(f"Generating topic distribution visualization with {n_rows} rows...")
    fig, axes = plt.subplots(n_rows, 5, figsize=(45, n_rows * 10), sharex=True)
    axes = axes.flatten()
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        topics[topic_idx+1] = dict()
        topics[topic_idx+1]["features"] = dict()
        for f,w in zip(top_features, weights):
            topics[topic_idx+1]["features"][f] = w
        topics[topic_idx+1]["words"] = top_features

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(os.path.join(dest, "topic_distribution"))
    return topics


def tune(texts):
    # Use Gensim's NMF to get the best num of topics via coherence score

    # Create a dictionary
    # In gensim a dictionary is a mapping between words and their integer id
    dictionary = Dictionary(texts)

    # Filter out extremes to limit the number of features
    dictionary.filter_extremes(
        no_below=3,
        no_above=0.85,
        keep_n=5000
    )

    # Create the bag-of-words format (list of (token_id, token_count))
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Create a list of the topic numbers we want to try
    topic_nums = list(np.arange(5, 50 + 1, 5))
    print(topic_nums)

    # Run the nmf model and calculate the coherence score
    # for each number of topics
    coherence_scores = []

    for num in topic_nums:
    
        start = time.perf_counter()
        if settings.DEBUG: print(f"Trying {num} topics...", end=" ")
        nmf = Nmf(
            corpus=corpus,
            num_topics=num,
            id2word=dictionary,
            chunksize=2000,
            passes=5,
            kappa=.1,
            minimum_probability=0.01,
            w_max_iter=300,
            w_stop_condition=0.0001,
            h_max_iter=100,
            h_stop_condition=0.001,
            eval_every=10,
            normalize=True,
            random_state=42
        )

        # Run the coherence model to get the score
        cm = CoherenceModel(
            model=nmf,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        coh = round(cm.get_coherence(), 5)
        coherence_scores.append(coh)
        print(f"Coherence = {coh}", end=", ")
        end = time.perf_counter()
        if settings.DEBUG: print(f"took {end-start:0.4f} seconds.")

    # Get the number of topics with the highest coherence score
    scores = list(zip(topic_nums, coherence_scores))
    best = sorted(scores, key=itemgetter(1), reverse=True)[0]
    print("Best number of topics and coherence found:", best)

    return best


def main():
    '''
    Driver code for the project.
    '''
    args = parser.parse_args()
    
    if os.path.isdir(args.dest):
        answer = input(f"The directory '{args.dest}' already exists! Remove it? (y/n): ")
        if answer.startswith("y"):
            print("Removing the destination directory...")
            os.system(f"rm -r {args.dest}")
            os.system(f"mkdir {args.dest}")
        else:
            print("Quitting...")
            sys.exit()
    else:
        os.system(f"mkdir {args.dest}")
    
    # Save folders we have already tuned to avoid unnecessary computation
    # Load the saved paths into the "known" dictionary
    known = {}
    topic_file = "num_topics.csv"
    if os.path.exists(topic_file):
        with open(topic_file, "r") as f:
            reader = csv.reader(f)
            reader.__next__() # skip header
            for row in reader:
                known[row[0]] = [int(row[1]), float(row[2])]
    
        print("Already tuned topics for:")
        print(known)
    else:
        with open(topic_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(["Path", "Number of Topics", "Coherence (Gensim)"])
    
    # Create the topic file (or reopen it for appending)
    topic_fobj = open(topic_file, "a")
    topic_writer = csv.writer(topic_fobj)
        

    # Create the data manager obj and load in the twitter data
    dm = DataManager(args.train_path, 'twitter')
    print("Loading data...")

    start = time.perf_counter()
    dm.load_data()
    end = time.perf_counter()
    if settings.DEBUG: print(f"Preparing the data (loading, normalizing) took {end-start:0.4f} seconds.")
    
    tweets = dm.get_all_data()
    texts = [t[1].split() for t in tweets]
    
    if args.train_path in known:
        # We have already tuned this data, skip tuning
        num_topics, gensim_coh = known[args.train_path]
    else:
        # Tune to find the best number of topics for this data (using coherence)
        num_topics, gensim_coh = tune(texts)

    # Use the resulting number of topics to run NMF using the sklearn implementation
    coh = Coherence()
    coh.mapWordsToVecs(dm.get_all_data())

    print("Training NMF model:")
    start = time.perf_counter()
    # Use the number of topics found by tuning
    transformed, model, vectorizer = dm.run_nmf(num_components=num_topics)
    end = time.perf_counter()
    if settings.DEBUG: print(f"        Training took {end-start:0.4f} seconds.")

    tfidf_feature_names = vectorizer.get_feature_names()

    topics = plot_top_words(model, tfidf_feature_names, num_topics, args.num_words, 
               'Topics in NMF model (Frobenius norm)', args.dest)


    print("Generating statistics...", end=" ")
    word_counts = defaultdict(int)
    document_frequency = defaultdict(int)
    docs = dm.get_all_data()
    # assumes the transformed docs keeps the same order (no reason to think otherwise)
    vader_scores = {"neg" : defaultdict(list), "compound" : defaultdict(list),
                    "neu" : defaultdict(list), "pos" : defaultdict(list) }
    docs_by_topic = defaultdict(list)
    
    # maps user ID to a list of their tweets and their screen name
    users = dict()
    
    # maps term to a set of users who mentioned that term
    term_to_user = defaultdict(set)
    topic_to_user = defaultdict(set)
    for doc, topic_results in zip(docs, transformed):
        tokens = doc[1].split() # normalized tokens
        ID = doc[2]
        data = dm.get_tweet_data_by_id(ID)
        uid = data["user_id"]
        
        for t in tokens:
            word_counts[t] += 1

        for t in set(tokens):
            document_frequency[t] += 1
            term_to_user[t].add(uid)
        

        tid = topic_results.argmax() + 1 # argmax will index by 0, we want ids starting from 1
        topic_to_user[tid].add(uid)
        
        if uid not in users:
            users[uid] = {"screen_name" : data["screen_name"], "tweets" : [], "topics" : set()}

        users[uid]["tweets"].append(ID)
        users[uid]["topics"].add(tid)

        vader_scores["neg"][tid].append(data["vader_score"]["neg"])
        vader_scores["neu"][tid].append(data["vader_score"]["neu"])
        vader_scores["pos"][tid].append(data["vader_score"]["pos"])
        vader_scores["compound"][tid].append(data["vader_score"]["compound"])

        # tid-1 to index correctly (added one earlier)
        docs_by_topic[tid].append([ID, topic_results[tid-1]])
        
    num_tweets = len(dm.get_all_data())
    num_users = len(users)
    print("Done!")

    print("Saving user stats...", end=" ")
    sorted_users = sorted(users.items(), key = lambda x : len(x[1]["tweets"]), reverse=True)
    user_path = os.path.join(args.dest, "user_stats.tsv")
    with open(user_path, "w") as fobj:
        writer = csv.writer(fobj, delimiter="\t")
        writer.writerow(["User ID", "Screen Name", "Number of Tweets", "Number of Topics", "Topics Discussed"])
        for u, d in sorted_users:
            tids = ", ".join([str(t) for t in d["topics"]])
            row = [u, d["screen_name"], len(d["tweets"]), len(d["topics"]), tids]
            writer.writerow(row)
    print("Done!")
    
    # save word counts, in descending order
    print(f"Saving token counts...", end=" ")
    ordered_word_counts = sorted(word_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    word_count_path = os.path.join(args.dest, "token_stats_general.csv")
    with open(word_count_path, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["Normalized Token", "Number of Occurences", "Number of Tweets", "Percentage of Tweets", "Number of Users", "Percentage of Users"])
        for token, count in ordered_word_counts:
            docs_percent = round((document_frequency[token] / num_tweets) * 100, 5)
            users_percent = round((len(term_to_user[token]) / num_users) * 100, 5)
            writer.writerow([token, count, document_frequency[token], docs_percent, len(term_to_user[token]), users_percent])
    print("Done!")

    # For each topic and for each of its top words,
    # save the number of occurences of each word (total count) and the number
    # of total documents in which that word appears (should be <= total count)
    topic_word_count_path = os.path.join(args.dest, "token_stats_by_topic.csv")
    print("Generating term and document frequencies...", end=" ")
    with open(topic_word_count_path, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["Topic", "Normalized Token", "Number of Occurences", "Number of Tweets", "Percentage of Tweetts", "Number of Users", "Percentage of Users"])
        for tid in topics.keys():
            words = topics[tid]['words']
            for token in words:
                docs_percent = round((document_frequency[token] / num_tweets) * 100, 5)
                users_percent = round((len(term_to_user[token]) / num_users) * 100, 5)
                writer.writerow([tid, token, word_counts[token], document_frequency[token], docs_percent, len(term_to_user[token]), users_percent])
    print("Done!")

    # aggregated vader scores, by topic
    print("Generating aggregated vader scores by topic...", end=" ")
    vader_path = os.path.join(args.dest, "vader_scores.csv")
    all_vader_scores = defaultdict(list)
    with open(vader_path, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["Topic Number", "Compound (Mean)", "Compound (Std Dev)", "Positive (Mean)", "Positive (Std Dev)", "Negative (Mean)", "Negative (Std Dev)", "Neutral (Mean)", "Neutral (Std Dev)"])
        for tid in topics.keys():
            row = [tid]
            for score in ["compound", "pos", "neg", "neu"]:
                score_list = vader_scores[score][tid]
                sd = round(np.std(score_list), 5)
                mu = round(np.mean(score_list), 5)
                row.append(mu)
                row.append(sd)
                all_vader_scores[score].extend(score_list)
            writer.writerow(row)
    print("Done!")
            
    # documents by topic 
    print("Counting the number of documents by topic...", end=" ")
    doc_path = os.path.join(args.dest, "topic_stats.csv")
    total = 0
    with open(doc_path, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["Topic Number", "Number of Tweets", "Percentage of Tweets", "Number of Users", "Percentage of Users"])
        for tid in topics.keys():
            user_list = topic_to_user[tid]
            num_topic_users = len(user_list)
            count = len(docs_by_topic[tid])
            docs_percent = round((count / num_tweets) * 100, 5)
            users_percent = round((num_topic_users / num_users) * 100, 5)
            writer.writerow([tid, count, docs_percent, num_topic_users, users_percent])
            total += count 
    print("Done!")
    
    # top 10 documents from each topic 
    print("Saving the top documents from each topic...", end=" ")
    doc_dir = os.path.join(args.dest, "top_15_tweets_per_topic")
    num_top_docs = 15
    os.system(f"mkdir {doc_dir}")
    for tid in topics.keys():
        doc_path = os.path.join(doc_dir, f"topic_{tid:02d}.csv")
        with open(doc_path, "w") as fobj:
            writer = csv.writer(fobj)
            # TODO: add document author (username and ID) and link to profile
            writer.writerow(["Tweet ID", "Weight", "Duplicates", "Tweet Text"])
            doc_list = docs_by_topic[tid]
            doc_list = sorted(doc_list, key=lambda x: x[1], reverse = True)
            num_valid = 0
            used = dict()
            rows = []
            for i in range(len(doc_list)):
                ID, weight = doc_list[i]
                data = dm.get_tweet_data_by_id(ID)
                tweet_text = data["full_text"]
                if tweet_text in used:
                    used[tweet_text] += 1
                    continue
                elif num_valid == num_top_docs:
                    # don't add any new documents
                    continue
                retweet = "Original"
                if data["retweeted_status"]:
                    retweet = "Retweet"
                rows.append([ID, round(weight, 5), retweet, 0, tweet_text])
                used[tweet_text] = 1
                num_valid += 1
            for r in rows:
                text = r[4] # full text
                r[3] = used[text] # count of times appeared
                writer.writerow(r)
    print("Done!")
    
    vocab_size = len(word_counts)
    print(f"Vocab Size: {vocab_size}")
    print(f"Tweets: {num_tweets}")
    print(f"Users: {num_users}")
    overall_stat_path = os.path.join(args.dest, "overall_statistics.json")
    with open(overall_stat_path, "w") as fobj:
        obj = {"Total Tweets" : num_tweets,
               "Vocabulary Size (Normalized Tokens)" : vocab_size,
               "Total Users" : num_users,
               "Vader Scores" : dict()}
        scores =  ["compound", "pos", "neg", "neu"]
        pretty_scores =  ["Compound", "Positive", "Negative", "Neutral"]
        for score, pretty in zip(scores, pretty_scores):
            score_list = all_vader_scores[score]
            obj["Vader Scores"][pretty] = dict()
            sd = round(np.std(score_list), 5)
            mu = round(np.mean(score_list), 5)
            
            obj["Vader Scores"][pretty]["Mean"] = mu 
            obj["Vader Scores"][pretty]["Standard Deviation"] = sd 
        print(json.dumps(obj, sort_keys=True, indent=4), file=fobj)
    
    print("Finding coherence of each topic:")
    coh_list = []
    for topic in topics:
        words = topics[topic]["words"]
        topic_coherence = coh.getCoherence(words)
        topics[topic]["coherence"] = topic_coherence
        coh_list.append(topic_coherence)
    avg_coh = sum(coh_list) / len(coh_list)
    print("    Average Coherence =", avg_coh)

    wordclouds_path = os.path.join(args.dest, "wordclouds")
    os.system(f"mkdir {wordclouds_path}")
    
    # generate word cloud for each topic    
    print("Generating word clouds for each topic...", end=" ")
    for topic in topics.keys():
        features = topics[topic]["features"]
        word_model = WordCloud(width = 800, height = 800, background_color = 'white',
                        min_font_size = 10, include_numbers= True, relative_scaling=0.7, stopwords="", prefer_horizontal=True)
        wordcloud = word_model.generate_from_frequencies(features)
        wordcloud.to_file("{}/topic{}.png".format(wordclouds_path, '{0:02}'.format(topic)))
    print("Done!")

    print("Storing words to output...", end=" ")
    topics["num_topics"] = int(num_topics)
    topics["gensim_coherence"] = float(gensim_coh)
    if args.train_path not in known:
        topic_writer.writerow([args.train_path, num_topics, gensim_coh])
    print("Done!")
    
    
    print("Pickling models...", end=" ")
    model_path = os.path.join(args.dest, "models")
    os.system(f"mkdir {model_path}")
    with open(os.path.join(model_path, "transformed.pickle"), "wb") as f:
        pickle.dump(transformed, f) 
    with open(os.path.join(model_path, "model.pickle"), "wb") as f:
        pickle.dump(model, f) 
    with open(os.path.join(model_path, "vectorizer.pickle"), "wb") as f:
        pickle.dump(vectorizer, f) 
    dm.save_words_as_json(topics, os.path.join(model_path, "topics.json"))
    print("Done!")
    
    # code to load a model from pickled file
    # with open(os.path.join(model_path, "transformed.pickle"), "rb") as f:
    #     transformed2 = pickle.load(f)

# Entry point to the run NMF program.
if __name__ == '__main__':
    main()

