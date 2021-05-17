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
import csv
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
    print(n_topics, n_topics % 5)
    if n_topics % 5 == 0:
        n_rows = (n_topics//5) 
    else:
        n_rows = (n_topics//5) + 1

    fig, axes = plt.subplots(n_rows, 5, figsize=(30, n_rows * 8), sharex=True)
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

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
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
    topic_nums = list(np.arange(5, 75 + 1, 5))
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
            for row in reader:
                known[row[0]] = [int(row[1]), float(row[2])]
    
        print("Already tuned topics for:")
        print(known)
    
    # Create the topic file (or reopen it for appending)
    topic_fobj = open(topic_file, "a")
    topic_writer = csv.writer(topic_fobj)
        

    # Create the data manager obj and load in the twitter data
    dm = DataManager(args.train_path, 'twitter')
    print("Loading data...")

    if os.path.exists("tweet_cache.cache"):
        os.system("rm tweet_cache.cache")
    start = time.perf_counter()
    dm.load_data("tweet_cache.cache")
    end = time.perf_counter()
    if settings.DEBUG: print(f"Preparing the data (loading, normalizing) took {end-start:0.4f} seconds.")
    
    texts = dm.get_all_data()
    texts = [t[1].split() for t in texts]
    
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

    print("Finding top words:")
    print("TOP WORDS:")
    for t in topics.keys():
        print(f"    {t}: {topics[t]['words']}")
    
    print("Finding coherence of each topic:")
    coh_list = []
    for topic in topics:
        words = topics[topic]["words"]
        topic_coherence = coh.getCoherence(words)
        print(topic, topic_coherence)
        topics[topic]["coherence"] = topic_coherence
        coh_list.append(topic_coherence)
    avg_coh = sum(coh_list) / len(coh_list)
    print("    Average Coherence =", avg_coh)

    wordclouds_path = os.path.join(args.dest, "wordclouds")
    os.system(f"mkdir {wordclouds_path}")
    
    # generate word cloud for each topic    
    for topic in topics.keys():
        """
        features = topics[topic]["features"]
        keywords = ""
        for f, w in features.items():
            freq = [f]*int(w*100)
            keywords += " ".join(freq)
            keywords += " "
        keywords.strip()
        """
        #keywords = " ".join(topics[topic]["words"])
        features = topics[topic]["features"]
        word_model = WordCloud(width = 800, height = 800, background_color = 'white',
                        min_font_size = 10, include_numbers= True, relative_scaling=0.7, stopwords="", prefer_horizontal=True)
        wordcloud = word_model.generate_from_frequencies(features)
        wordcloud.to_file("{}/topic{}.png".format(wordclouds_path, '{0:02}'.format(topic)))
    
    topics["num_topics"] = int(num_topics)
    topics["gensim_coherence"] = float(gensim_coh)

    print("Storing words to output...")
    if args.train_path not in known:
        topic_writer.writerow([args.train_path, num_topics, gensim_coh])

    dm.save_words_as_json(topics, os.path.join(args.dest, "topics.json"))


# Entry point to the run NMF program.
if __name__ == '__main__':
    main()

