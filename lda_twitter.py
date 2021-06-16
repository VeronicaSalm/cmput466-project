# -------------------------------------------------------------
# Runs LDA on the input and generates word clouds.
# -------------------------------------------------------------
from coherence import Coherence
from collections import defaultdict
import csv, json
import time
import argparse
import os, sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import pickle


from DataManager import DataManager

# Project-wide constants, file paths, etc.
import settings

parser = argparse.ArgumentParser(
    description='Runs LDA on the twitter data, reports the resulting coherence, and prints the results to the output path.'
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

    # Create the data manager obj and load in the twitter data
    dm = DataManager(args.train_path, 'twitter')
    print("Loading data...")

    start = time.perf_counter()
    dm.load_data()
    end = time.perf_counter()
    if settings.DEBUG: print(f"Preparing the data (loading, normalizing) took {end-start:0.4f} seconds.")

    tweets = dm.get_all_data()
    texts = [t[1].split() for t in tweets]

    num_topics = 6 # hard code


    # Use the resulting number of topics to run LDA using the sklearn implementation
    coh = Coherence()
    coh.mapWordsToVecs(dm.get_all_data())

    print("Training LDA model:")
    start = time.perf_counter()
    # Use the number of topics found by tuning
    transformed, model, vectorizer = dm.run_lda(num_components=num_topics)
    end = time.perf_counter()
    if settings.DEBUG: print(f"        Training took {end-start:0.4f} seconds.")

    tfidf_feature_names = vectorizer.get_feature_names()

    topics = plot_top_words(model, tfidf_feature_names, num_topics, args.num_words,
               'Topics in LDA model (Frobenius norm)', args.dest)

    wordclouds_path = os.path.join(args.dest, "wordclouds")
    os.system(f"mkdir {wordclouds_path}")

    # generate word cloud for each topic
    print("Generating word clouds for each topic...", end=" ")
    for topic in topics.keys():
        features = topics[topic]["features"]
        word_model = WordCloud(width = 800, height = 800, background_color = 'white',
                        min_font_size = 10, include_numbers= True, relative_scaling=0.7, stopwords="", prefer_horizontal=True)
        wordcloud = word_model.generate_from_frequencies(features)
        wordcloud.to_file("{}/topic_{}.png".format(wordclouds_path, '{0:02}'.format(topic)))
    print("Done!")

# Entry point to the run LDA program.
if __name__ == '__main__':
    main()

