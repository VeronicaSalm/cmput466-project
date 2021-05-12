# nmf.py
#
# Can be used to run NMF with a specified value of k,
# and reports the coherence. 

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

parser.add_argument('--k', type=int, nargs='?', default = 10,
                    help='the number of components (topics) to generate, defaults to 10')

parser.add_argument('--num_words', type=int, nargs='?', default = 15,
                    help='the number of words to generate for each topic, defaults to 15')

def plot_top_words(model, feature_names, n_top_words, title, dest):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
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


def main():
    '''
    Driver code for the project.
    '''
    args = parser.parse_args()
    
    if os.path.isdir(args.dest):
        print(f"The directory '{args.dest}' already exists! Removing...")
        os.system(f"rm -r {args.dest}")
        os.system(f"mkdir {args.dest}")
    else:
        os.system(f"mkdir {args.dest}")

    dm = DataManager(args.train_path, 'twitter')
    print("Loading data...")

    if os.path.exists("tweet_cache.cache"):
        os.system("rm tweet_cache.cache")
    start = time.perf_counter()
    dm.load_data("tweet_cache.cache")
    end = time.perf_counter()
    if settings.DEBUG: print(f"Preparing the data (loading, normalizing) took {end-start:0.4f} seconds.")

    print("Training word2vec...")
    coh = Coherence()
    coh.mapWordsToVecs(dm.get_all_data())

    # trying a bunch of values of k to compare the coherence
    print("Training NMF model:")
    start = time.perf_counter()
    # Train the model with the param choice.
    transformed, model, vectorizer = dm.run_nmf(num_components=args.k)
    # Compute the resulting accuracy on the validation set.
    end = time.perf_counter()
    if settings.DEBUG: print(f"        Training took {end-start:0.4f} seconds.")
    
    tfidf_feature_names = vectorizer.get_feature_names()
    topics = plot_top_words(model, tfidf_feature_names, args.num_words, 
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


    print("Storing words to output...")
    dm.save_words_as_json(topics, os.path.join(args.dest, "topics.json"))

    
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
    

# Entry point to the run NMF program.
if __name__ == '__main__':
    main()

