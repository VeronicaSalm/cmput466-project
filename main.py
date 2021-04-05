# --------------------------------------------------
# main.py
#
# Main entry point for the whole project.
# --------------------------------------------------
import csv
import time

from DataManager import DataManager
from coherence import Coherence

# Project-wide constants, file paths, etc.
import settings

def main():
    '''
    Driver code for the project.
    '''
    # Extract the data for LDA and divide into 10 folds
    dm = DataManager(settings.TWITTER_DIR, 'twitter')
    print("Loading data...")
    dm.load_data("tweet_cache.cache")

    print("Training word2vec...")
    coh = Coherence()
    coh.mapWordsToVecs(dm.get_all_data())

    best_k = None
    best_coh = 0

    # trying a bunch of values of k to compare the coherence
    for k in range(1, 21):
        print("Training LDA model:")
        start = time.perf_counter()
        # Train the model with the param choice.
        transformed, model, vectorizer = dm.run_lda(num_components=k)
        # Compute the resulting accuracy on the validation set.
        end = time.perf_counter()
        if settings.DEBUG: print(f"        Training took {end-start:0.4f} seconds.")

        print("Finding top words:")
        top_words = dm.get_top_words_per_topic(model, vectorizer, 10)
        print(top_words)


        print("Finding coherence of each topic:")
        coh_list = []
        for topic in top_words:
            topic_coherence = coh.getCoherence(top_words[topic])
            print(topic, topic_coherence)
            coh_list.append(topic_coherence)

        avg_coh = sum(coh_list) / len(coh_list)

        print("Average Coherence =", avg_coh)

        if avg_coh > best_coh:
            best_coh = avg_coh
            best_k = k

    print(f"Best average coherence found was {best_coh} with parameter value k={best_k}")


# Entry point to the program.
if __name__ == '__main__':
    main()

