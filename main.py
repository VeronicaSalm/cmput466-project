# --------------------------------------------------
# main.py
#
# Main entry point for the whole project.
# --------------------------------------------------
import csv
import time
import numpy as np

from DataManager import DataManager
from coherence import Coherence
from newsgroup_util import get_confusion_matrix, get_accuracy, get_precision, get_recall

# Project-wide constants, file paths, etc.
import settings

def main():
    '''
    Driver code for the project.
    '''

    # Title.
    print('\n--------------------------------------------------------------------------------')
    print('Using Topic Modelling to Understand Public Discourse on Twitter\n')
    print('Project by: Veronica Salm, Ian DeHaan, Noah Gergel, Siyuan Yu, Xiang Zhang')
    print('--------------------------------------------------------------------------------\n')

    # Intro blurb.
    print('In this file we are going to demonstrate some of the functionality of the project.')
    print('It won\'t do everything, in particular won\'t run on the twitter dataset, but will demo the algorithms and topic modelling.')

    print('\nFirst, we will load in the 20newsgroup dataset.')
    
    # Load in the 20newsgroup dataset.
    print('Loading 20newsgroup data ...', end='')

    dm = DataManager(settings.NEWSGROUP_DIR, 'newsgroup')
    dm.load_data()

    print(' done!')

    # Show example document.
    print('\nWe are now going to train our two topic modelling algorithms: LDA, and NMF.')
    
    # Train the algorithms.
    print('Training LDA ...', end='', flush=True)
    _, lda, lda_vectorizer = dm.run_lda(num_components=20)
    print(' done!')

    print('Training NMF ...', end='', flush=True)
    _, nmf, nmf_vectorizer = dm.run_nmf(num_components=20)
    print(' done!')

    # Print example topics.
    print('Extracting the topics ...', end='')

    nmf_top_words = dm.get_top_words_per_topic(nmf, nmf_vectorizer, 10)
    lda_top_words = dm.get_top_words_per_topic(lda, lda_vectorizer, 10)

    print(' done!')

    print("\nThe following are example topics from both LDA and NMF:")
    print('LDA:', ', '.join(lda_top_words[0]))
    print('NMF:', ', '.join(nmf_top_words[0]))

    real_topics = [x[0] for x in dm.get_all_data(True)]

    # Vectorize the data for LDA.
    lda_test_vectorized = lda_vectorizer.transform([doc[1] for doc in dm.get_all_data(True)])
    predicted = lda.transform(lda_test_vectorized)
    lda_dominant_topic = np.argmax(predicted, axis=1)

    # Same for NMF.
    nmf_test_vectorized = nmf_vectorizer.transform([doc[1] for doc in dm.get_all_data(True)])
    predicted = nmf.transform(nmf_test_vectorized)
    nmf_dominant_topic = np.argmax(predicted, axis=1)

    # Get the confusion matrices.
    lda_conf_mat = get_confusion_matrix(lda_dominant_topic, real_topics)
    nmf_conf_mat = get_confusion_matrix(nmf_dominant_topic, real_topics)

    # Print the accuracies.
    print('\nLDA accuracy:', get_accuracy(lda_conf_mat))
    print('NMF accuracy:', get_accuracy(nmf_conf_mat))

    # Get the average precision and recall.
    lda_recall, nmf_recall = get_recall(lda_conf_mat), get_recall(nmf_conf_mat)
    lda_precision, nmf_precision = get_precision(lda_conf_mat), get_precision(nmf_conf_mat)
    lda_avg_recall, lda_avg_precision = 0, 0
    nmf_avg_recall, nmf_avg_precision = 0, 0
    for i in lda_recall:
        lda_avg_recall += lda_recall[i]
        lda_avg_precision += lda_precision[i]
        nmf_avg_recall += nmf_recall[i]
        nmf_avg_precision += nmf_precision[i]
    
    print('LDA Avg. precision and recall:', lda_avg_precision/20, lda_avg_recall/20)
    print('NMF Avg. precision and recall:', nmf_avg_precision/20, nmf_avg_recall/20)

# Entry point to the program.
if __name__ == '__main__':
    main()

