# --------------------------------------------------
# lda_cross_validation_twitter.py
#
# Performs cross validation using LDA on the Twitter
# dataset, to tune the number of topics k.
# --------------------------------------------------
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import csv
import time
import argparse
import os

from DataManager import DataManager

# Project-wide constants, file paths, etc.
import settings

parser = argparse.ArgumentParser(
    description='Performs k-fold cross validation using LDA on the Twitter dataset, to tune the number of topics K using likelihood as the evaluation metric.'
    )

parser.add_argument('--train_path', type=str, nargs='?', default = "../TwitterDataset/data/Jan27-Feb02/",
                    help='the path to the training file, defaults to ../TwitterDataset/data/Jan27-Feb02/')
parser.add_argument('--results_path', type=str, nargs='?', default ="results/results_jan27-feb02.tmp",
                    help='the path to the results file, defaults to results/results_jan27-feb02.tmp')
parser.add_argument('--cache_path', type=str, nargs='?', default ="tweet_cache_jan27-feb02.cache",
                    help='the path to the file where tweets were cached from a previous run, defaults to tweet_cache_jan27-feb02.cache')
parser.add_argument('--num_folds', type=int, nargs='?', default = 10,
                    help='the number of folds for cross validation, defaults to 10')

parser.add_argument('-t', '--topic_numbers', type=int, nargs='*', default = [5, 10, 15, 25],
                    help='the number of folds for cross validation, defaults to 10')

def get_data_for_LDA(dm):
    """
    Vectorizes the current fold data for CV.

    Arguments:
        - dm (DataManager): an instance of the DataManager class, on which load_data has
                            and divide_into_folds has already been called

    Returns:
        - train: the vectorized training data
        - validate: the vectorized validation data
    """
    # ignore everything else, just grab the strings
    vectorizer = CountVectorizer(analyzer='word',
                                min_df=1,                        # minimum df
                                stop_words='english',             # remove stop words
                                lowercase=True,
                                token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                )
    all_data = dm.get_all_fold_data() + dm.get_all_validation_data()
    vectorized = vectorizer.fit_transform(x[1] for x in all_data)
    train, validate = vectorized[:len(dm.get_all_fold_data())], vectorized[len(dm.get_all_fold_data()):]

    return train, validate

def run_LDA_for_CV(train, k, doc_topic_prior=0.5, topic_word_prior=0.1, learning_decay=0.4, learning_offset=5, batch_size=135, num_iterations=5):
    """
    Runs LDA on the provided training data.

    Arguments:
        train: Data to run on. Same format as received from get_all_data, get_all_folds, etc.
        k: the number of topics to create
        doc_topic_prior (float): The doc_topic_prior hyperparam.
        topic_word_prior (float): The topic_word_prior hyperparam.
        learning_offset (float): The learning_offset hyperparam.
        batch_size (int): The batch_size hyperparam.
        num_iterations (int): The number of iterations to run for.

    Returns:
        - lda_model: the LDA model trained on the input data
    """

    if settings.DEBUG: print("    Running LDA...")
    lda_model = LatentDirichletAllocation(n_components=k,
                                          doc_topic_prior=doc_topic_prior,
                                          topic_word_prior=topic_word_prior,
                                          learning_method='online',
                                          learning_decay=learning_decay,
                                          learning_offset=learning_offset,
                                          max_iter=num_iterations,
                                          batch_size=batch_size,
                                          )
    lda_model.fit(train)

    return lda_model



def main():
    '''
    Runs cross validation on the input Twitter data.
    '''
    args = parser.parse_args()

    # Extract the data for LDA and divide into 10 folds
    dm = DataManager(args.train_path, 'twitter')
    if settings.DEBUG: print("Loading data...")

    # Time the process of loading in the data.
    start = time.perf_counter()

    # Load the data (possibly from the cache, if it exists)
    dm.load_data(args.cache_path)
    # The number of folds is passed in as a command-line arg
    dm.divide_into_folds(args.num_folds)
    end = time.perf_counter()
    if settings.DEBUG: print(f"Preparing the data (loading, dividing into folds) took {end-start:0.4f} seconds.")

    # Initialize the best k and best likelihood, along with the list of k values to try
    best_k = None
    best_likelihood = -float("inf")

    # Get the list of topic numbers to try as a command line arg too.
    possible_k_values = args.topic_numbers

    # Store the results to the result path. Add the headers if the file doesn't exist yet.
    if not os.path.exists(args.results_path):
        fout = open(args.results_path, "w")
        out_writer = csv.writer(fout)
        out_writer.writerow(["Model", "k", "Average Likelihood", "Number of Documents", "Source"])
    else:
        fout = open(args.results_path, "w")
        out_writer = csv.writer(fout)

    # Run cross validation once for each parameter value
    for k in possible_k_values:

        if settings.DEBUG: print(f"Trying k={k} components...")

        # We will create a list of accuracies for each validation set
        likelihoods = []
        for i in range(dm.get_num_folds()):
            if settings.DEBUG: print(f"    Iteration {i+1}/{dm.get_num_folds()}")

            # Update the validation fold.
            dm.set_validation(i)

            # Retrieve the training data and validation set.
            train, validate = get_data_for_LDA(dm)
            start = time.perf_counter()
            # Train the model with the param choice.
            lda_model = run_LDA_for_CV(train, k)
            # Compute the resulting accuracy on the validation set.
            likelihood = lda_model.score(validate)
            end = time.perf_counter()
            if settings.DEBUG: print(f"        likelihood = {likelihood}")
            if settings.DEBUG: print(f"        Training took {end-start:0.4f} seconds.")

            likelihoods.append(likelihood)


        avg_likelihood = sum(likelihoods) / len(likelihoods)
        out_writer.writerow(["LDA", k, avg_likelihood, len(dm.get_all_fold_data()), settings.TWITTER_DIR])
        if settings.DEBUG: print(f"    avg_likelihood = {avg_likelihood}")

        if avg_likelihood > best_likelihood:
            best_likelihood = avg_likelihood
            best_k = k

    print(f"Best average likelihood found was {best_likelihood} with parameter value k={best_k}")
    fout.close()

# Entry point to the cross validation (LDA) program.
if __name__ == '__main__':
    main()

