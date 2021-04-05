# --------------------------------------------------
# nmf_cross_validation_twitter.py
#
# Main entry point for the whole project.
# --------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import csv
import time
import argparse
import os

from DataManager import DataManager

# Project-wide constants, file paths, etc.
import settings

parser = argparse.ArgumentParser(
    description='Performs k-fold cross validation using NMF on the Twitter dataset, to tune the number of topics K using likelihood as the evaluation metric.'
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

def get_data_for_NMF(dm):
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
    all_data = dm.get_all_fold_data() + dm.get_all_validation_data()
    tfidf_vect = TfidfVectorizer(
        max_df=0.95,
        min_df=1,
        max_features=10000,
        stop_words='english'
    )
    vectorized = tfidf_vect.fit_transform([x[1] for x in all_data])
    train, validate = vectorized[:len(dm.get_all_fold_data())], vectorized[len(dm.get_all_fold_data()):]

    return train, validate

def run_NMF_for_CV(train, k, alpha=1.34, beta_loss='kullback-leibler', l1_ratio=0.66, solver='mu', num_iterations=1000):
    """
    Runs NMF on the provided training data.

    Arguments:
        train: Data to run on, vectorized by a CountVectorizer previously
        k (int): The number of components or topics NMF generates.
        alpha (float): The normalization constant for NMF.
        beta_loss (string): The beta-loss function to use.
        l1_ratio (float): The ratio in which NMF uses L1 regularization vs. L2.
        solver (string): Numerical solver to use for NMF.
        num_iterations (int): The number of iterations to run for.

    Returns:
        - nmf_model: the NMF model trained on the input data
    """
    if beta_loss not in ['kullback-leibler', 'frobenius', 'itakura-saito']:
        msg = 'Invalid beta-loss function given for NMF.\n'
        options = "', '".join(['kullback-leibler', 'frobenius', 'itakura-saito'])
        msg += f"Please use one of: '{options}'."
        raise Exception(msg)

    if solver not in ['mu', 'cd']:
        msg = 'Invalid numerical solver given for NMF.\n'
        msg += "Please use one of: 'mu', 'cd'."
        raise Exception(msg)

    if l1_ratio < 0 or l1_ratio > 1:
        raise Exception('Invalid L1 ration given for NMF.\nPlease make sure l1_ratio is in the range [0, 1].')

    if settings.DEBUG: print("    Running NMF...")
    nmf_model = NMF(
        init='nndsvda',
        n_components=k,
        random_state=1,
        beta_loss=beta_loss,
        solver=solver,
        max_iter=num_iterations,
        alpha=alpha,
        l1_ratio=l1_ratio
    )

    nmf_model.fit(train)

    return nmf_model



def main():
    '''
    Runs cross validation on the input Twitter data.
    '''
    args = parser.parse_args()

    # Extract the data for NMF and divide into 10 folds
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
            train, validate = get_data_for_NMF(dm)
            start = time.perf_counter()
            # Train the model with the param choice.
            nmf_model = run_NMF_for_CV(train, k)
            # Compute the resulting accuracy on the validation set.
            likelihood = nmf_model.score(validate)
            end = time.perf_counter()
            if settings.DEBUG: print(f"        likelihood = {likelihood}")
            if settings.DEBUG: print(f"        Training took {end-start:0.4f} seconds.")

            likelihoods.append(likelihood)


        avg_likelihood = sum(likelihoods) / len(likelihoods)
        out_writer.writerow(["NMF", k, avg_likelihood, len(dm.get_all_fold_data()), settings.TWITTER_DIR])
        if settings.DEBUG: print(f"    avg_likelihood = {avg_likelihood}")

        if avg_likelihood > best_likelihood:
            best_likelihood = avg_likelihood
            best_k = k

    print(f"Best average likelihood found was {best_likelihood} with parameter value k={best_k}")
    fout.close()

# Entry point to the cross validation (NMF) program.
if __name__ == '__main__':
    main()

