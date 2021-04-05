# --------------------------------------------------
# main.py
#
# Main entry point for the whole project.
# --------------------------------------------------
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import csv
import time

from DataManager import DataManager
from coherence import Coherence

# Project-wide constants, file paths, etc.
import settings

def get_data_for_LDA(dm):
    """
    Vectorizes the current fold data for CV.
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
    Driver code for the project.
    '''
    # Extract the data for LDA and divide into 10 folds
    dm = DataManager(settings.TWITTER_DIR, 'twitter')
    print("Loading data...")
    start = time.perf_counter()
    dm.load_data()
    dm.divide_into_folds(10)
    end = time.perf_counter()
    if settings.DEBUG: print(f"Preparing the data (loading, dividing into folds) took {end-start:0.4f} seconds.")

    best_k = None
    best_likelihood = -float("inf")
    possible_k_values = [10, 25, 50]


    fout = open("results.txt", "a")
    out_writer = csv.writer(fout)
    # out_writer.writerow(["Model", "k", "Average Likelihood", "Number of Documents", "Source"])

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

    print("Finding coherence of some stuff:")
    coh = Coherence()
    coh.mapWordsToVecs(dm.get_all_data())
    print("Coherence of 'god' and 'jesus' =", coh.getCoherence(["god", "jesus"]))
    print("Coherence of 'god', 'jesus', and 'linux' =", coh.getCoherence(["god", "jesus", "linux"]))
    
    print("Running NMF:")
    _, model, vectorizer = dm.run_nmf()

    print("Finding top words:")
    top_words = dm.get_top_words_per_topic(model, vectorizer, 10)
    print(top_words)

    print("Finding coherence of each topic:")
    for topic in top_words:
        print(topic, coh.getCoherence(top_words[topic]))


# Entry point to the program.
if __name__ == '__main__':
    main()

