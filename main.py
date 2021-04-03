# --------------------------------------------------
# main.py
#
# Main entry point for the whole project.
# --------------------------------------------------

from DataManager import DataManager

# Project-wide constants, file paths, etc.
import settings

def get_data():
    dm = DataManager(settings.TWITTER_DIR, 'twitter')
    print("Loading data...")
    dm.load_data()
    dm.divide_into_folds(10)
    
    # ignore everything else, just grab the strings
    vectorizer = CountVectorizer(analyzer='word',
                                min_df=10,                        # minimum df
                                stop_words='english',             # remove stop words
                                lowercase=True,
                                token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                )
    all_data = dm.get_all_fold_data() + dm.get_all_validation_data()
    vectorized = vectorizer.fit_transform(x[1] for x in all_data)
    train, validate = vectorized[:len(dm.get_all_fold_data())], vectorized[len(dm.get_all_fold_data()):]

    return train, validate, dm

def main():
    '''
    Driver code for the project.
    '''

    # Just simply initialize the data manager class for the
    # newsgroup dataset and load in the data.
    dm = DataManager(settings.TWITTER_DIR, 'twitter')
    print("Loading data...")
    dm.load_data()

    # Call this method to divide the data into folds
    # There are three modes:
    #    ROUND_ROBIN = 0
    #    RANDOM = 1
    #    EVEN_SPLIT = 2 (simple partition)
    # The default mode is ROUND_ROBIN.
    dm.divide_into_folds(10, settings.RANDOM)

    # Cross Validation Example
    best_param = None
    best_accuracy = 0
    possible_param_values = [1, 2, 3]

    # Run cross validation once for each parameter value
    for param in possible_param_values:

        # We will create a list of accuracies for each validation set
        accuracies = []
        for i in range(dm.get_num_folds()):
            # Update the validation fold.
            dm.set_validation(i)

            # Retrieve the training data and validation set.
            train = dm.get_all_fold_data()
            validation = dm.get_all_validation_data()

            # Train the model with the param choice.
            results = None
            # Compute the resulting accuracy on the validation set.
            accuracy = 0.3*param # fake values for demo

            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies)

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_param = param

    print(f"Best accuracy found was {best_accuracy} with parameter value {best_param}")


# Entry point to the program.
if __name__ == '__main__':
    main()

