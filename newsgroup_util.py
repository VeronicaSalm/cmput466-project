# --------------------------------------------------
# newsgroup_util.py
#
# Utility functions for working with the newsgroup dataset.
# --------------------------------------------------

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string, nltk, csv, os

# Project-wide constants, file paths, etc.
import settings


# You might get an error with nltk.
# It can be resoloved by the following lines of code:
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def get_confusion_matrix(generated_topics, real_topics):
    '''
    Gives a confusion matrix for a set of generated topics.
    Generated based on assigning real topics using the assign_topics function.

    Arguments:
        - generated_topics (list): The generated topics for each document.
        generated_topics[i] is the generated topic of the i'th document in the corpus.
        - real_topics (list): The real topics for each document, as labelled in the dataset.
        real_topics[i] is the real topic of the i'th document in the corpus.

        The dimensions of generated_topics and real_topics should match.

    Returns:
        The confusion matrix out, represented as a dict mapping tuples to counts.
        out[(topicA, topicB)] is the number of documents predicted as topicA that are actually topicB
    '''
    assigned_topics = assign_topics(generated_topics, real_topics)

    out = dict()
    for real_topic in set(real_topics):
        for real_topic2 in set(real_topics):
            out[(real_topic, real_topic2)] = 0

    for generated_topic, real_topic in zip(generated_topics, real_topics):
        out[(assigned_topics[generated_topic], real_topic)] += 1

    return out


def assign_topics(generated_topics, real_topics):
    '''
    Assign real names in a "heuristic" manner to auto generated topics.
    For each generated topic, assign it to the real topic that is most often assigned to it, throwing out real topics that we have aalready assigned.

    Arguments:
        - generated_topics (list): The generated topics for each document.
        generated_topics[i] is the generated topic of the i'th document in the corpus.
        - real_topics (list): The real topics for each document, as labelled in the dataset.
        real_topics[i] is the real topic of the i'th document in the corpus.

        The dimensions of generated_topics and real_topics should match.

    Returns:
        A dict mapping values in generated_topics to values in real_topics.
    '''
    out = dict()
    used_real_topics = set()
    for topic in set(generated_topics):
        real_topic_count = dict()
        for real_topic in set(real_topics):
            real_topic_count[real_topic] = 0
        for generated_topic, real_topic in zip(generated_topics, real_topics):
            if generated_topic == topic:
                real_topic_count[real_topic] += 1

        # sort in descending order by value
        sorted_counts = sorted(real_topic_count.items(), key = lambda x: -x[1])

        for real_topic, count in sorted_counts:
            if real_topic not in used_real_topics:
                # if we haven't used this topic, then use it and mark it so we don't use it again
                out[topic] = real_topic
                used_real_topics.add(real_topic)
                break

    return out


def download_newsgroup():
    '''
    Download the newsgroup data to be later loaded.
    Note that we get the file paths from the settings file,
    so they don't need to be passed as arguments.
    '''

    if settings.DEBUG: print('Downloading newsgroup dataset.')

    # Download the training and test datasets.
    train = fetch_20newsgroups(data_home=settings.NEWSGROUP_DIR, subset='train', remove=('headers', 'footers'))
    test = fetch_20newsgroups(data_home=settings.NEWSGROUP_DIR, subset='test', remove=('headers', 'footers'))

    # Write the class names tsv first.
    with open(settings.NEWSGROUP_CLASSES, 'w') as f:
        fout = csv.writer(f, delimiter='\t')
        fout.writerow(train.target_names)

    # Next write the training data to file.
    # Each line is the true label and the document, with tabs removed.
    with open(settings.NEWSGROUP_TRAIN, 'w') as f:
        fout = csv.writer(f, delimiter='\t')

        for i in range(len(train.data)):
            target = train.target_names[train.target[i]]
            fout.writerow([target, train.data[i].replace('\t', ' ')])

    # Lastly, write the test data to it's file.
    with open(settings.NEWSGROUP_TEST, 'w') as f:
        fout = csv.writer(f, delimiter='\t')

        for i in range(len(test.data)):
            target = test.target_names[test.target[i]]
            fout.writerow([target, test.data[i].replace('\t', ' ')])


def load_data_newsgroup():
    '''
    Load in the newsgroup data from the tsv files.
    Note that we get the file paths from the settings file,
    so they don't need to be passed as arguments.

    Return Values:
        - train (list): List of training files with true labels.
        - test (list): List of test files with true labels.
        - classes (list): List of all class names.
    '''

    if settings.DEBUG: print('Loading in the newsgroup dataset.')

    # First make sure the files exist on the system.
    if not os.path.exists(settings.NEWSGROUP_TRAIN) \
            or not os.path.exists(settings.NEWSGROUP_TEST) \
            or not os.path.exists(settings.NEWSGROUP_CLASSES):
        raise Exception('Can not load in training or test data, files do not exist.')

    # Read in the class names first.
    with open(settings.NEWSGROUP_CLASSES, 'r') as f:
        classes = list(csv.reader(f, delimiter="\t"))[0]

    # Read in the training and test data next.
    train, test = [], []

    with open(settings.NEWSGROUP_TRAIN, 'r') as f:
        fin = csv.reader(f, delimiter="\t")

        for doc in fin:
            train.append(doc)

    with open(settings.NEWSGROUP_TEST, 'r') as f:
        fin = csv.reader(f, delimiter="\t")

        for doc in fin:
            test.append(doc)

    return (train, test, classes)


def tokenize_newsgroup(text, remove_stopwords=False):
    '''
    Tokenize a given text.

    Arguments:
        - text (string): The text to tokenize.
        - remove_stopwords (boolean): Flag for if we should remove stopwords or not.

    Return Values:
        - (list): The tokenized text.
    '''

    # First tokenize the text.
    tokens = word_tokenize(text)

    # Handle stopwords if needed.
    if remove_stopwords:
        s_words = set(stopwords.words('english'))
        tokens = list(filter(lambda x: x not in s_words, tokens))

    return tokens

def normalize_newsgroup(tokens):
    '''
    Normalize a list of tokens.

    Arguments:
        - tokens (list): The tokens to normalize.

    Return Values:
        - (list): The normalized tokens.
    '''

    # Initialize the lemmatizer.
    lemmatizer = WordNetLemmatizer()

    # Filter out tokens that are punctuation too.
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation]

