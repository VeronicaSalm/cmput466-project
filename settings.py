# Project-wide settings file.
# Has stuff like constants, file paths, etc.

import os

# Debug flag. If set, more verbose output is given.
DEBUG = True

# Various file paths and directories.
PROJECT_DIR = os.path.abspath(os.getcwd())
NEWSGROUP_DIR = os.path.join(PROJECT_DIR, 'news-data')
TWITTER_DIR = os.path.join(PROJECT_DIR, 'twitter-data')

# File names we will use for the tsv files.
NEWSGROUP_TRAIN = os.path.join(NEWSGROUP_DIR, 'newsgroup_train.tsv')
NEWSGROUP_TEST = os.path.join(NEWSGROUP_DIR, 'newsgroup_test.tsv')
NEWSGROUP_CLASSES = os.path.join(NEWSGROUP_DIR, 'newsgroup_classes.tsv')