# CMPUT 466 Project: Using Topic Modelling to Understand Public Discourse on Twitter

Project by Veronica Salm, Ian DeHaan, Noah Gergel, Siyuan Yu, Xiang Zhang

For this project, we used topic modelling to analize and understand public discourse on twitter.
Our results and findings, as well as the approach we took, is discussed in our final report and video.
This project contains all of the relevant code files we used to achieve these results, and their
contents are briefly discussed in this README.

## File Structure
Below are all of the code files in this repository.
```
* intruder-detection/
    * intruder-detection.py
        - Contains code for running intruder detection on our results.
* scripts/
    * covid-policy-tracker/
        - Contains scripts for summarizing COVID-19 lockdown information, to determine the best weeks to use.
    * extract_data/
        - Code for extracting the tweets and constructs the dataset we run the algorithms on.
* coherence.py
    - Contains the functions we used to get the coherence of our topics.
* DataManager.py
    - Contains functions and a wrapper class we used for loading in and managing our datasets.
* lda_cross_validation_twitter.py
    - Performs 10-fold cross validation on the twitter dataset for determining the best number of K topics.
* newsgroup_util.py
    - Contains utility functions for working with the 20newsgroup dataset. (Downloading, normalizing, etc.)
* twitter_util.py
    - Contains utility functions for working with the twitter dataset.
* settings.py
    - Contains project-wide settings and constants. 
* main.py
    - Code demo for the project.
* stop_list_iter.py
    - Uses LDA to build a stop list of uninformative terms.
```

Note that `intruder-detection` and the sub-directories in `scripts` all have their own READMEs, please see them for more information.

## Libraries and Running Instructions
In order to run any of our code files, multiple libraries need to be installed.
In particular, `numpy`, `sklearn`, `nltk`, and `gensim`. There is sometimes an error with `nltk` after you install it,
just uncomment the lines 19-21 in `newsgroup_util.py` and run again to fix the `nltk` error.
Once these libraries are installed, simply run `python3 main.py` to see our demo code of running both topic modelling algorithms.
