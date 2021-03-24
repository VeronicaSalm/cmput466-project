"""
Intruder Detection Tool
"""

import csv, os, sys, argparse, random, json
from collections import defaultdict

parser = argparse.ArgumentParser(description="Intruder Detection Tool")

parser.add_argument("input_path",
                    help="The input file containing the words from each topic.",
                    type = str)

parser.add_argument("output_path",
                    help="The path where the results should be stored",
                    type = str)


parser.add_argument("k",
                    help="The number of times to test each topic",
                    type = int)

# the number of words per testcase
TESTCASE_SIZE = 5

TOPIC = "topic"
TESTCASE ="testcase"
INTRUDER_TOPIC = "intruder_topic"
INTRUDER = "intruder"
CORRECT = "correct"
TOTAL = "total"

def generate_testcase(topics, t, n):
    """
    Generates a new testcase using the chosen topic and a randomly selected
    intruder topic.

    A testcase consists of n-1 words from topic t, with one random high-
    probability word from another topic t' (the intruder) inserted.

    Arguments:
        - topics (dict): the dictionary containing the topic data
        - t (string): the id of the chosen topic
        - n (int): n >= 2, the number of words (including the intruder) to put in
                         this testcase

    Returns:
        - testcase (list): the list of words forming the testcase
        - intruder (string): the word chosen as the intruder
        - intruder_t (string): the id of the intruder topic
    """
    # from the other topics, choose one to be the source of the intruder
    other_topics = [i for i in topics.keys() if i != t]
    intruder_t = random.choice(other_topics)

    # randomly select an intruder word from the intruder topic
    intruder = random.choice(topics[intruder_t])

    # the rest of the testcase is randomly sampled from the original topic t
    testcase = random.sample(topics[t], n-1)
    testcase.append(intruder)

    return testcase, intruder, intruder_t

def display(testcase, intruder):
    """
    Display the given testcase with nicely separated columns.

    Arguments:
        - testcase (list of strings): the list of words to display
        - intruder (string): the word that is the intruder

    Returns:
        answer (int) the number corresponding to the intruder's position
    """
    random.shuffle(testcase)
    data = [testcase, [str(i+1) for i in range(len(testcase))]]

    # From: https://stackoverflow.com/questions/9989334/create-nice-column-output-in-python
    col_width = max(len(word) for row in data for word in row) + 2  # padding
    print("*"*col_width * len(testcase))
    for row in data:
        print("".join(word.ljust(col_width) for word in row))
    print("*"*col_width * len(testcase))
    print()

    # Return the correct answer, since the testcase has been shuffled
    return testcase.index(intruder) + 1

def check_input(inp, testcase):
    """
    Checks that the input string inp is valid for the given testcase.

    Arguments:
        - inp (string): the input string to check
        - testcase (list): the testcase to check against (used for bounds checking)

    Returns:
        valid (bool): True if the input is valid, False otherwise
    """
    valid = None
    try:
        inp = int(inp)
        if inp >= 1 and inp <= len(testcase):
            valid = True
        else:
            valid = False
    except ValueError:
        if inp.startswith("q"):
            print("Quitting...")
            sys.exit()
        else:
            valid = False

    if not valid:
        print("Invalid response. Please enter the number associated with the intruder or 'q' to quit.")
        r = input("Press Enter to continue... ")
        if r.startswith("q"):
            print("Quitting...")
            sys.exit()
        print()

    return valid

def create_example_data():
    """
    Create the example data in the expected format, for testing.

    Essentially, each topic should be mapped to its highest probability words
    (e.g., the top 10 words from each topic).
    """
    with open("example_input.json", "w") as fobj:
        results = dict()
        results[0] = ["god", "mary", "jesus", "truth", "book", "christian", "bible", "christians", "religion", "faith"]
        results[1] = ["game", "team", "year", "play", "games", "win", "season", "points", "softball", "division"]
        results[2] = ["edu", "file", "com", "program", "windows", "linux", "code", "repository", "library", "send"]
        results[3] = ["space", "government", "data", "rocket", "public", "nasa", "research", "number"]

        fobj.write(json.dumps(results, indent=4))

if __name__ == "__main__":
    args = parser.parse_args()

    out_obj = open(args.output_path, "w")

    create_example_data()

    # load the topic data from the input file
    with open(args.input_path, "r") as fobj:
        topic_data = json.loads(fobj.read())

    # we will test each topic args.k times
    topics_to_test = []
    topics = list(topic_data.keys())
    for t in topics:
        topics_to_test.extend([t]*args.k)

    # Shuffle the resulting list, so the order will be unknown.
    # This method ensures that each topic will be tested an equal number of times
    # and avoids missing any topics.
    random.shuffle(topics_to_test)

    # Keep track of the number of right and wrong answers, for statistical purposes
    stats = dict()
    for t in topics:
        stats[t] = {CORRECT: 0, TOTAL: 0}

    for i in range(len(topics_to_test)):
        # generate the next testcase
        t = topics_to_test[i]
        testcase, intruder, intruder_topic = generate_testcase(topic_data, t, TESTCASE_SIZE)
        print(f"Testcase #{i+1} out of {len(topics_to_test)}:\n")
        correct = display(testcase, intruder)

        # Loop until we get a valid answer
        done = False
        while not done:
            r = input("Select the intruder: ")
            done = check_input(r, testcase)

        # Check if the intruder was found
        # TODO: store the result
        if int(r) == correct:
            print("Correct!")
            stats[t][CORRECT] += 1
        else:
            print(f"False! The intruder was '{testcase[correct-1]}' ({correct})")

        result = {
                    TOPIC : t,
                    TESTCASE : testcase,
                    INTRUDER_TOPIC: intruder_topic,
                    INTRUDER : intruder,
                    CORRECT :  int(r) == correct
                 }

        stats[t][TOTAL] += 1
        print(json.dumps(result), file=out_obj)
        print("\n")

    print("Intruder Detection Testing Complete.")
    print("\nStatistics:")
    for t in topics:
        accuracy = round(100*(stats[t][CORRECT] / stats[t][TOTAL]), 2)
        print(f"Topic: {t}, Correct: {stats[t][CORRECT]} / {stats[t][TOTAL]}, Accuracy: {accuracy} %")

    out_obj.close()
