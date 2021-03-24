## **Intruder Detection Tool**

This tool provides a simple text-based interface for running intruder detection to evaluate the topics provided by a topic model. It follows the process described in the following video:
https://www.youtube.com/watch?v=ZQLiDh1NJK4&list=UUIXpjlxPL5Ow8rPO_gOHISQ&index=84

### Running Instructions

The script can be run from this directory using:
```
python3 intruder_detection.py input-path output-path k
```
where
* `input_path` is the path to the .json file with the input topic data. See the Input Format section below.
* `output_path` is the path to the .jsonl file where the output should be stored. See the Output Format section below.
* `k` is the number of times to test each topic. The number of testcases in a given run will always be `k*N`, where `N` is the number of topics in the provided input file.

This file is intended to be run by a tester who will respond to each testcase by selecting the word they think is the intruder. Here is an example testcase prompt:
```
Testcase #15 out of 16:                                                                               
                                                                                                      
*******************************************************                                               
jesus      bible      send       faith      christian                                                 
1          2          3          4          5                                                         
*******************************************************                                               
                                                                                                      
Select the intruder: 
```
In this case, the intruder is "send", so the tester should type `3` into the terminal prompt and press Enter. This process will repeat for all testcases, each of which is randomly generated.

### Input Format

The input file is expected to be a `json` object which maps each topic ID to the top-`n` words in each topic. The value for `n` doesn't matter, but I recommend 10 as a reasonable baseline. It is expected that the first word in the list is the highest probability word (i.e., the words are provided in decreasing probability order). Here's an example input file with four topics, `0,1,2,3`:
```
{
    "0": [
        "god",
        "mary",
        "jesus",
        "truth",
        "book",
        "christian",
        "bible",
        "christians",
        "religion",
        "faith"
    ],
    "1": [
        "game",
        "team",
        "year",
        "play",
        "games",
        "win",
        "season",
        "points",
        "softball",
        "division"
    ],
    "2": [
        "edu",
        "file",
        "com",
        "program",
        "windows",
        "linux",
        "code",
        "repository",
        "library",
        "send"
    ],
    "3": [
        "space",
        "government",
        "data",
        "rocket",
        "public",
        "nasa",
        "research",
        "number"
    ]
}
```

### Output Format

First, the statistics of the intruder detection process will be printed to standard out in the following format:
```
Statistics:                                                                                           
Topic: 0, Correct: 0 / 4, Accuracy: 0.0 %                                                             
Topic: 1, Correct: 2 / 4, Accuracy: 50.0 %                                                            
Topic: 2, Correct: 1 / 4, Accuracy: 25.0 %                                                            
Topic: 3, Correct: 1 / 4, Accuracy: 25.0 %   
```
`Correct` indicates the number of correct testcases out of the total for the given topic, along with the accuracy (correct / total * 100).    

Second, each testcase result is printed as a separate line to the output `.jsonl` file. The following information is saved:
```
{                                                                                    
     TOPIC : t,                       # the topic being tested                                                                      
     TESTCASE : testcase,             # the list of words that formed this testcase
     INTRUDER_TOPIC: intruder_topic,  # the topic the intruder was selected from
     INTRUDER : intruder,             # the intruder word
     CORRECT :  int(r) == correct     # True if the tester found the intruder successfully, or False otherwise
}
```
You can see an example output in the `example_output.jsonl` file provided in this folder.

### Future Work

The tool will probably need a "save" feature, since we may have many topics or want to run many testcases and take breaks in between. Also, it might be good to ensure that duplicate testcases are not possible (e.g., that we can never randomly generate and reuse exactly the same testcase (this is unlikely to occur, but we could add additional code to check this).
