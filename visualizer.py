# --------------------------------------------------
# visualizer.py
# 
# Visualizatoin tool, for visualizing keywords for each topic
# Can be used to run on a json file in the input path
# Produces a png file output in the output path
# --------------------------------------------------

import json, argparse
from wordcloud import WordCloud
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Visualization tool")

parser.add_argument("input_path", 
                     help="The input file containing the words from each topic.",
                     type =str)

parser.add_argument("output_path",
                     help = "The path where the results should be stored",
                     type = str)

if __name__ == "__main__":
    args = parser.parse_args()
    # load the  topic data from the input file
    with open(args.input_path, "r") as fobj:
        topic_data = json.loads(fobj.read())

    # generate word cloud for each topic    
    for topic in topic_data.keys():
        keywords = " ".join(topic_data[topic]) + " "
        wordcloud = WordCloud(width = 800, height = 800, background_color = 'white',
                              min_font_size = 10, include_numbers= True, stopwords="", prefer_horizontal=True).generate(keywords)
        wordcloud.to_file("{}/topic {}.png".format(args.output_path, topic))
