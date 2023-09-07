import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
from NLP_classes import NLP_reusable
from NLP_parsers import json_parser
import os

data = "dataset"


def main():

    """ User can create a list with their own stopwords. Keep in mind that the parser in the load_text method
    is already using stop-words from nltk.corpus. To know which stop-words is already included in the method do
    print(list(stopwords.words('english'))) """
    custom_stop_words = []

    """ After the list is cleaned, some words may be left behind that is not beneficial to the visualization so
    the user has the option to remove them. An example of this is "ve" which usually happens when the parser
    removes punctuation from the text "we've" so it becomes "we" and "ve". """
    unwanted_list = {'s', 'nt', 'll', 've'}

    # Assigning the class to a variable
    proj = NLP_reusable()

    # Loaded 4 Obama's transcripts from his past speeches from the website -- http://obamaspeeches.com/
    # Will be using the default parser which sorts through txt file.
    proj.load_text(os.path.join(data, "obama1.txt"), custom_stop_words, unwanted_list, '2009 Inaugural Address')
    proj.load_text(os.path.join(data, "obama2.txt"), custom_stop_words, unwanted_list, 'Katrina & Gulf Recovery and Iraq')
    proj.load_text(os.path.join(data, "obama3.txt"), custom_stop_words, unwanted_list, 'The American Promise Acceptance Speech')
    proj.load_text(os.path.join(data, "obama4.txt"), custom_stop_words, unwanted_list, 'Northwestern University Commencement Address')

    # Contains a dictionary of wordcount and counters for each text file.
    data_ = proj.data

    # Extract a list of labels from the data_ dictionary.
    labels = []
    for k, v in data_['wordcount'].items():
        labels.append(k)

    """ Create a sankey diagram of the 5 most common words from each file as default. User can specify how many top
    common words from each file they want or they can look at specific words from all the txt files by passing in a
    list of words they want. """
    proj.wordcount_sankey()

    """ Create a bigram plot for each file showing the top 10 most common 2-word phrases for that file. User 
    can specify the number of rows and columns for the layout. For example, if there are four files, user can 
    specify 1 row and 4 columns to create 4 subplots next to each other in a row. They can also specify 2 rows and 
    2 columns so that there are 2 plots on each row. User can also change figure size. """
    proj.bigram_plot(labels, 2, 2)

    """ Create a scatter plot for all files using sentiment analysis. User will pass in labels which will become 
    the legend, the title and also the chunk size. Chunk size for default is at 10 which means the graph is taking a
    sentiment analysis per every 10 words (stop-words are removed). User can also change figure size. """
    proj.plot_score(labels, 'Obama: Sentiment Analysis Over 4 Speeches')


if __name__ == '__main__':
    main()
