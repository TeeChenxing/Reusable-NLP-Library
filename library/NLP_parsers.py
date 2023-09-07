import json
from collections import Counter
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def json_parser(filename, stopword_list, unwanted_list):
    """
    :param filename: name of .json file
    :param stopword_list: list of custom stop-words
    :param unwanted_list: list of words user wants removed from cleaned list of words
    :return:
    """

    f = open(filename)
    raw = json.load(f)
    text = raw['text']

    # does the same thing as the default parser once json file is loaded and converted to raw text
    lines_lower = text.lower()
    all_stop_words = list(stopwords.words('english')) + stopword_list
    word_tokens = word_tokenize(lines_lower)
    filtered_sentence = [w for w in word_tokens if not w.lower() in all_stop_words]

    new_str = ' '.join(filtered_sentence)
    no_punc = new_str.translate(str.maketrans('', '', string.punctuation))
    lines_split = no_punc.split()
    unwanted = unwanted_list
    cleaned_list = [e for e in lines_split if e not in unwanted]

    f.close()

    results = {
    'wordcount': Counter(cleaned_list),
    'numwords': len(cleaned_list)
    }

    return results, cleaned_list