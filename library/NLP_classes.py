from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sankey as sk
import pandas as pd
from nltk import ngrams
from textblob import TextBlob


class NLP_reusable:

    def __init__(self):
        """ Constructor """
        # extracted data (state)
        self.data = defaultdict(dict)
        # extracted words from text
        self.wordlist = []
        # extracted label names from text
        self.label = []

    def _save_results(self, label, results, wl):
        """
        :param label: title of file
        :param results: cleaned dictionary of word count and number of words in the file
        :param wl: a list of cleaned words within the file
        :return: stores a 2D dictionary in self.data and a list of words in self.wordlist
        """
        for k, v in results.items():
            self.data[k][label] = v

        self.wordlist.append(wl)

    @staticmethod
    def _default_parser(filename, stopword_list=None, unwanted_list=None):
        """
        :param filename: name of .txt file
        :param stopword_list: list of custom stop-words
        :param unwanted_list: list of words user wants removed from cleaned list of words
        :return: 2-D dictionary of word count & num words and a cleaned list of words from the txt file
        """

        with open(filename) as f:
            lines = f.read()
            lines_lower = lines.lower()

            # create a list of stop-words
            all_stop_words = list(stopwords.words('english')) + stopword_list
            # divide text file into a substring. Ex. "Hello World!" --> ["Hello", "World", "!"]
            word_tokens = word_tokenize(lines_lower)
            # create a list with stop-words removed
            filtered_txt = [w for w in word_tokens if not w.lower() in all_stop_words]

            # combine list of strings into 1 blob of text
            new_str = ' '.join(filtered_txt)
            # remove punctuation
            no_punc = new_str.translate(str.maketrans('', '', string.punctuation))
            lines_split = no_punc.split()
            unwanted = unwanted_list
            # remove substrings in list if it is in the unwanted list
            cleaned_list = [e for e in lines_split if e not in unwanted]

        results = {
            'wordcount': Counter(cleaned_list),
            'numwords': len(cleaned_list)
        }
        return results, cleaned_list

    def load_text(self, filename, stopword_list, unwanted_list, label=None, parser=None):
        """ Registers the text file with the NLP framework """

        if parser == None:
            # if there is no parser, use default parser which is for .txt files
            results, wl = NLP_reusable._default_parser(filename, stopword_list, unwanted_list)
        else:
            results, wl = parser(filename, stopword_list, unwanted_list)

        if label is None:
            label = filename

        self._save_results(label, results, wl)

    def wordcount_sankey(self, word_list=None, k=5, threshold=0):
        """
        :param word_list: a list of words from the file
        :param k: union of k most common words for each file
        :return: a sankey diagram with file name as the source and words as the target
        """
        df_data = []

        # append word and how frequent it occurs inside the text into a list
        for key, value in self.data['wordcount'].items():
            for key2, value2 in value.items():
                df_data.append([key, key2, value2])

        # turn list into a data frame
        nlp_df = pd.DataFrame(df_data, columns=['Text_Name', 'Word', 'Count'])
        # only include words that have count greater than threshold
        nlp_df2 = nlp_df[nlp_df['Count'] > threshold]
        # convert count column into float
        nlp_df2['Count'] = nlp_df2['Count'].astype(float)

        # if word_list is provided, make sankey diagram words in the word list as the target
        if word_list != None:
            new_df = pd.DataFrame()
            for x in range(len(word_list)):
                nlp_dfwl = nlp_df2[nlp_df2['Word'] == word_list[x]]
                new_df = new_df.append(nlp_dfwl)

            # display sankey diagram
            sk.make_sankey(new_df, 'Text_Name', 'Word', 'Count', pad=50, thickness=20, line_width=2)

        # if word_list is not provided, make sankey diagram using union of k most common words for each file as
        # the target
        else:
            new_df = pd.DataFrame()

            for key in self.data['wordcount']:
                # make new data frame for each text file
                separate_df = nlp_df2[nlp_df2['Text_Name'] == key]
                # find top k most frequent words in that text file
                top_df = separate_df.nlargest(k, 'Count', keep='all')
                # append top k most frequent words of each file into new_df
                new_df = new_df.append(top_df)

            # display sankey diagram
            sk.make_sankey(new_df, 'Text_Name', 'Word', 'Count', pad=50, thickness=20, line_width=2)

    def bigram_plot(self, title_list, row_num, col_num, fig_x=10, fig_y=17):
        """
        :param title_list: list of each file name
        :param row_num: number of rows for subplot layout
        :param col_num: number of columns for subplot layout
        :param fig_x: length of x-axis of figure containing all subplots
        :param fig_y: length of y-axis of figure containing all subplots
        :return: one visualization showing all subplots (one for each file)
        """

        # adjust figure
        plt.figure(figsize=(fig_x, fig_y))
        plt.tight_layout()

        # create i number of subplots depending on the number of files
        for i in range(len(self.wordlist)):

            book = self.wordlist[i]
            book_cp = (ngrams(book, 2))
            # 10 most common phrases for book 1
            mcp = Counter(book_cp).most_common(10)

            str_ls = []
            int_ls = []

            # create a list of 2-word phrases
            for i2 in mcp:
                no_int = [y for y in i2 if not isinstance(y, int)]
                no_int2 = no_int[0]
                no_int3 = ', '.join(no_int2)
                str_ls.append(no_int3)

            # create a list of frequency of the 2-word phrases
            for i3 in mcp:
                for x in i3:
                    if type(x) == int:
                        int_ls.append(x)

            # create a dictionary from the two lists
            bigram_dict = {str_ls[i]: int_ls[i] for i in range(len(str_ls))}
            # re label the keys and frequency
            phrases = list(bigram_dict.keys())
            frequency = list(bigram_dict.values())

            df_dict = {'Phrases': phrases, 'Frequency': frequency}
            # create a dataframe
            df = pd.DataFrame(df_dict)

            # create subplots with modifications to make it look more visually appealing
            ax = plt.subplot(row_num, col_num, i+1)
            sns.barplot(data=df, x='Phrases', y='Frequency', color='lightblue')
            plt.xticks(fontsize=8, rotation=45)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.title.set_text(title_list[i])
            ax.yaxis.grid(True, color='#EEEEEE')
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            ax.grid(color='gray', linestyle='dashed')
            plt.subplots_adjust(hspace=0.4)

        plt.show()

    def score_text(self, file_num, chunk_size=10, min_pol=-1.0, max_pol=1.0, min_sub=0.0, max_sub=1.0):
        """
        :param text_num: position of the file in a list
        :param chunk_size: size of the chunk the method will be doing sentiment analysis on
        :param min_pol: minimum polarity value
        :param max_pol: maximum polarity value
        :param min_sub: minimum subjectivity value
        :param max_sub: maximum subjectivity value
        :return: a list of polarity and subjectivity value each in a tuple. Ex. [(sub1, pol1), (sub2, pol2), ...]
        """
        text1 = self.wordlist[file_num]
        # create a 2-D list that contains the separated chunks
        comp_list = [text1[x:x + chunk_size] for x in range(0, len(text1), chunk_size)]
        score_list = []

        for i in range(len(comp_list)):

            # join each chunk
            text_chunk = ' '.join(comp_list[i])
            # find subjectivity and polarity values for each chunk
            sub_analysis = TextBlob(text_chunk).subjectivity
            pol_analysis = TextBlob(text_chunk).polarity

            # make sure the polarity and subjectivity values don't go above a certain amount
            if min_pol <= pol_analysis <= max_pol and min_sub <= sub_analysis <= max_sub:
                score = (sub_analysis, pol_analysis)
                score_list.append(score)

        # remove polarity and subjectivity tuples that are both 0
        cleaned_score = [score for score in score_list if score[0] != 0 and score[1] != 0]

        return cleaned_score

    def plot_score(self, labels, title, chunksize=10, fig_x=12, fig_y=10):
        """
        :param labels: legend labels
        :param title: title of scatter plot
        :param chunksize: customize chunksize for sentiment analysis
        :param fig_x: length of x-axis of figure containing all subplots
        :param fig_y: length of y-axis of figure containing all subplots
        :return: a scatter plot showing sentiment analysis
        """

        plt.figure(figsize=(fig_x, fig_y))
        plt.style.use('seaborn-poster')
        # plot points for each file on 1 graph
        for i in range(len(self.wordlist)):
            x_axis, y_axis = list(map(list, zip(*self.score_text(i, chunksize))))
            plt.scatter(x_axis, y_axis, edgecolors='black', label=labels[i])
        plt.legend()
        plt.xlabel('Subjectivity')
        plt.title(title)
        plt.grid(color='gray', linestyle='dashed')
        plt.ylabel('Polarity')

        plt.show()
