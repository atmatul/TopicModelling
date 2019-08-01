"""
Utility File for micro training.

Functions that encapsulates the processes.
"""

import os
import string

import bibtexparser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from gensim.models import TfidfModel
from gensim.corpora import Dictionary

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from matplotlib import pyplot as plt

from wordcloud import WordCloud

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
wc = WordCloud(
    background_color="white",
    max_words=20,
    width=1024,
    height=720,
    stopwords=stopwords.words("english"),
    colormap='tab10'
)


def readfiles(path: str):
    """
    Read a folder's PDF file and extract Abstract section
    :param path: str: folder path
    :return: list: list of files in a directory
    """
    try:
        # method variables
        pdffiles = []
        for (dirpath, dirname, filenames) in os.walk(path):
            for file in filenames:
                if 'bib' in file.split('.')[-1]:
                    pdffiles.append(os.path.join(dirpath, file))
        if pdffiles:
            return pdffiles
        else:
            raise IOError("Empty Directory or Directory has not PDF files")
    except Exception as e:
        print(e)


def parsebibfiles(bibfiles: list):
    """
    Method to parsepdffiles into dict with sections
    :param bibfiles: list:
    :return: dict: [pdfsection]: value
    """
    try:
        # method variable
        lst_files = []

        # extract bib entries with abstract
        for file in bibfiles:
            with open(file, 'rb') as fh:
                bib_db = bibtexparser.load(fh)
                for entry in bib_db.entries:
                    if 'abstract' in entry.keys():
                        lst_files.append(entry)
            fh.close()
        # return interesting files
        return lst_files
    except Exception as e:
        print(e)


def clean_text(text: str):
    """
    Method to clean a text from nltk pipeline
    :param text: str
    :return: list: clean text
    """
    table = str.maketrans('', '', string.punctuation)

    # tokenize
    tokens = word_tokenize(text)

    # to lower case
    tokens = [token.lower() for token in tokens]

    # remove punctuations
    tokens = [token.translate(table) for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]

    # remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # lemm & stem
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens]

    # return variable
    return tokens


def clean_abstract(entry: dict):
    """
    Method to clean the abstract of a bibtex entry and return bow
    :param entry: dict: bibtex entry
    :return: list: clean bow
    """
    try:
        # clean abstract
        lst_clean_abstract = clean_text(entry.get('abstract'))

        # clean & append keywords
        if 'keywords' in entry.keys():
            lst_clean_keywords = clean_text(entry.get('keywords'))
            if lst_clean_keywords:
                for word in lst_clean_keywords:
                    lst_clean_abstract.append(word)

        # return cleaned text
        return lst_clean_abstract
    except Exception as e:
        print(e)

def bigrams(pds_corpus, num):
    try:
        # vectorized content
        vector = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(pds_corpus)
        bow = vector.transform(pds_corpus)
        sum_words = bow.sum(axis=0)

        # word freq
        words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]

        # descending sort
        words_freq = sorted(words_freq, key=lambda entry: entry[1], reverse=True)

        df_bigram = pd.DataFrame(words_freq, columns=['word', 'count'])

        df_bigram['count'] = (df_bigram['count'] - df_bigram['count'].min()) / (
                df_bigram['count'].max() - df_bigram['count'].min())

        df_bigram.sort_values(by=['count'])

        # write_to_excel(data=df_bigram.head(num), title='df_bigram_norm')

        # plot top num : pd.Dataframe, columns = ['words', 'count']
        df_bigram.head(num).groupby('word').sum()['count'].plot(kind='bar',
                                                                title='Top {0} Bigram Distribution'.format(num))
        plt.xlabel('Bigrams')
        plt.ylabel('Percent')
        # plt.savefig('top_{0}_bigram.png'.format(num), format='png', dpi=100)
        plt.show()

        return df_bigram

    except Exception as e:
        print(e)


def trigram_analysis(pds_corpus, num):
    """
    Method to vectorize the content of the corpus and return top #num words
    :param pds_corpus: pd.Series, corpus content
    :param num: int, number of top n results to return
    :return: None
    """
    try:
        # vectorized content
        vector = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(pds_corpus)
        bow = vector.transform(pds_corpus)
        sum_words = bow.sum(axis=0)

        # word freq
        words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]

        # descending sort
        words_freq = sorted(words_freq, key=lambda entry: entry[1], reverse=True)

        df_trigram = pd.DataFrame(words_freq, columns=['word', 'count'])

        df_trigram['count'] = (df_trigram['count'] - df_trigram['count'].min()) / (
                df_trigram['count'].max() - df_trigram['count'].min())

        df_trigram.sort_values(by=['count'])

        # write_to_excel(data=df_trigram.head(num), title='df_trigram_norm')

        # plot top num : pd.Dataframe, columns = ['words', 'count']
        df_trigram.head(num).groupby('word').sum()['count'].plot(kind='bar',
                                                                 title='Top {0} Trigram Distribution'.format(num))
        plt.xlabel('Trigrams')
        plt.ylabel('Percent')
        # plt.savefig('top_{0}_trigram.png'.format(num), format='png', dpi=100)
        plt.show()

    except Exception as e:
        print(e)


def word_cloud_analysis(df_corpus, lst_clean):
    """
    Method to provide tf-idf analysis on the corpus
    :param df_corpus: pd.Dataframe, corpus
    :return:
    """
    try:
        clean_corpus = lst_clean
        freq = {}

        # convert to corpus dict
        corpus_dict = Dictionary(clean_corpus)

        # convert to vector corpus
        vectors = [corpus_dict.doc2bow(text) for text in clean_corpus]

        # build tf-idf
        tfidf = TfidfModel(vectors)
        weights = tfidf[vectors[0]]

        # pair words with weights
        for pair in weights:
            if corpus_dict[pair[0]] not in freq.keys():
                freq[corpus_dict[pair[0]]] = pair[1]
            else:
                freq[corpus_dict[pair[0]]] = float(sum(freq.get(corpus_dict[pair[0]]), pair[1]) / 2)

        # generate word cloud
        wc.generate_from_frequencies(freq)

        # save to file
        wc.to_file("word_cloud.png")

    except Exception as e:
        print(e)


if __name__ == '__main__':
    bibfiles = readfiles(path=os.path.join(os.getcwd(), 'data'))
    lst_files = parsebibfiles(bibfiles)
    lst_clean_bow = [clean_abstract(file) for file in lst_files]

    pds_corpus = pd.Series([item.get('abstract') for item in lst_files])

    # bigrams(pds_corpus, 10)
    # trigram_analysis(pds_corpus, 10)

    word_cloud_analysis(pds_corpus, lst_clean_bow)

    print("Done")