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

import pyLDAvis
import pyLDAvis.gensim

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


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


def visualize_lda_model(model, corpus, corpus_dict):
    """
    Method to visualize LDA results
    :param model: lda model
    :param corpus: bow
    :param corpus_dict: id2word
    :return: Bool
    """
    vis = pyLDAvis.gensim.prepare(model, corpus, corpus_dict)
    pyLDAvis.save_html(vis, 'LDA_Visualization.html')
    return True


if __name__ == '__main__':
    bibfiles = readfiles(path=os.path.join(os.getcwd(), 'data'))
    lst_files = parsebibfiles(bibfiles)
    lst_clean_bow = [clean_abstract(file) for file in lst_files]
    print(lst_clean_bow)
