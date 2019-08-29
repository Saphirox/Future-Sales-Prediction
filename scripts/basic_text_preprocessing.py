import itertools
import numpy as np
import pandas as pd

import nltk

nltk.download('stopwords')

from collections import Counter
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import re

from functools import partial

"""
    Class BasicPreprocessText
    
    Basic preprocessing tool for text data:
    1. Remove punctuation
    2. Remove noisy symbols

"""


class BasicPreprocessText:
    def __init__(self, lang='english'):
        self.LANG = lang
        self.stemmer = SnowballStemmer(self.LANG)
        self.stopWords = set(stopwords.words(self.LANG))

    def parse_words(self, column_data):
        str_data = column_data.astype('str')

        str_data = str_data[str_data != 'nan']
        split_words = str_data.apply(lambda word: word.split(' '))

        strings_data = list(itertools.chain.from_iterable(split_words))
        return strings_data

    def vectorize_process_text(self, text_arr, leave_stop_words=None):
        if leave_stop_words is None:
            leave_stop_words = []

        assert isinstance(leave_stop_words, list)

        modified_process_text = partial(self.process_text, leave_stop_words=leave_stop_words)
        vectorize_process_text = np.vectorize(modified_process_text)
        return vectorize_process_text(text_arr)

    def clean_from_stop_words_and_punctuation(self, x):
        sw = stopwords.words(self.LANG)
        return [word.lower() for word in x.split() if word.lower() not in sw]

    def process_text(self, text, leave_stop_words):
        # remove html
        text_without_html = BeautifulSoup(text, 'lxml').text
        text_lowercase = text_without_html.lower()
        tokens = self.clean_from_stop_words_and_punctuation(text_lowercase)

        punc = list(string.punctuation)
        for i in leave_stop_words:
            punc.remove(i)
        punc = "".join(punc)

        punctuation_removed = [re.sub("([" + punc + "]|\s+)+", '', token) for token in tokens]
        stemmed_words = [self.stemmer.stem(plural) for plural in punctuation_removed]

        return " ".join(stemmed_words)
