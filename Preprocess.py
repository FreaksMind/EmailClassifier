from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from collections import Counter
import os
import re

def process_email(email: str) -> list[str]:
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    analyzer   = vectorizer.build_analyzer()
    email = re.sub(r'\d+', '', email)
    return set(analyzer(email))

def process_file(path: str) -> set:
    content = []
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, )
    analyzer   = vectorizer.build_analyzer()
    counter_spam = Counter()
    counter_clean = Counter()
    with open(path, 'r') as file:
        content = file.read()
        content = re.sub(r'\d+', '', content)

    return set(analyzer(content))

class Dataset:
    def __init__(self, path: str) -> None:
        self.content = []
        self.vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
        self.analyzer   = self.vectorizer.build_analyzer()
        self.counter_spam = Counter()
        self.counter_clean = Counter()
        self.spam_total = 0
        self.clean_total = 0
        self.emails = list()
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                p = os.path.join(dirpath, file)
                if p.endswith(".txt"):
                    if file.startswith('spm'):
                        self.emails.append([process_file(p), 'spam'])
                    else:
                        self.emails.append([process_file(p), 'clean'])
