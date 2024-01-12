import os
import re
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from collections import Counter

def process_email(email: str) -> list[str]:
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    analyzer   = vectorizer.build_analyzer()
    email = re.sub(r'\d+', '', email)
    return set(analyzer(email))

def process_file(path: str, label: str):
    content = []
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
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
            #print os.path.join(subdir, file)
            for file in filenames:
                p = os.path.join(dirpath, file)
                if p.endswith(".txt"):
                    if file.startswith('spm'):
                        self.emails.append([process_file(p, 'spam'), 'spam'])
                        #self.counter_spam.update(process_file(p, 'spam')[1])
                        #self.spam_total += 1
                    else:
                        self.emails.append([process_file(p, 'clean'), 'clean'])
                        #self.counter_clean.update(process_file(p, 'clean')[0])
                        #self.clean_total += 1
                        #print (f"{filepath} -> {len(set(self.analyzer(content)))}")
                        
class Datasetz:
    def __init__(self, path: str) -> None:
        self.content = []
        self.vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
        self.analyzer   = self.vectorizer.build_analyzer()
        self.counter_spam = Counter()
        self.counter_clean = Counter()
        self.spam_total = 0
        self.clean_total = 0
        for filename in os.listdir(path):
            #print os.path.join(subdir, file)
            if filename.endswith(".txt"):
                filepath = path + os.sep + filename
                with open(filepath, 'r') as file:
                    content = file.read()
                    content = re.sub(r'\d+', '', content)
                    if filename.startswith('spm'):
                        self.counter_spam.update(set(self.analyzer(content)))
                        self.spam_total += 1
                    else:
                        self.counter_clean.update(set(self.analyzer(content)))
                        self.clean_total += 1
                    #print (f"{filepath} -> {len(set(self.analyzer(content)))}")