import math
from collections import Counter

class NaiveBayes:
    def __init__(self) -> None:
        self.spam_words = Counter()
        self.clean_words = Counter()
        self.spam_total = 0
        self.clean_total = 0
        self.spam_class_prob = 0
        self.clean_class_prob = 0
        self.spam_words_prob = dict()
        self.clean_words_prob = dict()

    def train(self, emails):
        for email in emails:
            if email[1] == 'spam':
                self.spam_words.update(email[0])
                self.spam_total += 1
            else:
                self.clean_words.update(email[0])
                self.clean_total += 1

        self.spam_class_prob = math.log(self.spam_total / (self.spam_total + self.clean_total))
        self.clean_class_prob = math.log(self.clean_total / (self.spam_total + self.clean_total))

        self.calculate_probabilities()

    def calculate_probabilities(self):
        vocabulary_size = len(set(list(self.clean_words.keys()) + list(self.spam_words.keys())))

        for word, count in self.clean_words.items():
            self.clean_words_prob[word] = math.log((count + 1) / (self.clean_total + vocabulary_size))

        for word, count in self.spam_words.items():
            self.spam_words_prob[word] = math.log((count + 1) / (self.spam_total + vocabulary_size))

    def predict(self, email: list[str]):
        probs = {'spam': self.spam_class_prob, 'clean': self.clean_class_prob}
        vocabulary_size = len(set(list(self.clean_words.keys()) + list(self.spam_words.keys())))

        for word in email:
            if word in self.clean_words:
                probs['clean'] += self.clean_words_prob.get(word, math.log(1 / (self.clean_total + vocabulary_size)))
            else:
                probs['clean'] += math.log(1 / (self.clean_total + vocabulary_size))

            if word in self.spam_words:
                probs['spam'] += self.spam_words_prob.get(word, math.log(1 / (self.spam_total + vocabulary_size)))
            else:
                probs['spam'] += math.log(1 / (self.spam_total + vocabulary_size))

        return max(probs, key=probs.get)