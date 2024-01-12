from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import Preprocess as Prep
import NaiveBayes
import re

def process_email(email: str) -> list[str]:
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    analyzer   = vectorizer.build_analyzer()
    email = re.sub(r'\d+', '', email)
    return set(analyzer(email))

def cvloo(emails: list) -> list[int]:
    correct = 0
    accuracies = []
    for i in range(len(emails)):
        training_data = emails[:i] + emails[i+1:]
        test_data = emails[i]
        NB = NaiveBayes.NaiveBayes()
        NB.train(training_data)
        prediction = NB.predict(test_data[0])
        
        if prediction == test_data[1]:
            correct += 1
    
        accuracy = int(correct/len(emails) * 100)
        accuracies.append(accuracy)
    return accuracies

if __name__ == "__main__":
    p = Prep.Dataset('dataset\\training')
    NB = NaiveBayes.NaiveBayes()
    NB.train(p.emails)
    print(f'Test dataset accuracy: {NB.test_accuracy('lingspam_public\\bare\\part10')}%')
    accuracies = cvloo(p.emails)

plt.plot(accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Split')
plt.ylabel('Accuracy')
plt.title('Leave One Out Cross Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
