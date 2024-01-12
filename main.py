import Preprocess as Prep
import NaiveBayes
import re
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from collections import Counter
import os

def process_email(email: str) -> list[str]:
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    analyzer   = vectorizer.build_analyzer()
    email = re.sub(r'\d+', '', email)
    return set(analyzer(email))

if __name__ == "__main__":
    p = Prep.Dataset('dataset\\training')
    NB = NaiveBayes.NaiveBayes()
    NB.train(p.counter_clean, p.clean_total, p.counter_spam, p.spam_total)
    #NB.predict(process_email('Subject: books on functional linguistics john benjamins publishing would like to call your attention to the following new title in the field of functional linguistics : grammatical relations a functionalist perspective t . givon ( eds . ) 1997 viii , 350 pp . typological studies in language , 35 us / canada : cloth : 1 55619 645 8 price : us $ 86 . 00 paper : 1 55619 646 6 price : us $ 29 . 95 rest of the world : cloth : 90 272 2931 7 price : hfl . 165 , - - paper : 90 272 2932 5 price : hfl . 60 , - - john benjamins publishing web site : http : / / www . benjamins . com for further information via e-mail : service @ benjamins . com this volume presents a functional perspective on grammatical relations ( grs ) without neglecting their structural correlates . ever since the 1970s , the discussion of grs by functionally-oriented linguists has focused primarily on their functional aspects , such as reference , cognitive accessibility and discourse topicality . with some exceptions , functionalists have thus ceded the discussion of the structural correlates of grs to various formal schools . ever since edward keenan \'s pioneering work on subject properties ( 1975 , 1976 ) , it has been apparent that subjecthood and objecthood can only be described properly by a basket of neither necessary nor sufficient properties - thus within a framework akin to rosch \'s theory of prototype . some gr properties ar functional ( reference , topicality , accessibility ) ; others involve overt coding ( word-order , case marking , verb agreement ) . others yet are more abstract , involving control of grammatical processes ( rule-governed behavior ) . building on keenan \'s pioneering work , this volume concentrates on the structural aspects of grs within a functionalist framework . following a theoretical introduction , the papers in the volume deal primarily with recalcitrant typological issues : the dissociation between overt coding properties of grs and their behavior-and - control properties ; grs in serial verb constructions ; grs in ergative languages ; the impact of clause union and grammaticalization on grs . - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - bernadette martinez - keck tel : ( 215 ) 836-1200 publicity / marketing fax : ( 215 ) 836-1204 john benjamins north america e-mail : bernie @ benjamins . com po box 27519 philadelphia pa 19118-0519 check out the john benjamins web site : http : / / www . benjamins . com'))
    test_folder = 'lingspam_public\\bare\\part10'
    total = 0
    correct = 0
    for filename in os.listdir(test_folder):
        path = test_folder + os.sep + filename
        total += 1
        with open(path, 'r') as file:
            content = file.read()
            label = 'spam' if filename.startswith('spm') else 'clean'

            prediction = NB.predict(process_email(content))

            if(prediction == label):
                correct += 1


    print(f'Accuracy: {int((correct/total)*100)}%')
