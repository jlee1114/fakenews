import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

path = './data/news.csv'


def preprocess(path):
    '''
    returns x_train, x_test,y_train, y_test
    '''
    data = pd.read_csv(path)
    df, labels = data.iloc[:,:-1], data.iloc[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(df.text, labels, test_size= 0.2, random_state = 7)
    return x_train, x_test, y_train, y_test


def tfidf(x_train, x_test, max_df = 0.7):
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = max_df)
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)
    return tfidf_train, tfidf_test

def score():
    x_train,x_test, y_train, y_test = preprocess(path)
    tfidf_train, tfidf_test = tfidf(x_train,x_test,0.7)
    #model
    pac = PassiveAggressiveClassifier(max_iter = 50)
    pac.fit(tfidf_train, y_train)
    #scoring
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100,2) }%')
    cm = confusion_matrix(y_test,y_pred, labels = ['FAKE','REAL'])
    print('-'*20)
    print('Confusion Matrix:')
    print(cm)

if __name__ == '__main__':
    score()