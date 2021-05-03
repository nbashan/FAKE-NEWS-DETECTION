import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import numpy as np

def get_features(tf, data, data1):
    array_of_words = []
    row1, temp1 = data.shape
    for row in range(row1):
        sentence = data[row][0] + ' ' + data1[row][0]
        array_of_words.append(sentence)
    x = tf.fit_transform(array_of_words)
    y = TfidfTransformer().fit_transform(x.toarray())
    return y

def get_X_y(path :str, sheet_name :str, tfidf :TfidfVectorizer, x_feature :int, y_feature :int, label :int):
    df_train = np.array(pd.read_excel(open(path, 'rb'), sheet_name=sheet_name))
    return get_features(tfidf, df_train[:, [x_feature]], df_train[:, [y_feature]]), df_train[:, [label]].astype('int'), tfidf.get_feature_names()

def get_score(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, np.ravel(y_train))
    return clf.score(X_test, y_test)