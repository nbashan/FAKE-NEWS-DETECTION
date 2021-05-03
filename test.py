from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from fake import functions

methods = {"svc": LinearSVC(),
           "rf": RandomForestClassifier(),
           "mlp": MLPClassifier(),
           "lr": LogisticRegression(),
           "mnb": MultinomialNB(),
           "lnr": LinearRegression()
           }
X_train, y_train, voc = functions.get_X_y(path='data_set_1.xlsx', sheet_name='train',
                                          tfidf=TfidfVectorizer(ngram_range=(1, 5), analyzer='word',
                                                                stop_words='english'), x_feature=0, y_feature=1,
                                          label=2)
X_dev, y_dev, temp = functions.get_X_y(path='data_set_1.xlsx', sheet_name='dev',
                                       tfidf=TfidfVectorizer(vocabulary=voc, ngram_range=(1, 5), analyzer='word',
                                                             stop_words='english'), x_feature=0, y_feature=1,
                                       label=2)
for i in methods:
    print(functions.get_score(clf=methods[i], X_train=X_train, y_train=y_train, X_test=X_dev, y_test=y_dev),
          '\n**************')
