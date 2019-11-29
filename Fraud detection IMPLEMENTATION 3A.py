import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

# Classificatori vari

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import collections

# Altro

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import _name_estimators
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv(r'C:\Users\io\Desktop\dati exp\creditcard.csv')
df.head()

df['Class'].value_counts()[0]
df['Class'].value_counts()[1]

# NO GRAFICO



# USARE ROBUST SCALER PRIMA DELLA DIVISIONE TRAIN TEST PER LE COLONNE AMOUNT E TIME

from sklearn.preprocessing import RobustScaler

rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# DIVISIONE DATASET

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# UTILIZZO SMOTE, SOLO SU TRAIN

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# APPRENDIMENTO DI INSIEME, PRIMA FUNZIONE MAJORITY VOTE, POI OGNI TIPO DI CLASSIFICATORE POI ENSEMBLE CON CROSS VALIDATION 10, E ALLA FINE GRIDSEARCH.
# MAJORITY VOTE COPIATA DA SEBASTIAN RASCHKA

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers,
                 vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for
                                  key, value in
                                  _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    def fit(self, X, y):
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,
                              self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),
                                 axis=1)
        else:  
            predictions = np.asarray([clf.predict(X)
                                      for clf in
                                      self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                           lambda x:
                           np.argmax(np.bincount(x,
                                        weights=self.weights)),
                           axis=1,
                           arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas,
                               axis=0, weights=self.weights)
        return avg_proba
    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier,
                         self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in\
                    six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(
                        step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out



clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf_labels = ['Logistic Regression', 'Decision Tree']

print('10-fold cross validation:\n')

for clf, label in zip([clf1, clf2], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train_res, y=y_train_res, cv=10, scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

mv_clf = MajorityVoteClassifier(classifiers=[clf1, clf2])
clf_labels += ['Majority Voting']
all_clf = [clf1, clf2, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train_res, y=y_train_res, cv=10, scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


#grid search for tuning params for classifier
    
from sklearn.model_selection import learning_curve, GridSearchCV    
params = {'decisiontreeclassifier__max_depth': [1, 2],'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10,  scoring='accuracy', refit = True)
grid.fit(X_train_res, y_train_res)
GridSearch_table_plot(grid, "C", negative=False)



scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T



