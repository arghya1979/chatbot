import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
#from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
#from sklearn import preprocessing 
from sklearn.externals.joblib import parallel_backend
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline




dataset = pd.read_csv('filepath.csv', encoding="ISO-8859-1")
lst = ['How','When','Where','Which','What','I', 'about','me','please', ' ']
#unlising list of list
#lst_un = ' '.join(str(r) for x in lst for r in x )


col = ['Utterence', 'name']
dataset = dataset[col]
lst_un = ' '.join(str(x) for x in lst)
dataset['Utterence'] = dataset['Utterence'].apply(lambda x: "{}{}".format(lst_un,x))
dataset = dataset.drop_duplicates(subset=['Utterence'],keep='first')
#dataset['Utterence'] = dataset.Utterence.str.split()

#values = dataset['name'].value_counts()
def clean_text(text):
    text = text.lower()
    text_f = re.sub('[^A-Za-z]', ' ', text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text_f)
    words_stop = [words for words in word_tokens if words not in stop_words]
    word_pos = nltk.pos_tag(words_stop)
    is_noun = lambda pos: pos[:2] == 'NN'
    is_noun_p = lambda pos: pos[:2] == 'NNP'
    sent = [w for (w,i) in word_pos if is_noun(i) and is_noun_p(i)]
    sent= " ".join(sent)
    return sent

dataset['X'] = dataset['Utterence'].apply(lambda x : re.sub(r'[^A-Za-z0-9]',' ',x))
X = dataset['X']
Y = dataset['name']

vect = TfidfVectorizer(use_idf=True,smooth_idf = True, max_df = 0.25, sublinear_tf = True, ngram_range=(1,2))
X = vect.fit_transform(X).todense()
Y = vect.fit_transform(Y).todense()

X_Train,X_Test,Y_Train,y_test = train_test_split(X,Y, random_state=0, test_size=0.33, shuffle=True)

text_clf =make_pipeline([('smt', SMOTE(random_state=5)),('scale', StandardScaler(with_mean=False)),('clf', LinearSVC(class_weight='balanced'))])

parameters = {#'vect__ngram_range': [(1, 1), (1, 2)],
              #'vect__use_idf': (True, False),
              #'vect__max_df': [0.25, 0.5, 0.75, 1.0],
              #'vect__smooth_idf': (True, False),
              #'vect__sublinear_tf' : (True,False),
              'clf__C': [1.0,5.0,10.0,20.0]
              }    
#bandwidth_range = np.arange(0.7,2,0.2)   
grid = GridSearchCV(text_clf, parameters, cv=4, n_jobs=-1, scoring = 'accuracy')   
    
    
#X = preprocessing.scale(X)
with parallel_backend('threading'):
    grid.fit(X,Y)
predict = grid.predict(X_Test)
accuracy = np.mean(predict==y_test)
cm = confusion_matrix(y_test, predict,labels=np.unique(predict))
cr = classification_report(y_test, predict,labels=np.unique(predict))
accuracy_score = accuracy_score(y_test, predict)
f1_score = f1_score(y_test, predict,average='weighted')

joblib.dump(grid.best_estimator_,'filepath.pkl')
    
