
import pandas as pd

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
data=pd.read_csv('/home/vicky/Desktop/csds340/case_study/spamTrain_pattern.csv')
X = data.drop('class', axis=1)
y= data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier()
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)


