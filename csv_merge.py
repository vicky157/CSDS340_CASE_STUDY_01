import pandas as pd

data1 = pd.read_csv('spamTrain1.csv')
data2 = pd.read_csv('spamTrain2.csv')
output1 = pd.merge(data1, data2,on='f1')
output1.to_csv('merged_spam_train.csv', index=False)
