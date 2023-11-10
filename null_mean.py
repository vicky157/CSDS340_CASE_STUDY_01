import pandas as pd

features = []
for i in range(1, 31):
    features.append("f" + str(i))
features.append("class")
df = pd.read_csv('/home/vicky/Desktop/csds340/case_study/spamTrain.csv', names=features)
df.replace(-1, df.mean(), inplace=True)
df.to_csv('/home/vicky/Desktop/csds340/case_study/spamTrain_mean.csv', index=False)
