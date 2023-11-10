import pandas as pd
features = []
for i in range(1, 31):
    features.append("f" + str(i))
features.append("class")
df = pd.read_csv('/home/vicky/Desktop/csds340/case_study/spamTrain.csv', names=features)
df = df.replace(-1, df.median())
df.to_csv('/home/vicky/Desktop/csds340/case_study/spamTrain_median.csv', index=False)
