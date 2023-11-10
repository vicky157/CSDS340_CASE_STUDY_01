import pandas as pd

features = [f'f{i}' for i in range(1, 31)] + ['class']

df = pd.read_csv('/home/vicky/Desktop/csds340/case_study/spamTrain1.csv', names=features)
count_minus_ones = df.isna().sum()
print(count_minus_ones)

df.replace(-1, inplace=True)
df.to_csv('/home/vicky/Desktop/csds340/case_study/spamTrain_pattern.csv', index=False)
count_minus_ones = df.isna().sum()
print(count_minus_ones)
