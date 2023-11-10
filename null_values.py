import pandas as pd

features = [f'f{i}' for i in range(1, 31)] + ['class']
values=0
df = pd.read_csv('/home/vicky/Desktop/csds340/case_study/spamTrain1.csv', names=features)
for i in df['f3']:
    if i==-1:
        values+=1

print(values)
# count_minus_ones = df.isna().sum()
# print(count_minus_ones)

