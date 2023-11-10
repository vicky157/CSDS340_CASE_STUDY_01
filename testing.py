import pandas as pd
df=pd.read_csv("spamTrain1_mode.csv")
col=[]
for i in df["f1"]:
    col.append(i)
print(len(col))