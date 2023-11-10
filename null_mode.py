import pandas as pd
import numpy as np

csv_file_path = '/home/vicky/Desktop/csds340/case_study/spamTrain.csv'
features = ["f" + str(i) for i in range(1, 31)] + ["class"]
df = pd.read_csv(csv_file_path, names=features)
for column in df.columns:
    if df[column].dtype == np.int64:
        mode_value = df[column].mode().values[0]
        df[column] = df[column].replace(-1, mode_value)
output_csv_path = '/home/vicky/Desktop/csds340/case_study/spamTrain_mode.csv'
df.to_csv(output_csv_path, index=False)

print(f"Replaced -1 values with mode and saved to {output_csv_path}")
