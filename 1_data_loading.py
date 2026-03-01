import pandas as pd
import glob
""""
file_loc = "D:\\FINANCE PROJECTS\\DATASETS\\archive\\3MINDIA_minute.csv"

df = pd.read_csv(file_loc)
df['date'] = pd.to_datetime(df['date'])

print(df.isna().any())
"""
data_path = r"D:\\FINANCE PROJECTS\\DATASETS\\archive\\*.csv"
files = glob.glob(data_path)
print(len(files))

summary = []

for f in files:
    df = pd.read_csv(
        f,
        usecols=['close','volume']
    )
    avg_turnover = (df['close'] * df['volume']).mean()
    summary.append((f, avg_turnover))

summary_df = pd.DataFrame(
    summary,
    columns=['file','volume']
)
summary_df.to_csv("nifty500_peak_volume_rank.csv", index=False)

