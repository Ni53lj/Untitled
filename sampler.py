import pandas as pd

sample_path = "master_table_500.parquet\\ticker=RELIANCE"
sample_path2 = "master_table_500.parquet\\ticker=ACC"
sample_path3 = "master_table_500.parquet\\ticker=ACE"

df_sample = pd.read_parquet(sample_path)
df_sample2 = pd.read_parquet(sample_path2)
df_sample3 = pd.read_parquet(sample_path3)

print(df_sample.head())
print(df_sample.info())
print(df_sample2.head())
print(df_sample2.info())
print(df_sample3.head())
print(df_sample3.info())

"""
main_main = 'D:\\FINANCE PROJECTS\\Portfolio Risk Assessment using VaR & Expected Shortfall\\portfolio_returns.parquet'
df_sample_main = pd.read_parquet(main_main)
print(df_sample_main.head())
print(df_sample_main.shape)
print(df_sample_main.columns)
"""