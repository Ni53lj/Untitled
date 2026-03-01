import pandas as pd
import numpy as np

############################################
# PROTFOLIO CONSTRUCTION
############################################

master_df = pd.read_parquet("master_table_500.parquet")

master_df = master_df.sort_values(['date','ticker'])

master_df['trade_date'] = master_df['date'].dt.date

daily_turnover = (
     master_df
     .groupby(['trade_date','ticker'])
     ['turnover'].sum()
     .reset_index()
)

daily_turnover['decile'] = (
    daily_turnover
    .groupby('trade_date')
    ['turnover'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))
)

master_df = master_df.merge(
    daily_turnover[['trade_date','ticker','decile']],
    on=['trade_date','ticker'],
    how='left'
)

portfolio_returns = (
    master_df
    .groupby(['date','decile'])
    ['log_return'].mean()
    .reset_index()
)

master_df.groupby(['trade_date','decile'])['ticker'].nunique()

portfolio_returns.to_parquet("portfolio_returns.parquet", compression="snappy")

