import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

file_loc = "D:\\FINANCE PROJECTS\\Portfolio Risk Assessment using VaR & Expected Shortfall\\nifty500_peak_volume_rank.csv"

dataset = pd.read_csv(file_loc)

dataset_main = dataset.sort_values("volume",ascending=False)

dataset_main = dataset_main.iloc[:,:].copy()

top500 = dataset_main['file'].tolist()

files = []

############################################
# CLEANING FUNCTION
############################################

def cleaning_function(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    df = df.sort_values('date')
    df = df.drop_duplicates('date')

    df = df[df['close'] > 0]

    df['close'] = df['close'].ffill()
    df['volume'] = df['volume'].fillna(0)

    start = pd.Timestamp("09:15").time()
    end = pd.Timestamp("16:00").time()

    df = df[(df['date'].dt.time >= start) & 
        (df['date'].dt.time <= end)]

    return df


############################################
# FEATURE ENGINEERING
############################################

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Faster + numerically stable
    df['log_return'] = np.log(df['close']).diff()

    df = df.dropna(subset=['log_return'])

    # Adaptive outlier filter like median absolute deviation
    median = df['log_return'].median()
    mad = np.median(np.abs(df['log_return']-median))
    df = df[np.abs(df['log_return'] - median) < 10 * mad]

    df['turnover'] = df['close'] * df['volume']

    return df


############################################
# PIPELINE
############################################

master_df = []

output_file = "master_table_500.parquet"

for path in tqdm(top500, desc="Building minute dataset"):

    try:

        df = pd.read_csv(
            path,
            usecols=['date','close','volume'],
            dtype={
                'close':'float32',
                'volume':'float32'
            },
            parse_dates=['date']
        )

        df = cleaning_function(df)
        df = engineer_features(df)

        # Extract ticker safely
        ticker = os.path.basename(path)
        df['ticker'] = re.sub(r'_minute\.csv$','',ticker)

        df = df[['date','ticker','close','volume','log_return','turnover']]

        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            partition_cols=['ticker'],
            index=False
        )

        del df

    except MemoryError:
        print(f"Out of RAM while processing: {path}")

    except Exception as e:
        print(f"Failed: {path}")
        print(f"Reason: {e}\n")