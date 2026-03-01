import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############################################
# VISUALIZATION
############################################

master_df = 'D:\\FINANCE PROJECTS\\Portfolio Risk Assessment using VaR & Expected Shortfall\\master_table_500.parquet'

master_df = master_df.sort_values(['date','ticker'])

print(master_df.tail())
print(master_df['ticker'])

plt.figure(figsize=(14,7))

for ticker, data in master_df.groupby('ticker'):
    
    data = data.sort_values('date')
    
    plt.plot(
        data['date'],
        data['close'],
        label=ticker
    )

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Prices by Ticker")

plt.legend()
plt.show()

