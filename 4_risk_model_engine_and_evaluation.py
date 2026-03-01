############################################
# 4_RISK_MODEL_AND_EVALUATION
############################################

import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from tqdm import tqdm

############################################
# PARAMETERS
############################################

CONF_LEVEL = 0.95
WINDOW_DAYS = 60        # rolling estimation window
MC_SIMULATIONS = 5000   # Monte Carlo draws

############################################
# LOAD DATA
############################################

portfolio_returns = pd.read_parquet("portfolio_returns.parquet")

portfolio_returns = portfolio_returns.sort_values(['decile','date'])

portfolio_returns['trade_date'] = portfolio_returns['date'].dt.floor('D')

############################################
# HELPER FUNCTIONS
############################################

def historical_var(returns, alpha=CONF_LEVEL):
    var = np.quantile(returns, 1 - alpha)
    es = returns[returns <= var].mean()
    return var, es


def parametric_var(returns, alpha=CONF_LEVEL):
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1 - alpha)
    var = mu + z * sigma
    es = mu - sigma * norm.pdf(z) / (1 - alpha)
    return var, es


def monte_carlo_var(returns, alpha=CONF_LEVEL, simulations=MC_SIMULATIONS):
    mu = returns.mean()
    sigma = returns.std()
    simulated = np.random.normal(mu, sigma, simulations)
    var = np.quantile(simulated, 1 - alpha)
    es = simulated[simulated <= var].mean()
    return var, es


def kupiec_test(exceedances, alpha=CONF_LEVEL):
    n = len(exceedances)
    x = exceedances.sum()
    pi = x / n
    
    if x == 0:
        return 1.0
    
    LR = -2 * (
        np.log(((1 - alpha)**(n - x) * alpha**x)) -
        np.log(((1 - pi)**(n - x) * pi**x))
    )
    
    p_value = 1 - chi2.cdf(LR, df=1)
    return p_value


############################################
# ROLLING VAR ENGINE
############################################

results = []

for decile in portfolio_returns['decile'].unique():

    decile_data = portfolio_returns[
        portfolio_returns['decile'] == decile
    ].copy()

    decile_data = decile_data.sort_values('date')

    # Daily aggregation of minute returns
    daily_returns = (
        decile_data
        .groupby('trade_date')['log_return']
        .sum()
        .reset_index()
    )

    daily_returns = daily_returns.sort_values('trade_date')

    for i in tqdm(range(WINDOW_DAYS, len(daily_returns)),
                  desc=f"Processing decile {decile}"):

        window = daily_returns.iloc[i-WINDOW_DAYS:i]
        current_return = daily_returns.iloc[i]['log_return']
        current_date = daily_returns.iloc[i]['trade_date']

        hist_var, hist_es = historical_var(window['log_return'])
        para_var, para_es = parametric_var(window['log_return'])
        mc_var, mc_es = monte_carlo_var(window['log_return'])

        results.append({
            'date': current_date,
            'decile': decile,
            'model': 'historical',
            'VaR': hist_var,
            'ES': hist_es,
            'actual_return': current_return
        })

        results.append({
            'date': current_date,
            'decile': decile,
            'model': 'parametric',
            'VaR': para_var,
            'ES': para_es,
            'actual_return': current_return
        })

        results.append({
            'date': current_date,
            'decile': decile,
            'model': 'monte_carlo',
            'VaR': mc_var,
            'ES': mc_es,
            'actual_return': current_return
        })


############################################
# RESULTS DATAFRAME
############################################

results_df = pd.DataFrame(results)

results_df['exceedance'] = (
    results_df['actual_return'] < results_df['VaR']
).astype(int)

############################################
# BACKTEST SUMMARY
############################################

summary = []

for (decile, model), group in results_df.groupby(['decile','model']):

    exceed_rate = group['exceedance'].mean()
    kupiec_p = kupiec_test(group['exceedance'])

    summary.append({
        'decile': decile,
        'model': model,
        'avg_VaR': group['VaR'].mean(),
        'avg_ES': group['ES'].mean(),
        'exceedance_rate': exceed_rate,
        'kupiec_p_value': kupiec_p
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv("risk_model_summary.csv", index=False)
results_df.to_parquet("rolling_var_results.parquet", compression="snappy")

print("Risk modeling completed successfully.")
