import pandas as pd
import numpy as np

def calculate_weekly_sharpe_ratio(cvar_simulation_file="cvar_simulation.csv", cvar_results_file="cvar_results.csv"):
    cvar_simulation = pd.read_csv(cvar_simulation_file, parse_dates=['Date'])
    cvar_simulation.set_index('Date', inplace=True)
    cvar_results = pd.read_csv(cvar_results_file, parse_dates=['Date'])
    cvar_results['PortfolioValue'] = cvar_results['Date'].map(cvar_simulation['PortfolioValue'])
    cvar_results['Return'] = cvar_results['PortfolioValue'].pct_change()
    rebalancing_returns = cvar_results['Return'].dropna()
    mean_return = rebalancing_returns.mean()
    std_return = rebalancing_returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(52) if std_return != 0 else np.nan
    rebalancing_returns.to_csv('weekly_returns.csv', index=False)
    with open('sharpe_ratio.csv', 'w') as f:
        f.write(f"Annualized Sharpe Ratio,{sharpe_ratio}\n")

    return sharpe_ratio

