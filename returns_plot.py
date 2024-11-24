import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
import os

##################### PARTS OF THIS CODE WAS WRITTEN WITH THE HELP OF GENERATIVE AI #####################


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_cvar import optimize_cvar_portfolio, calculate_var_cvar_for_portfolio


def plot_returns(data_path, weights, start_date, end_date, var_values=None, display_days=False):

    data = pd.read_csv(data_path)
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    days = np.arange(len(data))

    for i, weight_list in enumerate(weights):
        returns = []
        for _, row in data.iterrows():
            weighted_sum = sum(row[stock] * weight for stock, weight in weight_list)
            returns.append(weighted_sum)

        return_days = {}
        for day in days:
            return_days[day] = returns[day]

        sorted_returns = sorted(return_days.items(), key=lambda x: x[1])
        l = [returns[1] for returns in sorted_returns]
        ld = [returns[0] for returns in sorted_returns]

        plt.plot(np.arange(len(ld)), l, label=f"Portfolio {'Optimal' if i == 0 else i+1}")

        if var_values is not None and i < len(var_values):
            var = var_values[i]
            var_index = next((i for i, v in enumerate(l) if v >= var), None)
            if var_index is not None:
                plt.plot(var_index-1, var, 'o', color=plt.gca().lines[-1].get_color(), 
                         label=f"VaR {var:.4f} (Portfolio {'Optimal' if i == 0 else i+1})")

                plt.vlines(x=var_index-1, ymin=-1, ymax=var, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=0.7)

                plt.hlines(y=var, xmin=0, xmax=var_index-1, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=0.7)

    if len(weights) == 1 and display_days:
        plt.xticks(ticks=np.arange(0, len(ld), 10), labels=ld[::10], rotation=90)

    plt.xlabel('Worst to best returns (days)')
    plt.xlim(left=-1)
    plt.ylim(min(returns))
    plt.ylabel('Returns')
    plt.title('Returns over Scenarios')
    plt.legend()
    plt.show()

def plot_distribution(data, stock):
    stock_data = data[stock]
    plt.figure(figsize=(10, 6))
    plt.hist(stock_data, bins=100, alpha=0.6, color='g', density=True)
    
    mu, std = stock_data.mean(), stock_data.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.title(f'Distribution of {stock} Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":

    start_date = '2000-08-09'
    end_date = '2024-02-03'
    data_path = 'data/grouped_data_return_daily.csv'
    
    portfolio_weights = [
        ('AAPL', 1)
    ]

    cvar, var, weights = optimize_cvar_portfolio(data_path, start_date, end_date, beta=0.05)
    cvar10, var10, weights10 = optimize_cvar_portfolio(data_path, start_date, end_date, beta=0.3)
    cvarAAPL, varAAPL, weightsAAPL = calculate_var_cvar_for_portfolio(data_path, start_date, end_date, portfolio_weights, beta=0.05)

    plot_returns(data_path, [weights, weights10, weightsAAPL], start_date, end_date, var_values=[var, var10, varAAPL])
