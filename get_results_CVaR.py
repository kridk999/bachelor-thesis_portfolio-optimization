import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sharpe import calculate_weekly_sharpe_ratio
from CVaR_optimization_time_decay import optimize_portfolio, simulate_portfolio, find_rebalancing_dates

test_years = {
    2021: ["2021_3"],
    2022: ["2022_3"],
    2023: ["2023_3"]
}

parameters = [
    {"CI": 0.5, "Max Allocation": 0.25, "Time Decay": 1/45},
]

def run_testing_for_year(start_date, end_date, param_set, folder_name):
    file_path_returns = 'Bachelor_project/portfolio-optimization/grouped_data_return_daily.csv'
    initial_capital = 1000000
    transaction_cost_rate = 0.0002
    window_size_years = 3
    results_folder = f'Bachelor_project/portfolio-optimization/{folder_name}'
    os.makedirs(results_folder, exist_ok=True)

    try:
        data_returns = pd.read_csv(file_path_returns, parse_dates=['date'])
        data_returns.set_index('date', inplace=True)
        data_returns = data_returns.loc[start_date:end_date]
    except Exception as e:
        print(f"Error loading return data for {start_date} to {end_date}: {e}")
        return

    try:
        rebalancing_dates = find_rebalancing_dates(file_path_returns, start_date, end_date)
    except Exception as e:
        print(f"Error finding rebalancing dates for {start_date} to {end_date}: {e}")
        return

    try:
        cvar_allocations_history, assets = optimize_portfolio(
            file_path=file_path_returns,
            start_date=start_date,
            end_date=end_date,
            trans_cost=transaction_cost_rate,
            conf_level=param_set["CI"],
            window_years=window_size_years,
            init_capital=initial_capital,
            lambda_decay=param_set["Time Decay"],
            max_allocation=param_set["Max Allocation"]
        )
    except Exception as e:
        print(f"Error during CVaR optimization for {start_date} to {end_date}: {e}")
        return

    try:
        simulation_filename = f"simulation_CI_{int(param_set['CI']*100)}_MaxAlloc_{int(param_set['Max Allocation']*100)}_TimeDecay_{int(1/param_set['Time Decay'])}.csv"
        results_filename = f"results_CI_{int(param_set['CI']*100)}_MaxAlloc_{int(param_set['Max Allocation']*100)}_TimeDecay_{int(1/param_set['Time Decay'])}.csv"
        
        cvar_simulation = simulate_portfolio(
            allocation_history=cvar_allocations_history,
            asset_list=assets,
            data=data_returns,
            init_capital=initial_capital,
            trans_cost=transaction_cost_rate
        )
        
        cvar_simulation.to_csv(os.path.join(results_folder, simulation_filename), index=False)
        
        cvar_allocations_df = pd.DataFrame(cvar_allocations_history)
        cvar_allocations_df.to_csv(os.path.join(results_folder, results_filename), index=False)
        
        print(f"Saved simulation to {simulation_filename} and allocations to {results_filename}")
    except Exception as e:
        print(f"Error during CVaR simulation for {start_date} to {end_date}: {e}")
        return

    try:
        sharpe_ratio = calculate_weekly_sharpe_ratio(
            cvar_simulation_file=os.path.join(results_folder, simulation_filename),
            cvar_results_file=os.path.join(results_folder, results_filename)
        )

        if 'PortfolioValue' in cvar_simulation.columns:
            initial_value = cvar_simulation['PortfolioValue'].iloc[0]
            final_value = cvar_simulation['PortfolioValue'].iloc[-1]
            num_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25

            total_return = (final_value / initial_value) - 1
            average_return = (1 + total_return) ** (1 / num_years) - 1

            weekly_returns = cvar_simulation['PortfolioValue'].pct_change().dropna()
            volatility = weekly_returns.std() * np.sqrt(52)
        else:
            raise ValueError("PortfolioValue column is missing in cvar_simulation data")

        metrics = {
            "Start Date": start_date,
            "End Date": end_date,
            "CI": param_set["CI"],
            "Max Allocation": param_set["Max Allocation"],
            "Time Decay": 1/param_set["Time Decay"],
            "Sharpe Ratio": sharpe_ratio,
            "Average Annual Return": average_return,
            "Volatility": volatility
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(results_folder, "performance_metrics.csv"), index=False)
        print(f"Saved performance metrics for {start_date} to {end_date} in {folder_name}")

    except Exception as e:
        print(f"Error calculating performance metrics for {start_date} to {end_date}: {e}")

if __name__ == "__main__":
    for year, folders in test_years.items():
        for param_set, folder_name in zip(parameters, folders):
            print(f"Testing year {year} with parameters: {param_set}")
            run_testing_for_year(f'{year}-01-01', f'{year}-12-31', param_set, folder_name)
    
    cumulative_folders = [ "2021-2023_3"]
    for param_set, folder_name in zip(parameters, cumulative_folders):
        print(f"Testing from 2021 to 2023 with parameters: {param_set}")
        run_testing_for_year('2021-01-01', '2023-12-31', param_set, folder_name)
