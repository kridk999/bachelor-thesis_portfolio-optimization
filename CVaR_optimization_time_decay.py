import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum, Var
import os

def find_rebalancing_dates(file_path, start_date, end_date):
    try:
        data = pd.read_csv(file_path, parse_dates=['date'])
    except ValueError:
        try:
            data = pd.read_csv(file_path, parse_dates=['Date'])
            data.rename(columns={'Date': 'date'}, inplace=True)
        except ValueError:
            return []

    if 'date' not in data.columns and 'Index' in data.columns:
        data.rename(columns={'Index': 'date'}, inplace=True)

    if 'date' not in data.columns:
        return []

    data.set_index('date', inplace=True)
    rebal_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')

    adjusted_dates = []
    for date in rebal_dates:
        week = pd.date_range(start=date, end=date + pd.Timedelta(days=6))
        available_dates = [d for d in week if d in data.index]
        if available_dates:
            adjusted_dates.append(available_dates[0])
    return adjusted_dates

def optimize_portfolio(file_path, start_date, end_date, trans_cost=0.001, conf_level=0.95,
                       window_years=3, init_capital=1000000, lambda_decay=1/365, max_allocation=0.2):
    try:
        data = pd.read_csv(file_path, parse_dates=['date'])
    except ValueError:
        try:
            data = pd.read_csv(file_path, parse_dates=['Date'])
            data.rename(columns={'Date': 'date'}, inplace=True)
        except ValueError:
            return [], []

    if 'date' not in data.columns and 'Index' in data.columns:
        data.rename(columns={'Index': 'date'}, inplace=True)

    if 'date' not in data.columns:
        return [], []

    data.set_index('date', inplace=True)
    asset_list = data.columns.tolist()

    if len(asset_list) != len(set(asset_list)):
        return [], asset_list

    rebal_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    adjusted_dates = []
    for date in rebal_dates:
        week = pd.date_range(start=date, end=date + pd.Timedelta(days=6))
        available_dates = [d for d in week if d in data.index]
        if available_dates:
            adjusted_dates.append(available_dates[0])
    rebal_dates = adjusted_dates

    window_size = pd.DateOffset(years=window_years)
    portfolio_value = init_capital
    holdings = pd.Series(0.0, index=asset_list)
    allocation_history = []

    for idx, date in enumerate(rebal_dates):
        training_start_date = date - window_size
        training_end_date = date - pd.Timedelta(days=1)
        training_data = data[training_start_date:training_end_date]

        if len(training_data) < 2:
            continue

        scenarios = training_data.values
        scenario_dates = training_data.index
        days_since = np.array([(date - d).days for d in scenario_dates])
        p_t = np.exp(-lambda_decay * days_since)
        p_t /= p_t.sum()

        model = Model()
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 4)
        model.setParam('NodeLimit', 2000)
        model.setParam('MemLimit', 24000)

        weight_vars = model.addVars(asset_list, lb=0, ub=max_allocation * portfolio_value, name="weights")
        eta = model.addVar(name="eta")
        shortfalls = model.addVars(len(scenarios), lb=0, name="shortfalls")
        transaction_costs = trans_cost * quicksum(weight_vars[asset] for asset in asset_list)

        model.addConstr(quicksum(weight_vars[asset] for asset in asset_list) + transaction_costs == portfolio_value, name='portfolio_balance')

        scaling_factor = 1 / (1 - conf_level)
        model.setObjective(eta - scaling_factor * quicksum(p_t[s] * shortfalls[s] for s in range(len(scenarios))), GRB.MAXIMIZE)
        for s in range(len(scenarios)):
            portfolio_return = quicksum(scenarios[s, i] * weight_vars[asset_list[i]] for i in range(len(asset_list)))
            model.addConstr(shortfalls[s] >= eta - portfolio_return, name=f'cvar_constraint_{s}')

        try:
            model.optimize()
        except Exception:
            continue

        if model.Status == GRB.OPTIMAL:
            optimal_weights = [weight_vars[asset].X for asset in asset_list]
            total_weights = sum(optimal_weights)
            if not np.isclose(total_weights, portfolio_value, atol=1):
                optimal_weights = [w / total_weights * portfolio_value for w in optimal_weights]

            allocations = [w / portfolio_value for w in optimal_weights]
            allocation_history.append({'Date': date, 'Allocations': allocations})
            holdings = pd.Series(optimal_weights, index=asset_list)

    return allocation_history, asset_list



def simulate_portfolio(allocation_history, asset_list, data, init_capital, trans_cost):
    portfolio_value = init_capital
    holdings = pd.Series(0.0, index=asset_list)
    results = []
    prev_date = None

    for entry in allocation_history:
        date = entry['Date']
        allocations = entry['Allocations']

        if prev_date is not None:
            start_sim_date = prev_date + pd.Timedelta(days=1)
            end_sim_date = date - pd.Timedelta(days=1)
            if start_sim_date > end_sim_date:
                date_range = []
            else:
                date_range = pd.date_range(start=start_sim_date, end=end_sim_date, freq='D')

            for current_date in date_range:
                if current_date in data.index and holdings.sum() > 0:
                    returns_data = data.loc[current_date]
                    weights = holdings / portfolio_value
                    daily_return = returns_data.dot(weights)
                    portfolio_value *= (1 + daily_return)

                row = {'Date': current_date, 'PortfolioValue': portfolio_value}
                row.update(dict(zip(asset_list, holdings / portfolio_value)))
                results.append(row)

        allocations = np.array(allocations)
        if not np.isclose(allocations.sum(), 1.0):
            allocations = allocations / allocations.sum()

        desired_holdings = allocations * portfolio_value
        delta_holdings = desired_holdings - holdings
        transaction_costs = trans_cost * np.sum(np.abs(delta_holdings))
        portfolio_value -= transaction_costs
        holdings[:] = desired_holdings

        row = {'Date': date, 'PortfolioValue': portfolio_value}
        row.update(dict(zip(asset_list, allocations)))
        results.append(row)

        prev_date = date

    results_df = pd.DataFrame(results)
    return results_df
