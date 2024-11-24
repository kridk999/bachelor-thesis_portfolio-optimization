import pandas as pd
import numpy as np
import torch
from LSTM.model import *
from visual import *
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42) 

def main(model_type=2021_2, year=2021, model_path=None, base_path="LSTM_results"):
    # file path er grouped_data_return_daily.csv
    def find_rebalancing_dates(file_path, start_date, end_date):
        data = pd.read_csv(file_path, parse_dates=['date'])
        data.set_index('date', inplace=True)
        rebal_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')

        adjusted_dates = []
        for date in rebal_dates:
            week = pd.date_range(start=date, end=date + pd.Timedelta(days=6))
            available_dates = [d for d in week if d in data.index]
            if available_dates:
                adjusted_dates.append(available_dates[0])
        return adjusted_dates


    #dates = find_rebalancing_dates('data/grouped_data_return_daily.csv', '2021-01-01', '2021-12-31')

    def load_model(model_path):
        checkpoint = torch.load(model_path, weights_only=True)
        config = checkpoint["config"]
        

        model = LSTMAllocationModelWithAttention(input_size=config["input_size"],
                                    hidden_sizes=config["hidden_sizes"],
                                    dropout_rate=config["dropout_rate"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    
    if model_path == "random":
        model="random"
    if model_path == "baseline":
        model="baseline"
    if model_path != "random" and model_path != "baseline":
        model = load_model(model_path)
        
    def generate_allocation_history(model, rebalancing_dates, data, asset_list, sequence_length=30, scale_data=True):
        allocation_history = []

        

        for date in rebalancing_dates:
            if date not in data.index:
                continue

            end_idx = data.index.get_loc(date)
            start_idx = max(0, end_idx - sequence_length + 1)  
            
            input_data = data.iloc[start_idx:end_idx + 1].values  
            if input_data.shape[0] < sequence_length:

                continue


            input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            if scale_data:
                scaler = MinMaxScaler()
                input_data = torch.tensor(scaler.fit_transform(input_data.squeeze(0)), dtype=torch.float32)

            with torch.no_grad():
                if model_path != "random" and model_path != "baseline":
                    allocations = model(input_data.unsqueeze(0)).squeeze().numpy() 
                    l = []
                    g = []
                    for i in range(10):
                        l.append(input_data[:, i].mean())
                        g.append(input_data[:, i].std())
                    print(g, l, date, allocations)
                    print(1)
                if model == "random":
                    vector = np.random.rand(data.columns.size)  # Generate 148 random values between 0 and 1
                    allocations = vector / vector.sum()
                if model == "baseline":
                    allocations = np.full(data.columns.size, 1/data.columns.size)


                #print(allocations, np.max(allocations), np.min(allocations), np.sum(allocations))

            allocation_history.append({"Date": date, "Allocations": allocations.tolist()})  

        allocation_df = pd.DataFrame(allocation_history)
        allocation_df.to_csv(f'{base_path}/{model_type}/allocation_history.csv', index=False)

        return allocation_df


    rebalancing_dates = find_rebalancing_dates('data/grouped_data_return_daily.csv', f'{year}-01-01', f'{year}-12-31')
    data = pd.read_csv('data/grouped_data_return_daily.csv', usecols=range(149), parse_dates=['date']).set_index('date')  #HERE
    asset_list = data.columns.to_list()

    allocation_history_df = generate_allocation_history(model, rebalancing_dates, data, asset_list)
    #print(allocation_history_df)

    #Portfolio simulation
    # Allocation history er LSTM output pr date
    # Asset list er de assets vi har i common_stocks - Copy.csv
    # Data er de returns vi har i grouped_data_return_daily.csv
    # Init capital = 1000000
    # trans cost = 0.0002
    def simulate_portfolio(allocation_history, asset_list, data, init_capital, trans_cost):
        portfolio_value = init_capital
        holdings = pd.Series(0.0, index=asset_list)
        results = []
        prev_date = None
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        allocation_history['Date'] = pd.to_datetime(allocation_history['Date'])

        for _, entry in allocation_history.iterrows():
            date = entry['Date']
            allocations = entry['Allocations'] 

            if prev_date is not None:
                start_sim_date = prev_date + pd.Timedelta(days=1)
                end_sim_date = date - pd.Timedelta(days=1)
                if start_sim_date > end_sim_date:
                    date_range = []
                else:
                    date_range = pd.date_range(start=start_sim_date, end=end_sim_date, freq='D')
                    print(f"start_sim_date: {start_sim_date}, end_sim_date: {end_sim_date}, date_range: {date_range}")


                for current_date in date_range:
                    if current_date in data.index and holdings.sum() > 0:
                        #print(f"Processing current_date: {current_date}")
                        returns_data = data.loc[current_date]
                        #print(f"Returns data for {current_date}: {returns_data}")
                        weights = holdings / portfolio_value
                        daily_return = returns_data.dot(weights)
                        portfolio_value *= (1 + daily_return)
                        #print(f"Updated Portfolio Value: {portfolio_value}")
                    else:
                        print(f"Skipping current_date: {current_date} (not in data.index or zero holdings)")


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
        results_df.to_csv(f'{base_path}/{model_type}/LSTM_simulation1.csv', index=False)
        return results_df



    #SHARPE RATIO beregnet ud fra csv output fra simulate_portfolio og allocation history csv fra LSTM som du også bruger i simulate_portfolio
    def calculate_weekly_sharpe_ratio(cvar_simulation_file=f"{base_path}/{model_type}/LSTM_simulation1.csv", cvar_results_file=f"{base_path}/{model_type}/allocation_history.csv"):
        cvar_simulation = pd.read_csv(cvar_simulation_file, parse_dates=['Date'])
        cvar_simulation.set_index('Date', inplace=True)
        cvar_results = pd.read_csv(cvar_results_file, parse_dates=['Date'])

        cvar_results['PortfolioValue'] = cvar_results['Date'].map(cvar_simulation['PortfolioValue'])
        cvar_results['Return'] = cvar_results['PortfolioValue'].pct_change()
        weekly_returns = cvar_results['Return'].dropna()
        mean_return = weekly_returns.mean()
        std_return = weekly_returns.std()
        num_rebalancing_periods = len(weekly_returns)
        
        sharpe_ratio = (mean_return / std_return) * np.sqrt(52) if std_return != 0 else np.nan
        weekly_returns.to_csv('weekly_returns.csv', index=False)
        with open('sharpe_ratio.csv', 'w') as f:
            f.write(f"Annualized Sharpe Ratio,{sharpe_ratio}\n")

        return sharpe_ratio
    

    data = pd.read_csv("data/grouped_data_return_daily.csv", usecols=range(149))  #HERE
    asset_list = data.columns.to_list()[1:]


    allocation_history = pd.read_csv(f"{base_path}/{model_type}/allocation_history.csv")
    allocation_history['Allocations'] = allocation_history['Allocations'].apply(ast.literal_eval) 

    simulation_results = simulate_portfolio(allocation_history, asset_list, data, init_capital=1000000, trans_cost=0.0002)
    sharpe_ratio = calculate_weekly_sharpe_ratio()
    print(f"Annualized Sharpe Ratio: {sharpe_ratio}")

if __name__ == "__main__":
    import ast

    #allocation_history = generate_allocation_history(model, rebalancing_dates, data, asset_list)

    model1 = "models1/5TEST10best_model_layers_1_seq_30_hunits[148]_lr0.001_daydecay1_maxnorminf.pth"
    base_path = "LSTM_tests"

    main(model_type="2021_baseline", year="2021", model_path="baseline", base_path = base_path)
    main(model_type="2021_random", year="2021", model_path="random", base_path = base_path)
    main(model_type="2021_1", year="2021", model_path=model1, base_path = base_path)
    #main(model_type="2021_2", year="2021", model_path=model2, base_path = base_path)

    main(model_type="2022_baseline", year="2022", model_path="baseline", base_path = base_path)
    main(model_type="2022_random", year="2022", model_path="random", base_path = base_path)
    main(model_type="2022_1", year="2022", model_path=model1, base_path = base_path)
    #main(model_type="2022_2", year="2022", model_path=model2, base_path = base_path)

    main(model_type="2023_baseline", year="2023", model_path="baseline", base_path = base_path)
    main(model_type="2023_random", year="2023", model_path="random", base_path =base_path)
    main(model_type="2023_1", year="2023", model_path=model1, base_path = base_path)
    #main(model_type="2023_2", year="2023", model_path=model2, base_path = base_path)
        
            
            
    visual_main(base_path=base_path)

