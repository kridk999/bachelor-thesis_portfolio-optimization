import pandas as pd

##################### PARTS OF THIS CODE WAS WRITTEN WITH THE HELP OF GENERATIVE AI #####################


file_path = 'data/closing_prices.csv'
data = pd.read_csv(file_path)


data['date'] = pd.to_datetime(data['date'], errors='coerce')

data = data.dropna(subset=['company_name', 'date', 'symbol'])
data = data[data['date'] >= '2000-01-03']
data.set_index('date', inplace=True)

data_pivoted = data.pivot_table(index='date', columns='symbol', values='close')



data_weekly = data_pivoted.fillna(method='ffill').fillna(0) 

print(data_weekly.head())

data_weekly.to_csv("data/grouped_closing.csv")

asset_returns = data_weekly.values  # Convert to a numpy array for Gurobi
