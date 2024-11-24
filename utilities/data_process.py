import pandas as pd


file_path = 'data/dataset_with_weekly_returns.csv'
data = pd.read_csv(file_path)

data['date'] = pd.to_datetime(data['date'], errors='coerce')

data = data.dropna(subset=['weekly_return', 'date', 'stock'])

data.set_index('date', inplace=True)

data_pivoted = data.pivot_table(index='date', columns='stock', values='weekly_return')

data_weekly = data_pivoted.resample('W').last()

data_weekly = data_weekly.fillna(method='ffill').fillna(0) 

#data_weekly = data_weekly + 1

print(data_weekly.head())

data_weekly.to_csv("data/grouped_data_return_weekly.csv")

asset_returns = data_weekly.values  