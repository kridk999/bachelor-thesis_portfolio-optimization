import pandas as pd

# Step 1: Load the data
file_path = 'data/closing_prices.csv'
data = pd.read_csv(file_path)

# Step 2: Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Step 3: Filter out any rows where 'weekly_return', 'stock', or 'date' are missing
data = data.dropna(subset=['company_name', 'date', 'symbol'])
data = data[data['date'] >= '2000-01-03']
# Step 4: Set 'date' as the index to enable resampling by date
data.set_index('date', inplace=True)

# Step 5: Pivot the data to have one column per stock and 'weekly_return' as values
data_pivoted = data.pivot_table(index='date', columns='symbol', values='close')



# Step 7: Optional: Fill missing values if any (forward-fill/backward-fill, depending on your needs)
# Here, we can use forward-fill and fill remaining NaNs with 0 after resampling
data_weekly = data_pivoted.fillna(method='ffill').fillna(0)  # Forward fill, then fill remaining NaNs with 0

#data_weekly = data_weekly + 1

# Now you have a DataFrame where each column represents the weekly returns of a different stock
print(data_weekly.head())

# Save the processed data to a CSV file for further analysis or use in Gurobi
data_weekly.to_csv("data/grouped_closing.csv")

# This is the data you'll use in your Gurobi optimization model:
asset_returns = data_weekly.values  # Convert to a numpy array for Gurobi
