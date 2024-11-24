import pandas as pd

def fetch_closingprice(date, daily_returns_path, closing_prices_path):
    # Convert the date input to a datetime object
    date = pd.to_datetime(date)

    # Load the daily data
    daily_data = pd.read_csv(daily_returns_path, parse_dates=['date'])

    # Filter the data for the specific date
    filtered_returns = daily_data[daily_data['date'] == date]

    # Load the closing prices data
    closing_data = pd.read_csv(closing_prices_path, parse_dates=['date'])

    # Ensure 'stock' column exists
    if 'stock' not in closing_data.columns:
        raise KeyError("'stock' column not found in the dataset.")

    # Filter the data for the specific date
    filtered_closing = closing_data[closing_data['date'] == date].set_index('stock')

    # Handle missing data
    if filtered_closing.empty:
        # Return default prices or handle accordingly
        return {k: 0 for k in filtered_returns.columns}

    # Create a dictionary with closing prices
    closing_prices_dict = filtered_closing['closing_price'].to_dict()

    # Return the closing prices with 0 for any missing data
    return {k: closing_prices_dict.get(k, 0) for k in filtered_returns.columns if k in closing_prices_dict}

if __name__ == "__main__":
    closing = fetch_closingprice(
        date="2010-01-03", 
        daily_returns_path="data/grouped_data_return_daily.csv",
        closing_prices_path="data/cleaned_dataset_with_returns.csv"
    )
    print(closing)
