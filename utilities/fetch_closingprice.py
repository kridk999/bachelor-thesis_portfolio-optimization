import pandas as pd


##################### PARTS OF THIS CODE WAS WRITTEN WITH THE HELP OF GENERATIVE AI #####################


def fetch_closingprice(date, daily_returns_path, closing_prices_path):
    date = pd.to_datetime(date)

    daily_data = pd.read_csv(daily_returns_path, parse_dates=['date'])

    filtered_returns = daily_data[daily_data['date'] == date]

    closing_data = pd.read_csv(closing_prices_path, parse_dates=['date'])

    if 'stock' not in closing_data.columns:
        raise KeyError("'stock' column not found in the dataset.")

    filtered_closing = closing_data[closing_data['date'] == date].set_index('stock')

    if filtered_closing.empty:
        return {k: 0 for k in filtered_returns.columns}

    closing_prices_dict = filtered_closing['closing_price'].to_dict()

    return {k: closing_prices_dict.get(k, 0) for k in filtered_returns.columns if k in closing_prices_dict}

if __name__ == "__main__":
    closing = fetch_closingprice(
        date="2010-01-03", 
        daily_returns_path="data/grouped_data_return_daily.csv",
        closing_prices_path="data/cleaned_dataset_with_returns.csv"
    )
    print(closing)
