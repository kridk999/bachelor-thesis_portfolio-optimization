import pandas as pd


def common_stocks():
    # Read the CSV file
    df = pd.read_csv('assets/S&P 500 Historical Components & Changes(08-17-2024).csv')

    # Split the tickers into a list and expand into separate rows
    df['tickers'] = df['tickers'].apply(lambda x: x.split(','))

    # Create a set of tickers from the first row
    common_stocks = set(df['tickers'].iloc[0])

    # Iterate over the remaining rows and find intersection with common_stocks
    for tickers in df['tickers'][1:]:
        common_stocks.intersection_update(tickers)

    # Convert the set of common stocks to a DataFrame
    common_stocks_df = pd.DataFrame(list(common_stocks), columns=['tickers'])

    # Save the result to a new CSV file
    common_stocks_df.to_csv('assets/common_stocks.csv', index=False)

    print("Common stocks list saved to 'common_stocks.csv'")
