import pandas as pd

##################### PARTS OF THIS CODE WAS WRITTEN WITH THE HELP OF GENERATIVE AI #####################


def common_stocks():
    df = pd.read_csv('assets/S&P 500 Historical Components & Changes(08-17-2024).csv')

    df['tickers'] = df['tickers'].apply(lambda x: x.split(','))

    common_stocks = set(df['tickers'].iloc[0])

    for tickers in df['tickers'][1:]:
        common_stocks.intersection_update(tickers)

    common_stocks_df = pd.DataFrame(list(common_stocks), columns=['tickers'])

    common_stocks_df.to_csv('assets/common_stocks.csv', index=False)

    print("Common stocks list saved to 'common_stocks.csv'")
