import datetime
import pandas as pd
from tqdm import tqdm
import time

class component_loader():
    def __init__(self, **kwargs):
        self.component_path = kwargs.get('component_path')
        self.raw_data = pd.read_csv(self.component_path)

    def load_symbol_data(self):  #TODO: Integrate joblib parralel
        all_data = []

        # Iterate through the date ranges
        for i in tqdm(range(len(pd.read_csv(self.component_path)['tickers']) - 1)):
            start_date = 820540800
            end_date = 1720396800
            symbols = pd.read_csv(self.component_path).iloc[i]['tickers']

            # Convert dates to UNIX timestamps
            # start_timestamp = int(pd.Timestamp(start_date).timestamp())
            # end_timestamp = int(pd.Timestamp(end_date).timestamp())

            # For each symbol, fetch stock data between the current date and the next date   
            try:
                stock_data = self._fetch_data(symbols, start_date, end_date)
                if stock_data is not None:
                    all_data.append(stock_data)
            except Exception as e:
                print(f"Error fetching data for {symbols} from {start_date} to {end_date}: {e}")
            time.sleep(1)  # Avoid rate-limiting

        # Concatenate all the data into one DataFrame
        self.final_data = pd.concat(all_data, ignore_index=True)
        self.final_data.to_csv('data/closing_prices.csv')
        return self.final_data

    def _fetch_data(self, symbol, start, end):
        # Construct the new Yahoo Finance URL
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start}&period2={end}&interval=1d&events=history"
        res = pd.read_json(url)
        timestamps = res['chart']['result'][0]['timestamp']
        closing_prices = res['chart']['result'][0]['indicators']['quote'][0]['close']
        
        dates = [datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]

        # Create a DataFrame
        df = pd.DataFrame({
        'date': dates,
        'closing_price': closing_prices,
        'symbol': res['chart']["result"][0]["meta"]["symbol"],
        'Company name' : res['chart']['result'][0]["meta"]["shortName"]
        })

        return df

if __name__ == "__main__":
    loader = component_loader(
        component_path='assets/common_stocks.csv'
    )
    final_df = loader.load_symbol_data()
    print(1)