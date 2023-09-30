import os
import pandas as pd
import pendulum as time


class EuronextDataLoader:
    data_path = './data'
    test_runs_path = './test_runs'

    def __init__(self, tickers) -> None:
        self.__tickers = tickers
        self.__dataframes = {}

    def load_data(self) -> None:
        for ticker in self.__tickers:
            dataframe = pd.read_csv(f'./{self.data_path}/{ticker}.csv', sep=',')
            dataframe['TRADE_TIMESTAMP'] = pd.to_datetime(
                pd.to_datetime(dataframe['TRADE_DATE'], format='%d/%m/%Y').astype(str) + ' ' +
                dataframe['TRADE_TIME'].astype(str), format='%Y-%m-%d %H%M%S%f')
            dataframe.set_index('TRADE_TIMESTAMP', inplace=True)
            dataframe.drop(['ISIN', 'NAME', 'TRADE_DATE', 'TRADE_TIME', 'TRADESIZE', 'TURNOVER'], axis=1,
                           inplace=True)
            self.__dataframes.update({ticker: dataframe})

    def prepare_test_runs(self, test_runs_limit) -> None:
        for ticker in self.__tickers:
            test_run = 0
            ticker_path = f'./{self.test_runs_path}/{ticker}'
            if not os.path.exists(ticker_path):
                os.makedirs(ticker_path)

            first_trading_day = time.instance(self.__dataframes.get(ticker).index[0], tz=None).start_of('day')
            last_trading_day = time.instance(self.__dataframes.get(ticker).index[-1], tz=None).start_of('day')
            for trading_day in time.period(first_trading_day, last_trading_day).range('days'):
                if test_run >= test_runs_limit:
                    break
                if trading_day.to_date_string() in self.__dataframes.get(ticker).index:
                    trading_day_dataframe = self.__dataframes.get(ticker)[trading_day.to_date_string():
                                                                          trading_day.to_date_string()].copy()
                    trading_day_dataframe = trading_day_dataframe.groupby(trading_day_dataframe.index).agg({
                        'PRICE': 'mean'})
                    trading_day_dataframe_1sec = trading_day_dataframe.resample('1s').mean().ffill()
                    trading_day_dataframe_5sec = trading_day_dataframe.resample('5s').mean().ffill()
                    trading_day_dataframe_10sec = trading_day_dataframe.resample('10s').mean().ffill()
                    trading_day_dataframe.to_csv(f'./{ticker_path}/{trading_day.to_date_string()}_ms.csv')
                    trading_day_dataframe_1sec.to_csv(f'{ticker_path}/{trading_day.to_date_string()}_1s.csv')
                    trading_day_dataframe_5sec.to_csv(f'./{ticker_path}/{trading_day.to_date_string()}_5s.csv')
                    trading_day_dataframe_10sec.to_csv(f'./{ticker_path}/{trading_day.to_date_string()}_10s.csv')
                    test_run += 1
