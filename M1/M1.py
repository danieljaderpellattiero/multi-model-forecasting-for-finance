from LSTM import LSTM
from Config import Config
from DataManager import DataManager
from Autoencoder import Autoencoder


class M1:

    def __init__(self, config_params, tickers) -> None:
        self.__uuid = 'M1'
        self.__alias = 'WSAEs-LSTM'
        self.__config = Config(self.__uuid, config_params)
        self.__data_mgmt = DataManager(self.__config, tickers)

    def run(self) -> None:
        self.__data_mgmt.init_periods()
        self.__data_mgmt.import_dataframes()
        if not self.__data_mgmt.check_dfs_availability():
            self.__data_mgmt.download_dataframes()
            if self.__data_mgmt.check_dfs_availability():
                self.__data_mgmt.normalize_dataframes()
                self.__data_mgmt.denoise_dataframes()
                self.__data_mgmt.export_dataframes()
            else:
                exit(1)
        self.__data_mgmt.init_datasets()

        for ticker in self.__data_mgmt.tickers:
            sae = Autoencoder(ticker, self.__config)
            lstm = LSTM(ticker, self.__config)

            if not sae.import_model():
                self.generate_sae(ticker, sae)
            if not lstm.import_model():
                self.generate_lstm(ticker, sae, lstm)
                self.generate_predictions(ticker, lstm)
            else:
                self.generate_predictions(ticker, lstm)

    def generate_sae(self, ticker, sae):
        sae.define_model(self.__data_mgmt.datasets.get(ticker).get(0).get('training').get('inputs').shape)
        sae.compile_model()
        sae.train_model(self.__data_mgmt.datasets.get(ticker).get(0).get('training').get('inputs'),
                        self.__data_mgmt.datasets.get(ticker).get(0).get('validation').get('inputs'),
                        0, 1)
        sae.evaluate_model(self.__data_mgmt.datasets.get(ticker).get(0).get('test').get('inputs'))
        sae.detach_components()

    def generate_lstm(self, ticker, sae, lstm):
        lstm.define_model(sae.encoder, sae.decoder)
        for test_run in range(0, self.__config.tr_amt):
            if test_run == 0:
                lstm.compile_model()
            if test_run == 1:
                lstm.compile_model(True)
            lstm.train_model(self.__data_mgmt.datasets.get(ticker).get(test_run).get('training').get('inputs'),
                             self.__data_mgmt.datasets.get(ticker).get(test_run).get('training').get('targets'),
                             self.__data_mgmt.datasets.get(ticker).get(test_run).get('validation').get('inputs'),
                             self.__data_mgmt.datasets.get(ticker).get(test_run).get('validation').get('targets'),
                             test_run, self.__config.tr_amt)
            lstm.evaluate_model(self.__data_mgmt.datasets.get(ticker).get(test_run).get('test').get('inputs'),
                                self.__data_mgmt.datasets.get(ticker).get(test_run).get('test').get('targets'))

    def generate_predictions(self, ticker, lstm):
        for test_run in range(0, self.__config.tr_amt):
            lstm.predict(self.__data_mgmt.dataframes.get(ticker).get(test_run),
                         self.__data_mgmt.datasets.get(ticker).get(test_run),
                         ticker, test_run)
