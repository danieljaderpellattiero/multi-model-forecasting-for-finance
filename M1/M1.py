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
        self.__data_mgmt.import_local_data()
        if not self.__data_mgmt.check_local_data_availability():
            self.__data_mgmt.download_dataframes()
            if self.__data_mgmt.check_local_data_availability():
                self.__data_mgmt.normalize_dataframes()
                self.__data_mgmt.denoise_dataframes()
                self.__data_mgmt.export_data()
            else:
                exit(1)
        self.__data_mgmt.init_datasets()
        self.__data_mgmt.init_batches()
        self.__data_mgmt.init_alternative_dataset()

        for ticker in self.__data_mgmt.tickers:
            sae = Autoencoder(ticker, self.__config)
            lstm = LSTM(ticker, self.__config)
            if not sae.import_model():
                self.generate_sae(ticker, sae)
            if not lstm.import_model():
                self.generate_lstm(ticker, sae, lstm)
            self.generate_predictions(ticker, lstm)
            self.__data_mgmt.reconstruct_and_export_results(ticker)

    def generate_sae(self, ticker, sae):
        sae.define_model(self.__data_mgmt.tr_datasets.get(ticker).get(0).get('training').get('inputs').shape)
        sae.compile_model()
        sae.train_model(self.__data_mgmt.tr_btch_datasets.get(ticker).get(0).get('training'),
                        self.__data_mgmt.tr_btch_datasets.get(ticker).get(0).get('validation'),
                        0, 1)
        sae.evaluate_model(self.__data_mgmt.tr_btch_datasets.get(ticker).get(0).get('test').get('inputs'))
        sae.detach_components()

    def generate_lstm(self, ticker, sae, lstm):
        lstm.define_model(sae.encoder, sae.decoder)
        for test_run in range(0, self.__config.tr_amt):
            if test_run == 0:
                lstm.compile_model()
            if test_run == 1:
                lstm.compile_model(True)
            lstm.train_model(self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('training'),
                             self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('validation'),
                             test_run, self.__config.tr_amt)
            lstm.evaluate_model(self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('test').get('inputs'),
                                self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('test').get('targets'))

    def generate_predictions(self, ticker, lstm):
        predictions = {}
        backtracked_predictions = {}
        for test_run in range(0, self.__config.tr_amt):
            predictions.update({
                test_run: lstm.predict(self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('test')
                                                                                                  .get('inputs'))
            })
            backtracked_predictions_tmp = self.__data_mgmt.init_backtrack_buffer(predictions.get(test_run))
            for ref_index, ref in enumerate(self.__data_mgmt.tr_bt_datasets.get(ticker).get(test_run)):
                backtracked_sequence_set = ('training' if ref <= (self.__data_mgmt.tr_datasets.get(ticker).get(test_run)
                                                                  .get('training').get('inputs').shape[0] - 1)
                                            else 'validation')
                actual_ref = (ref if backtracked_sequence_set == 'training'
                              else ref - self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get('training')
                                                                     .get('inputs').shape[0])
                backtracked_sequence = (self.__data_mgmt.tr_datasets.get(ticker).get(test_run)
                                                                    .get(backtracked_sequence_set)
                                                                    .get('inputs')[actual_ref])
                backtracked_predictions_tmp[ref_index][0] = lstm.predict(
                                                                backtracked_sequence.reshape(1,
                                                                                             self.__config.window_size))
                backtracked_predictions_tmp[ref_index][1] = (self.__data_mgmt.tr_datasets.get(ticker).get(test_run)
                                                                                         .get(backtracked_sequence_set)
                                                                                         .get('targets')[actual_ref])
            backtracked_predictions.update({test_run: backtracked_predictions_tmp})
        self.__data_mgmt.tr_predictions.update({ticker: predictions})
        self.__data_mgmt.tr_bt_predictions.update({ticker: backtracked_predictions})
