from Config import Config
from SAEsLSTM import SAEsLSTM
from DataManager import DataManager
from StackedAutoEncoders import StackedAutoEncoders


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
            self.__data_mgmt.normalize_dataframes()
            self.__data_mgmt.denoise_dataframes()
            self.__data_mgmt.export_dataframes()
            if not self.__data_mgmt.check_local_data_availability():
                exit(1)
        self.__data_mgmt.init_datasets()
        self.__data_mgmt.init_batches()
        self.__data_mgmt.init_alternative_dataset()

        for ticker in self.__data_mgmt.tickers:
            SAEs = StackedAutoEncoders(ticker, self.__config)
            model = SAEsLSTM(ticker, self.__config)
            if not SAEs.import_model():
                self.generate_saes(ticker, SAEs)
            if not model.import_model():
                self.generate_model(ticker, SAEs, model)
            self.generate_predictions(ticker, model)
            self.__data_mgmt.reconstruct_and_export_results(ticker)

    def generate_saes(self, ticker, SAEs) -> None:
        SAEs.define_model(self.__data_mgmt.tr_datasets.get(ticker).get(0).get('training').get('inputs').shape)
        SAEs.compile_model()
        SAEs.train_model(self.__data_mgmt.tr_btch_datasets.get(ticker).get(0).get('training'),
                         self.__data_mgmt.tr_btch_datasets.get(ticker).get(0).get('validation'),
                         0, 1)
        SAEs.evaluate_model(self.__data_mgmt.tr_btch_datasets.get(ticker).get(0).get('test').get('inputs'))
        SAEs.detach_components()

    def generate_model(self, ticker, SAEs, model) -> None:
        model.define_model(SAEs.encoder, SAEs.decoder)
        for test_run in range(0, self.__config.tr_amt):
            if test_run == 0:
                model.compile_model()
            if test_run == 1:
                model.compile_model(True)
            model.train_model(self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('training'),
                              self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('validation'),
                              test_run, self.__config.tr_amt)
            model.evaluate_model(self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('test').get('inputs'),
                                 self.__data_mgmt.tr_btch_datasets.get(ticker).get(test_run).get('test').get('targets'))

    def generate_predictions(self, ticker, lstm) -> None:
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
