import os
import numpy as np

from tensorflow import keras
from keras import Input, Model
from colorama import Fore, Style
from keras.optimizers import Adam
from keras.activations import relu, tanh, sigmoid
from keras.losses import MeanSquaredError as LossMSE
from keras.metrics import MeanAbsoluteError as MetricMAE
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Bidirectional, Concatenate


class MPCNNBDLSTM:
    models_path = './models'

    def __init__(self, ticker, parent_model_config) -> None:
        self.__model = None
        self.__ticker = ticker
        self.__config = parent_model_config
        self.__loss_function = LossMSE()
        self.__activation_functions = [relu, tanh]
        self.__recurrent_activation = sigmoid
        self.__dropout = 0.5
        self.__recurrent_dropout = 0.5
        self.__model_metric = MetricMAE()
        self.__optimizer = Adam(learning_rate=1e-3)
        self.__early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=50, mode='auto')
        self.__model_checkpoints = ModelCheckpoint(f'{self.models_path}/{self.__ticker}/MP-CNN-BDLSTM',
                                                   monitor='val_loss', verbose=self.__config.verbosity,
                                                   save_weights_only=False, save_best_only=True, mode='auto')

    def import_model(self) -> bool:
        cnn_lstm_path = f'{self.models_path}/{self.__ticker}/MP-CNN-BDLSTM'
        if os.path.exists(cnn_lstm_path):
            self.__model = keras.models.load_model(cnn_lstm_path, compile=False, safe_mode=True)
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} ] Local MP-CNN-BDLSTM found. '
                  f'{Style.RESET_ALL}')
            return True
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} | {self.__ticker} ] No local MP-CNN-BDLSTM found. '
                  f'{Style.RESET_ALL}')
            return False

    def define_model(self, dataset_shape) -> None:
        data_source = Input(shape=(dataset_shape[1], 1), name='source')
        p1_cnn_l1 = Conv1D(128, kernel_size=2, strides=1, activation=self.__activation_functions[0],
                           name='p1_cnn_1')(data_source)
        p1_cnn_l2 = Conv1D(256, kernel_size=4, strides=1, activation=self.__activation_functions[0],
                           name='p1_cnn_2')(p1_cnn_l1)
        p1_cnn_mxp_l1 = MaxPooling1D(pool_size=2, name='p1_cnn_mxp_1')(p1_cnn_l2)
        p2_cnn_l1 = Conv1D(128, kernel_size=2, strides=1, activation=self.__activation_functions[0],
                           name='p2_cnn_1')(data_source)
        p2_cnn_l2 = Conv1D(256, kernel_size=4, strides=1, activation=self.__activation_functions[0],
                           name='p2_cnn_2')(p2_cnn_l1)
        p2_cnn_mxp_l1 = MaxPooling1D(pool_size=2, name='p2_cnn_mxp_1')(p2_cnn_l2)
        p3_cnn_l1 = Conv1D(128, kernel_size=2, strides=1, activation=self.__activation_functions[0],
                           name='p3_cnn_1')(data_source)
        p3_cnn_l2 = Conv1D(256, kernel_size=4, strides=1, activation=self.__activation_functions[0],
                           name='p3_cnn_2')(p3_cnn_l1)
        p3_cnn_mxp_l1 = MaxPooling1D(pool_size=2, name='p3_cnn_mxp_1')(p3_cnn_l2)
        p1_bdlstm_l1 = Bidirectional(LSTM(400, activation=self.__activation_functions[1], return_sequences=False,
                                          recurrent_activation=self.__recurrent_activation, dropout=self.__dropout,
                                          recurrent_dropout=self.__recurrent_dropout, name='p1_bdlstm_1'),
                                     merge_mode='concat')(p1_cnn_mxp_l1)
        p2_bdlstm_l1 = Bidirectional(LSTM(400, activation=self.__activation_functions[1], return_sequences=False,
                                          recurrent_activation=self.__recurrent_activation, dropout=self.__dropout,
                                          recurrent_dropout=self.__recurrent_dropout, name='p2_bdlstm_1'),
                                     merge_mode='concat')(p2_cnn_mxp_l1)
        p3_bdlstm_l1 = Bidirectional(LSTM(400, activation=self.__activation_functions[1], return_sequences=False,
                                          recurrent_activation=self.__recurrent_activation, dropout=self.__dropout,
                                          recurrent_dropout=self.__recurrent_dropout, name='p3_bdlstm_1'),
                                     merge_mode='concat')(p3_cnn_mxp_l1)
        p1_dense_l1 = Dense(32, activation=None, name='p1_dense_1')(p1_bdlstm_l1)
        p1_dense_l2 = Dense(1, activation=None, name='p1_dense_2')(p1_dense_l1)
        p2_dense_l1 = Dense(32, activation=None, name='p2_dense_1')(p2_bdlstm_l1)
        p2_dense_l2 = Dense(1, activation=None, name='p2_dense_2')(p2_dense_l1)
        p3_dense_l1 = Dense(32, activation=None, name='p3_dense_1')(p3_bdlstm_l1)
        p3_dense_l2 = Dense(1, activation=None, name='p3_dense_2')(p3_dense_l1)
        mp_concat_l1 = Concatenate(axis=1, name='mp_concat_1')([p1_dense_l2, p2_dense_l2, p3_dense_l2])
        mp_dense_l1 = Dense(1, activation=None, name='mp_dense_1')(mp_concat_l1)
        self.__model = Model(data_source, mp_dense_l1, name='MP-CNN-BDLSTM')

    def compile_model(self) -> None:
        if self.__model is not None:
            self.__model.compile(optimizer=self.__optimizer, loss=self.__loss_function, metrics=self.__model_metric)
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} ] Cannot compile an undefined model. '
                  f'{Style.BRIGHT}')

    # Utility method.
    def summary(self) -> None:
        if self.__model is not None:
            self.__model.summary()
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} ] '
                  f'Cannot print summary of an undefined model. {Style.BRIGHT}')

    def train_model(self, training_dataset, validation_dataset, test_run, shuffled=True) -> None:
        if self.__model is not None:
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {test_run} ] '
                  f'Training MP-CNN-BDLSTM ... {Style.RESET_ALL}')
            self.__model.fit(training_dataset, epochs=self.__config.epochs, shuffle=shuffled,
                             validation_data=validation_dataset,
                             callbacks=[self.__early_stopper, self.__model_checkpoints],
                             verbose=self.__config.verbosity)
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} ] Cannot train an undefined model. '
                  f'{Style.BRIGHT}')

    def evaluate_model(self, test_set_inputs, test_set_targets) -> None:
        if self.__model is not None:
            metrics_labels = self.__model.metrics_names
            metrics_scalars = self.__model.evaluate(test_set_inputs, test_set_targets, verbose=self.__config.verbosity)
            for index in range(len(metrics_labels)):
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} ] '
                      f'{metrics_labels[index]} : {np.round(metrics_scalars[index], 4)} {Style.RESET_ALL}')
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} ] Cannot evaluate an undefined model. '
                  f'{Style.BRIGHT}')

    def predict(self, test_set_inputs) -> np.ndarray:
        predictions = self.__model.predict(test_set_inputs, verbose=self.__config.verbosity)
        return predictions
