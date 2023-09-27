import os
import numpy as np

from tensorflow import keras
from keras import Input, Model
from colorama import Fore, Style
from keras.optimizers import Adam
from keras.activations import relu, tanh
from keras.losses import MeanSquaredError as LossMSE
from keras.metrics import MeanAbsoluteError as MetricMAE
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense


class CNNLSTM:

    def __init__(self, ticker, parent_model_config) -> None:
        self.__model = None
        self.__ticker = ticker
        self.__config = parent_model_config
        self.__loss_function = LossMSE()
        self.__activation_functions = [relu, tanh]
        self.__model_metric = MetricMAE()
        self.__optimizer = Adam(learning_rate=1e-3)
        self.__early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=50, mode='auto')
        self.__model_checkpoints = ModelCheckpoint(f'./models/{self.__ticker}/CNN-LSTM', monitor='val_loss',
                                                   verbose=self.__config.verbosity, save_weights_only=False,
                                                   save_best_only=True, mode='auto')

    def import_model(self) -> bool:
        cnn_lstm_path = f'./models/{self.__ticker}/CNN-LSTM'
        if os.path.exists(cnn_lstm_path):
            self.__model = keras.models.load_model(cnn_lstm_path, compile=False, safe_mode=True)
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} ] Local CNN-LSTM found. '
                  f'{Style.RESET_ALL}')
            return True
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} | {self.__ticker} ] No local CNN-LSTM found. '
                  f'{Style.RESET_ALL}')
            return False

    def define_model(self, dataset_shape) -> None:
        data_source = Input(shape=(dataset_shape[1], 1), name='source')
        cnn_l1 = Conv1D(64, kernel_size=2, activation=self.__activation_functions[0], name='cnn_1')(data_source)
        cnn_l2 = Conv1D(128, kernel_size=2, activation=self.__activation_functions[0], name='cnn_2')(cnn_l1)
        cnn_maxp = MaxPooling1D(pool_size=2, name='cnn_max_pooling')(cnn_l2)
        lstm_l1 = LSTM(200, activation=self.__activation_functions[1], return_sequences=False, name='lstm_1')(cnn_maxp)
        dense_l1 = Dense(32, activation=None, name='dense_1')(lstm_l1)
        dense_l2 = Dense(1, activation=None, name='dense_2')(dense_l1)
        self.__model = Model(data_source, dense_l2, name='CNN-LSTM')

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
                  f'Training CNN-LSTM {Style.RESET_ALL}')
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
