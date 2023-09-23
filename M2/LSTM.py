import os
import numpy as np

from tensorflow import keras
from colorama import Fore, Style
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class LSTM:

    def __init__(self, parent_model_config, ticker, test_run, component, epochs) -> None:
        self.__model = None
        self.__ticker = ticker
        self.__test_run = test_run
        self.__component = component
        self.__config = parent_model_config
        self.__epochs = epochs
        self.__model_metric = 'mse'
        self.__loss_function = 'mse'
        self.__activation = 'tanh'
        self.__recurrent_activation = 'sigmoid'
        self.__dropout = 0.2
        self.__recurrent_dropout = 0.2
        self.__optimizer = Adam(learning_rate=1e-3)
        self.__early_stopper = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
        self.__reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')
        self.__model_checkpoints = ModelCheckpoint(f'./models/{ticker}/test_run_{test_run}/CEEMDAN_LSTM_{component}',
                                                   monitor='val_loss', verbose=0, save_weights_only=False,
                                                   save_best_only=True,
                                                   mode='auto')

    def import_model(self) -> bool:
        lstm_path = f'./models/{self.__ticker}/test_run_{self.__test_run}/CEEMDAN_LSTM_{self.__component}'
        if os.path.exists(lstm_path):
            self.__model = keras.models.load_model(lstm_path, compile=False)
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {self.__test_run} ] '
                  f'Local LSTM found for component {self.__component.upper()}. {Style.RESET_ALL}')
            return True
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {self.__test_run} ] '
                  f'No local LSTM found for component {self.__component.upper()}. {Style.RESET_ALL}')
            return False

    def define_model(self, dataset_shape) -> None:
        data_source = keras.Input(shape=(dataset_shape[1], 1), name='source')
        lstm_l1 = keras.layers.LSTM(128, activation=self.__activation, return_sequences=True,
                                    recurrent_activation=self.__recurrent_activation, dropout=self.__dropout,
                                    recurrent_dropout=self.__recurrent_dropout, name='lstm_1')(data_source)
        lstm_l2 = keras.layers.LSTM(64, activation=self.__activation, return_sequences=True,
                                    recurrent_activation=self.__recurrent_activation, dropout=self.__dropout,
                                    recurrent_dropout=self.__recurrent_dropout, name='lstm_2')(lstm_l1)
        lstm_l3 = keras.layers.LSTM(32, activation=self.__activation, return_sequences=False,
                                    recurrent_activation=self.__recurrent_activation, dropout=self.__dropout,
                                    recurrent_dropout=self.__recurrent_dropout, name='lstm_3')(lstm_l2)
        dense_l1 = keras.layers.Dense(1, activation=self.__activation, name='dense_1')(lstm_l3)
        self.__model = keras.Model(data_source, dense_l1, name='CEEMDAN_LSTM')

    def compile_model(self) -> None:
        if self.__model is not None:
            self.__model.compile(optimizer=self.__optimizer, loss=self.__loss_function, metrics=self.__model_metric)
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {self.__test_run} ] '
                  f'Cannot compile an undefined model. {Style.BRIGHT}')

    # Utility method.
    def summary(self) -> None:
        if self.__model is not None:
            self.__model.summary()
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {self.__test_run} ] '
                  f'Cannot print summary of an undefined model. {Style.BRIGHT}')

    def train_model(self, training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets,
                    shuffled=True) -> None:
        if self.__model is not None:
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {self.__test_run} ] '
                  f'Training LSTM ({self.__component.upper()} with {self.__epochs} epochs) {Style.RESET_ALL}')
            self.__model.fit(training_set_inputs, training_set_targets, epochs=self.__epochs, shuffle=shuffled,
                             validation_data=(validation_set_inputs, validation_set_targets),
                             callbacks=[self.__early_stopper, self.__reduce, self.__model_checkpoints],
                             verbose=self.__config.verbosity)
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {self.__test_run} ] '
                  f'Cannot train an undefined model. {Style.BRIGHT}')

    def evaluate_model(self, test_set_inputs, test_set_targets) -> None:
        if self.__model is not None:
            metrics_labels = self.__model.metrics_names
            metrics_scalars = self.__model.evaluate(test_set_inputs, test_set_targets, verbose=self.__config.verbosity)
            for index in range(len(metrics_labels)):
                print(f'{Fore.LIGHTGREEN_EX} '
                      f'[ {self.__config.uuid} | {self.__ticker} | Test run {self.__test_run} ] '
                      f'{metrics_labels[index]} : {np.round(metrics_scalars[index], 4)} {Style.RESET_ALL}')
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {self.__test_run} ] '
                  f'Cannot evaluate an undefined model. {Style.BRIGHT}')

    def predict(self, test_set_inputs) -> np.ndarray:
        predictions = self.__model.predict(test_set_inputs, verbose=self.__config.verbosity)
        return predictions
