import os
import numpy as np

from tensorflow import keras
from keras import Input, Model
from colorama import Fore, Style
from keras.optimizers import Adam
from keras.activations import tanh, sigmoid
from keras.layers import LSTM, Reshape, Lambda
from keras.losses import MeanAbsoluteError as LossMAE
from keras.metrics import MeanAbsoluteError as MetricMAE
from keras.callbacks import EarlyStopping, ModelCheckpoint


class SAEsLSTM:
    models_path = './models'

    def __init__(self, ticker, parent_model_config) -> None:
        self.__model = None
        self.__ticker = ticker
        self.__config = parent_model_config
        self.__loss_function = LossMAE()
        self.__model_metric = MetricMAE()
        self.__activation_functions = [tanh, sigmoid]
        self.__dropout = 0.2
        self.__recurrent_dropout = 0.2
        self.__optimizers = [Adam(learning_rate=1e-3), Adam(learning_rate=1e-4)]
        self.__early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=50, mode='auto')
        self.__model_checkpoints = ModelCheckpoint(f'{self.models_path}/{self.__ticker}/SAEs-LSTM', monitor='val_loss',
                                                   verbose=self.__config.verbosity, save_weights_only=False,
                                                   save_best_only=True, mode='auto')

    def import_model(self) -> bool:
        lstm_path = f'{self.models_path}/{self.__ticker}/SAEs-LSTM'
        if os.path.exists(lstm_path):
            self.__model = keras.models.load_model(lstm_path, compile=False, safe_mode=True)
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} ] Local SAEs-LSTM found. '
                  f'{Style.RESET_ALL}')
            return True
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} | {self.__ticker} ] No local SAEs-LSTM found. '
                  f'{Style.RESET_ALL}')
            return False

    def define_model(self, encoder, decoder) -> None:
        data_source = Input(shape=(encoder.input.shape[1],), name='source')
        encoder = encoder(data_source)
        lstm_source = Reshape(target_shape=(encoder.shape[1], 1), input_shape=encoder.shape, name='lstm_input')(encoder)
        lstm_l1 = LSTM(10, activation=self.__activation_functions[0], return_sequences=True, name='lstm_1')(lstm_source)
        lstm_l2 = LSTM(1, activation=self.__activation_functions[1], return_sequences=True, name='lstm_2')(lstm_l1)
        decoder = decoder(lstm_l2)
        lstm_lambda = Lambda(lambda tensor: tensor[:, -1], name='lstm_output')(decoder)
        self.__model = Model(data_source, lstm_lambda, name='SAEs-LSTM')

    def compile_model(self, fine_tuning=False) -> None:
        if self.__model is not None:
            if fine_tuning:
                self.__model.get_layer('Encoder').trainable = False
                self.__model.get_layer('Decoder').trainable = False
            self.__model.compile(optimizer=self.__optimizers[0] if not fine_tuning else self.__optimizers[1],
                                 loss=self.__loss_function, metrics=self.__model_metric)
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
            if test_run == 0:
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {test_run} ] '
                      f'Training SAEs-LSTM ... {Style.RESET_ALL}')
            else:
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {test_run} ] '
                      f'Fine tuning SAEs-LSTM ... {Style.RESET_ALL}')
            self.__model.fit(training_dataset, epochs=self.__config.epochs, shuffle=shuffled,
                             validation_data=validation_dataset,
                             callbacks=[self.__early_stopper, self.__model_checkpoints],
                             verbose=self.__config.verbosity)
        else:
            if test_run == 0:
                print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {test_run} ] '
                      f'Cannot train an undefined model. {Style.BRIGHT}')
            else:
                print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {test_run} ] '
                      f'ERROR : Cannot fine-tune an undefined model. {Style.BRIGHT}')

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
