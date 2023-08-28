import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from colorama import Fore, Style

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class LSTM:

    def __init__(self, ticker, parent_model_config) -> None:
        self.__model = None
        self.__loss_function = 'mae'
        self.__activation_functions = ['tanh', 'sigmoid']
        self.__model_metrics = ['mae']
        self.__dropout = 0.2
        self.__recurrent_dropout = 0.2
        self.__ticker = ticker
        self.__config = parent_model_config
        self.__optimizers = [keras.optimizers.Adam(learning_rate=1e-3), keras.optimizers.Adam(learning_rate=1e-4)]
        self.__early_stopper = keras.callbacks.EarlyStopping(monitor='mae', min_delta=1e-5, patience=10, mode='auto')
        self.__model_checkpoints = keras.callbacks.ModelCheckpoint(f'./models/{self.__ticker}/SAEs-LSTM.h5',
                                                                   monitor='val_loss', verbose=0,
                                                                   save_weights_only=False, save_best_only=True,
                                                                   mode='min')

    def import_model(self) -> bool:
        local_lstm_path = f'./models/{self.__ticker}/SAEs-LSTM.h5'
        if os.path.exists(local_lstm_path):
            self.__model = keras.models.load_model(local_lstm_path, compile=False, safe_mode=True)
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} § {self.__ticker} ] Local SAEs-LSTM found. '
                  f'{Style.RESET_ALL}')
            return True
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} § {self.__ticker} ] No local SAEs-LSTM found. '
                  f'{Style.RESET_ALL}')
            return False

    def define_model(self, encoder, decoder) -> None:
        data_source = keras.Input(shape=(encoder.input.shape[1],), name='data_source')
        encoder_layer = encoder(data_source)
        lstm_input = keras.layers.Reshape(target_shape=(encoder_layer.shape[1], 1),
                                          input_shape=encoder_layer.shape, name='lstm_input')(encoder_layer)
        lstm_l1 = keras.layers.LSTM(10, activation=self.__activation_functions[0], return_sequences=True,
                                    name='lstm_1')(lstm_input)
        lstm_l2 = keras.layers.LSTM(1, activation=self.__activation_functions[1], return_sequences=True,
                                    name='lstm_2')(lstm_l1)
        decoder_layer = decoder(lstm_l2)
        lstm_output = keras.layers.Lambda(lambda tensor: tensor[:, -1], name='lstm_output')(decoder_layer)
        self.__model = keras.Model(data_source, lstm_output, name='SAEs-LSTM')

    def compile_model(self, fine_tuning=False) -> None:
        if self.__model is not None:
            self.__model.compile(optimizer=self.__optimizers[0] if not fine_tuning else self.__optimizers[1],
                                 loss=self.__loss_function, metrics=self.__model_metrics)
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} § {self.__ticker} ] Cannot compile an undefined model. '
                  f'{Style.BRIGHT}')

    # Utility method
    def summary(self) -> None:
        if self.__model is not None:
            self.__model.summary()
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} § {self.__ticker} ] Cannot print summary of an undefined '
                  f'model. {Style.BRIGHT}')

    def train_model(self, training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets,
                    test_run, test_run_amount, shuffled=True) -> None:
        if self.__model is not None:
            if test_run == 0:
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} § {self.__ticker} ] Training SAEs-LSTM '
                      f'({test_run + 1} of {test_run_amount}) {Style.RESET_ALL}')
            else:
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} § {self.__ticker} ] Fine-tuning SAEs-LSTM '
                      f'({test_run + 1} of {test_run_amount}) {Style.RESET_ALL}')
            self.__model.fit(training_set_inputs, training_set_targets, epochs=self.__config.epochs, shuffle=shuffled,
                             validation_data=(validation_set_inputs, validation_set_targets),
                             callbacks=[self.__early_stopper, self.__model_checkpoints],
                             verbose=self.__config.verbosity)
        else:
            if test_run == 0:
                print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} § {self.__ticker} ] Cannot train an undefined model. '
                      f'{Style.BRIGHT}')
            else:
                print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} § {self.__ticker} ] ERROR : Cannot fine-tune an '
                      f'undefined model. {Style.BRIGHT}')

    def evaluate_model(self, test_set_inputs, test_set_targets) -> None:
        if self.__model is not None:
            metrics_labels = self.__model.metrics_names
            metrics_scalars = self.__model.evaluate(test_set_inputs, test_set_targets, verbose=self.__config.verbosity)
            for index in range(len(metrics_labels)):
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} § {self.__ticker} ] SAEs-LSTM '
                      f'{metrics_labels[index]} : {np.round(metrics_scalars[index], 4)} {Style.RESET_ALL}')
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} § {self.__ticker} ] Cannot evaluate an undefined model. '
                  f'{Style.BRIGHT}')

    def predict(self, dataframe, dataset, ticker, test_run) -> None:
        png_export_path = f'./images/predictions/{ticker}'
        if not os.path.exists(png_export_path):
            os.makedirs(png_export_path)

        predictions = self.__model.predict(dataset.get('test').get('inputs'), verbose=self.__config.verbosity)
        predictions_index = dataframe.get('test').index[-predictions.shape[0]:]
        predictions_pd_series = pd.Series(predictions.flatten(), index=predictions_index)

        plt.figure(figsize=(16, 9))
        plt.title(f'{ticker} forecasting results (test run {test_run})')
        plt.plot(dataframe.get('training'), label='training_set')
        plt.plot(dataframe.get('validation'), label='validation_set')
        plt.plot(dataframe.get('test'), label='test_set')
        plt.plot(predictions_pd_series, label='model_predictions')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{png_export_path}/test_run_{test_run}.png')
        plt.close()
