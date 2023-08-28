import os
import logging
import numpy as np

from tensorflow import keras
from colorama import Fore, Style

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class Autoencoder:

    def __init__(self, ticker, parent_model_config) -> None:
        self.__model = None
        self.__encoder = None
        self.__decoder = None
        self.__loss_function = 'mae'
        self.__activation_function = 'elu'
        self.__model_metrics = ['mae']
        self.__ticker = ticker
        self.__config = parent_model_config
        self.__layer_weight_regularizer = keras.regularizers.L2(0.01)
        self.__optimizer = keras.optimizers.Adadelta(learning_rate=1.0)
        self.__early_stopper = keras.callbacks.EarlyStopping(monitor='mae', min_delta=1e-4, patience=10,
                                                             mode='auto')
        self.__model_checkpoints = keras.callbacks.ModelCheckpoint(f'./models/{self.__ticker}/SAEs.h5',
                                                                   monitor='val_loss', verbose=0,
                                                                   save_weights_only=False, save_best_only=True,
                                                                   mode='min')

    @property
    def encoder(self) -> keras.Model:
        return self.__encoder

    @property
    def decoder(self) -> keras.Model:
        return self.__decoder

    def import_model(self) -> bool:
        local_autoencoder_path = f'./models/{self.__ticker}/SAEs.h5'
        if os.path.exists(local_autoencoder_path):
            self.__model = keras.models.load_model(local_autoencoder_path, compile=False, safe_mode=True)
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} § {self.__ticker} ] Local SAEs found. '
                  f'{Style.RESET_ALL}')
            self.detach_components()
            return True
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} § {self.__ticker} ] No local SAEs found. '
                  f'{Style.RESET_ALL}')
            return False

    def define_model(self, dataset_shape) -> None:
        encoder_input = keras.Input(shape=(dataset_shape[1],), name='enc_input')
        encoder_l1 = keras.layers.Dense(15, activation=self.__activation_function,
                                        activity_regularizer=self.__layer_weight_regularizer,
                                        name='enc_1')(encoder_input)
        encoder_l2 = keras.layers.Dense(12, activation=self.__activation_function,
                                        activity_regularizer=self.__layer_weight_regularizer,
                                        name='enc_2')(encoder_l1)
        encoder_output = keras.layers.Dense(10, activation=self.__activation_function,
                                            activity_regularizer=self.__layer_weight_regularizer,
                                            name='enc_output')(encoder_l2)

        decoder_input = keras.layers.Dense(12, activation=self.__activation_function,
                                           activity_regularizer=self.__layer_weight_regularizer,
                                           name='dec_input')(encoder_output)
        decoder_l1 = keras.layers.Dense(15, activation=self.__activation_function,
                                        activity_regularizer=self.__layer_weight_regularizer,
                                        name='dec_1')(decoder_input)
        decoder_output = keras.layers.Dense(dataset_shape[1], activation=self.__activation_function,
                                            activity_regularizer=self.__layer_weight_regularizer,
                                            name='dec_output')(decoder_l1)

        self.__model = keras.Model(encoder_input, decoder_output, name='SAE')

    def compile_model(self) -> None:
        if self.__model is not None:
            self.__model.compile(optimizer=self.__optimizer, loss=self.__loss_function, metrics=self.__model_metrics)
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

    def train_model(self, training_set, validation_set, test_run, test_run_amount, shuffled=True) -> None:
        if self.__model is not None:
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} § {self.__ticker} ] Training SAEs ({test_run + 1} '
                  f'of {test_run_amount}) {Style.RESET_ALL}')
            self.__model.fit(training_set, training_set, epochs=self.__config.epochs,
                             batch_size=self.__config.batch_size, shuffle=shuffled,
                             validation_data=(validation_set, validation_set),
                             callbacks=[self.__early_stopper, self.__model_checkpoints],
                             verbose=self.__config.verbosity)
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} § {self.__ticker} ] Cannot train an undefined model. '
                  f'{Style.BRIGHT}')

    def evaluate_model(self, test_set) -> None:
        if self.__model is not None:
            metrics_labels = self.__model.metrics_names
            metrics_scalars = self.__model.evaluate(test_set, test_set, verbose=self.__config.verbosity)
            for index in range(len(metrics_labels)):
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} § {self.__ticker} ] SAEs {metrics_labels[index]} : '
                      f'{np.round(metrics_scalars[index], 4)} {Style.RESET_ALL}')
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} § {self.__ticker} ] Cannot evaluate an undefined model. '
                  f'{Style.BRIGHT}')

    def detach_components(self) -> None:
        self.__encoder = keras.Model(self.__model.input, self.__model.get_layer('enc_output').output,
                                     name='Encoder')
        self.__decoder = keras.Model(self.__model.get_layer('dec_input').input, self.__model.output,
                                     name='Decoder')
