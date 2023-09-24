import os
import numpy as np

from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense
from colorama import Fore, Style
from keras.regularizers import L2
from keras.activations import elu
from keras.optimizers import Adadelta
from keras.losses import MeanAbsoluteError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import EarlyStopping, ModelCheckpoint


class StackedAutoEncoders:

    def __init__(self, ticker, parent_model_config) -> None:
        self.__model = None
        self.__encoder = None
        self.__decoder = None
        self.__ticker = ticker
        self.__config = parent_model_config
        self.__activation_function = elu
        self.__loss_function = MeanAbsoluteError()
        self.__model_metric = MeanAbsoluteError()
        self.__layer_weight_regularizer = L2(0.01)
        self.__optimizer = Adadelta(learning_rate=1.0)
        self.__early_stopper = EarlyStopping(monitor=self.__loss_function, patience=50, mode='auto')
        self.__model_checkpoints = ModelCheckpoint(f'./models/{self.__ticker}/SAEs', monitor='val_loss',
                                                   verbose=0, save_weights_only=False, save_best_only=True, mode='auto')

    @property
    def encoder(self) -> keras.Model:
        return self.__encoder

    @property
    def decoder(self) -> keras.Model:
        return self.__decoder

    def import_model(self) -> bool:
        autoencoder_path = f'./models/{self.__ticker}/SAEs'
        if os.path.exists(autoencoder_path):
            self.__model = keras.models.load_model(autoencoder_path, compile=False, safe_mode=True)
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} ] Local SAEs found. '
                  f'{Style.RESET_ALL}')
            self.detach_components()
            return True
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} | {self.__ticker} ] No local SAEs found. '
                  f'{Style.RESET_ALL}')
            return False

    def define_model(self, dataset_shape) -> None:
        data_source = Input(shape=(dataset_shape[1],), name='encoder_input')
        dense_l1 = Dense(15, activation=self.__activation_function,
                         activity_regularizer=self.__layer_weight_regularizer, name='encoder_l1')(data_source)
        dense_l2 = Dense(12, activation=self.__activation_function,
                         activity_regularizer=self.__layer_weight_regularizer, name='encoder_l2')(dense_l1)
        dense_l3 = Dense(10, activation=self.__activation_function,
                         activity_regularizer=self.__layer_weight_regularizer, name='encoder_output')(dense_l2)
        dense_l4 = Dense(12, activation=self.__activation_function,
                         activity_regularizer=self.__layer_weight_regularizer, name='decoder_input')(dense_l3)
        dense_l5 = Dense(15, activation=self.__activation_function,
                         activity_regularizer=self.__layer_weight_regularizer, name='decoder_l1')(dense_l4)
        dense_l6 = Dense(dataset_shape[1], activation=self.__activation_function,
                         activity_regularizer=self.__layer_weight_regularizer, name='decoder_output')(dense_l5)
        self.__model = Model(data_source, dense_l6, name='SAEs')

    def compile_model(self) -> None:
        if self.__model is not None:
            self.__model.compile(optimizer=self.__optimizer, loss=self.__loss_function, metrics=self.__model_metric)
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run 0 ] '
                  f'Cannot compile an undefined model. {Style.BRIGHT}')

    # Utility method.
    def summary(self) -> None:
        if self.__model is not None:
            self.__model.summary()
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker}  | Test run 0 ] '
                  f'Cannot print summary of an undefined model. {Style.BRIGHT}')

    def train_model(self, training_dataset, validation_dataset, test_run, shuffled=True) -> None:
        if self.__model is not None:
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {test_run} ] '
                  f'Training SAEs {Style.RESET_ALL}')
            self.__model.fit(training_dataset, epochs=self.__config.epochs,
                             shuffle=shuffled, validation_data=validation_dataset,
                             callbacks=[self.__early_stopper, self.__model_checkpoints],
                             verbose=self.__config.verbosity)
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run {test_run} ] '
                  f'Cannot train an undefined model. {Style.BRIGHT}')

    def evaluate_model(self, test_set) -> None:
        if self.__model is not None:
            metrics_labels = self.__model.metrics_names
            metrics_scalars = self.__model.evaluate(test_set, test_set, verbose=self.__config.verbosity)
            for index in range(len(metrics_labels)):
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {self.__ticker} ] '
                      f'{metrics_labels[index]} : {np.round(metrics_scalars[index], 4)} {Style.RESET_ALL}')
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} | {self.__ticker} | Test run 0 ] '
                  f'Cannot evaluate an undefined model. {Style.BRIGHT}')

    def detach_components(self) -> None:
        self.__encoder = keras.Model(self.__model.input, self.__model.get_layer('encoder_output').output,
                                     name='Encoder')
        self.__decoder = keras.Model(self.__model.get_layer('decoder_input').input, self.__model.output,
                                     name='Decoder')
