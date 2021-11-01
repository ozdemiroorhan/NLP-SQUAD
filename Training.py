import os
import tensorflow as tf

from matplotlib import pyplot as plt
from keras.layers import Input, LSTM
from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from keras import preprocessing, utils, layers, activations, models
from tensorflow.python.client import device_lib

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


class Training:
    def __init__(self,):
        self.dimensionality = 512
        self.batch_size = 32
        self.epochs = 300
        self.data_path = "Data"
        self.weight_path = "Weight"
        self.model_name = "model.h5"

        self.model_path = os.path.join(self.weight_path, self.model_name)

        print("Available GPU's:", self.get_available_gpus())

    def train_model(self, num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data,
                    decoder_target_data):

        encoder_inputs = Input(shape=(None, ))
        encoder_embedding = Embedding(num_input_tokens, self.dimensionality, mask_zero=True)(encoder_inputs)
        encoder_outputs, state_h, state_c = LSTM(self.dimensionality, return_state=True, recurrent_dropout=0.2,
                                                 dropout=0.2)(encoder_embedding)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(num_output_tokens, self.dimensionality, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(self.dimensionality, return_state=True, return_sequences=True,
                            recurrent_dropout=0.2, dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(num_output_tokens, activation=tf.keras.activations.softmax)
        output = decoder_dense(decoder_outputs)

        if not os.path.exists(self.model_path):
            with tf.device('/gpu:0'):

                model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
                model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
                model.summary()

                history = model.fit([encoder_input_data, decoder_input_data],
                                    decoder_target_data,
                                    validation_split=0.33,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs,
                                    shuffle=True)

                self.save_history(history)
                model.save(self.model_path)
        else:
            model = load_model(self.model_path)

        return encoder_inputs, encoder_states, decoder_lstm, decoder_embedding, \
               decoder_dense, decoder_inputs, encoder_input_data

    def save_history(self, history):
        print(history.history.keys())
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy- Loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
