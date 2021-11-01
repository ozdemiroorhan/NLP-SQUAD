import os
import numpy as np
import tensorflow as tf

from keras.layers import Input, LSTM
from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from keras import preprocessing, layers, models


class Test:
    def __init__(self, num_input_tokens, num_output_tokens, input_word_dict,
                 max_input_seq_len, output_word_dict, max_output_len):
        self.dimensionality = 512

        self.data_path = "Data"
        self.weight_path = "Weight"
        self.model_name = "model.h5"
        self.enc_model_name = "enc_model.h5"
        self.dec_model_name = "dec_model.h5"

        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.input_word_dict = input_word_dict
        self.max_input_seq_len = max_input_seq_len
        self.output_word_dict = output_word_dict
        self.max_output_len = max_output_len

        self.model_path = os.path.join(self.weight_path, self.model_name)
        self.enc_path = os.path.join(self.weight_path, self.enc_model_name)
        self.dec_path = os.path.join(self.weight_path, self.dec_model_name)

    def make_inference_models(self, encoder_inputs, encoder_states, decoder_lstm,
                              decoder_embedding, decoder_dense, decoder_inputs):

        encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

        decoder_state_input_h = tf.keras.layers.Input(shape=(self.dimensionality,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(self.dimensionality,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = tf.keras.models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        encoder_model.save(self.enc_path)
        decoder_model.save(self.dec_path)

        return encoder_model, decoder_model

    def str_to_tokens(self, sentence: str, input_word_dict, max_input_length):
        words = sentence.lower().split()
        tokens_list = []
        word = """ coming from user's input"""

        try:
            for word in words:
                tokens_list.append(input_word_dict[word])
        except Exception as e:
            print(f" ' {word} 'not exist in the vocabulary".format(word=word))

        return preprocessing.sequence.pad_sequences([tokens_list], maxlen=max_input_length, padding='post')

    def predict(self, user_input):

        if not os.path.exists(self.enc_path) and not os.path.exists(self.dec_path):

            encoder_inputs = Input(shape=(None,))
            encoder_embedding = Embedding(self.num_input_tokens, self.dimensionality, mask_zero=True)(encoder_inputs)
            encoder_outputs, state_h, state_c = LSTM(self.dimensionality, return_state=True, recurrent_dropout=0.2,
                                                     dropout=0.2)(encoder_embedding)
            encoder_states = [state_h, state_c]

            decoder_inputs = Input(shape=(None,))
            decoder_embedding = Embedding(self.num_output_tokens, self.dimensionality, mask_zero=True)(decoder_inputs)
            decoder_lstm = LSTM(self.dimensionality, return_state=True, return_sequences=True,
                                recurrent_dropout=0.2, dropout=0.2)

            decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
            decoder_dense = tf.keras.layers.Dense(self.num_output_tokens, activation=tf.keras.activations.softmax)
            output = decoder_dense(decoder_outputs)

            enc_model, dec_model = self.make_inference_models(encoder_inputs, encoder_states, decoder_lstm,
                                                              decoder_embedding, decoder_dense, decoder_inputs)
        else:
            enc_model = load_model(self.enc_path)
            dec_model = load_model(self.dec_path)

        states_values = enc_model.predict(self.str_to_tokens(user_input, self.input_word_dict, self.max_input_seq_len))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = self.output_word_dict['start']
        stop_condition = False
        decoded_translation = ''

        while not stop_condition:

            dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None

            for word, index in self.output_word_dict.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > self.max_output_len:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        return decoded_translation.replace("end", "")
