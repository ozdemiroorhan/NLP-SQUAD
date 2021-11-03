import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import load_dataset
from keras import preprocessing, utils, layers, activations, models

global num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data, decoder_target_data,\
    input_word_dict, max_input_seq_len, output_word_dict, max_output_len


class Preprocessing:
    def __init__(self):

        self.data_limit = 10000
        self.data_folder = "Data"
        self.data_name = "SquadDataset.csv"
        self.data_final_name = "SquadDatasetFinal.csv"

        self.data_path = os.path.join(self.data_folder, self.data_name)
        self.final_data_path = os.path.join(self.data_folder, self.data_final_name)

        self.model_name = "model.h5"
        self.enc_model_name = "enc_model.h5"
        self.dec_model_name = "dec_model.h5"
        self.data_name = "lines.csv"

        self.data_frame = pd.DataFrame()

    def start(self):
        global num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data, decoder_target_data, \
            input_word_dict, max_input_seq_len, output_word_dict, max_output_len

        if os.path.exists(self.final_data_path):
            self.data_frame = pd.read_csv(self.final_data_path)
        else:
            self.check_file()
            self.clean_data()
            self.regulate_phrase()

        tokenized_input_lines, max_input_seq_len, tokenizer = self.set_input_tokens(self.data_frame)
        encoder_input_data, input_word_dict, num_input_tokens = self.pad_input_lines(tokenized_input_lines,
                                                                                     max_input_seq_len,
                                                                                     tokenizer)

        tokenized_output_lines, max_output_len, decoder_tokenizer = self.set_output_tokens(self.data_frame)
        decoder_input_data, output_word_dict, num_output_tokens = self.pad_output_lines(tokenized_output_lines,
                                                                                        max_output_len,
                                                                                        decoder_tokenizer)

        decoder_target_data = self.set_target_tokens(tokenized_output_lines, max_output_len, num_output_tokens)

    def check_file(self):
        if os.path.exists(self.data_path):
            self.data_frame = pd.read_csv(self.data_path)
        else:
            self.download_dataset()

    def download_dataset(self):
        self.data_frame = load_dataset('squad', split='train')
        self.data_frame = pd.DataFrame(data=self.data_frame)

        self.data_frame.to_csv(self.data_path)

    def clean_data(self):
        answers_list = []

        self.data_frame = self.data_frame.dropna(axis=0)
        answers = self.data_frame["answers"]

        for index, row in answers.iteritems():
            answers_list.append(row["text"][0])

        self.data_frame["answers"] = answers_list
        self.data_frame.to_csv(self.data_path)

    def replace_phrase(self, text):
        text = text.lower()
        text = re.sub(r"there's", "there is", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
        text = text.strip()

        return text

    def regulate_phrase(self):
        self.data_frame = self.data_frame[["question", "answers"]]
        self.data_frame["question"] = self.data_frame["question"].apply(self.replace_phrase)
        self.data_frame["answers"] = self.data_frame["answers"].apply(self.replace_phrase)
        self.data_frame = self.data_frame.iloc[:self.data_limit]
        self.data_frame.to_csv(self.final_data_path)

    def set_input_tokens(self, pairs):
        input_lines, input_seq_length = [], []

        for line in pairs.question:
            input_lines.append(line)

        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(input_lines)
        tokenized_input_lines = tokenizer.texts_to_sequences(input_lines)

        for token_seq in tokenized_input_lines:
            input_seq_length.append(len(token_seq))

        max_input_seq_len = np.array(input_seq_length).max()

        print("Max. input sequence length is {}".format(max_input_seq_len))

        return tokenized_input_lines, max_input_seq_len, tokenizer

    def pad_input_lines(self, tokenized_input_lines, max_input_seq_len, tokenizer):
        padded_input_lines = preprocessing.sequence.pad_sequences(tokenized_input_lines,
                                                                  maxlen=max_input_seq_len,
                                                                  padding='post')
        encoder_input_data = np.array(padded_input_lines)

        print('Encoder input data shape: {}'.format(encoder_input_data.shape))

        input_word_dict = tokenizer.word_index
        num_input_tokens = len(input_word_dict) + 1

        print("Number of input tokens = {}".format(num_input_tokens))

        return encoder_input_data, input_word_dict, num_input_tokens

    def set_output_tokens(self, pairs):
        output_lines, output_len_list = [], []

        for line in pairs.answers:
            output_lines.append('<START> ' + line + ' <END>')

        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(output_lines)
        tokenized_output_lines = tokenizer.texts_to_sequences(output_lines)

        for token_seq in tokenized_output_lines:
            output_len_list.append(len(token_seq))

        max_output_len = np.array(output_len_list).max()
        print("Output max length is {}".format(max_output_len))

        return tokenized_output_lines, max_output_len, tokenizer

    def pad_output_lines(self, tokenized_output_lines, max_output_len, decoder_tokenizer):
        padded_output_lines = preprocessing.sequence.pad_sequences(tokenized_output_lines,
                                                                   maxlen=max_output_len,
                                                                   padding="post")
        decoder_input_data = np.array(padded_output_lines)
        print("Decoder input data shape: {}".format(decoder_input_data.shape))

        output_word_dict = decoder_tokenizer.word_index
        num_output_tokens = len(output_word_dict) + 1
        print("Number of output tokens: {}".format(num_output_tokens))

        return decoder_input_data, output_word_dict, num_output_tokens

    def set_target_tokens(self, tokenized_output_lines, max_output_len, num_output_tokens):
        decoder_target_data = []

        for token_seq in tokenized_output_lines:
            decoder_target_data.append(token_seq[1:])

        padded_output_lines = preprocessing.sequence.pad_sequences(decoder_target_data,
                                                                   maxlen=max_output_len,
                                                                   padding='post')
        onehot_output_lines = tf.keras.utils.to_categorical(padded_output_lines, num_output_tokens)
        decoder_target_data = np.array(onehot_output_lines)
        print('Decoder target data shape: {}'.format(decoder_target_data.shape))

        return decoder_target_data

    @staticmethod
    def get_training_parameters():
        global num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data, decoder_target_data
        return num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data, decoder_target_data

    @staticmethod
    def get_test_parameters():
        global input_word_dict, max_input_seq_len, output_word_dict, max_output_len
        return input_word_dict, max_input_seq_len, output_word_dict, max_output_len
