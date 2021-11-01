from Preprocessing import Preprocessing
from Interface import Interface
from Training import Training
from Test import Test


if __name__ == '__main__':
    preprocessing = Preprocessing()
    preprocessing.start()

    num_input_tokens, num_output_tokens, encoder_input_data, decoder_input_data, decoder_target_data =\
        preprocessing.get_training_parameters()

    training = Training()
    encoder_inputs, encoder_states, decoder_lstm, decoder_embedding,\
    decoder_dense, decoder_inputs, encoder_input_data = training.train_model(num_input_tokens,
                                                                             num_output_tokens,
                                                                             encoder_input_data,
                                                                             decoder_input_data,
                                                                             decoder_target_data)

    input_word_dict, max_input_seq_len, output_word_dict, max_output_len = preprocessing.get_test_parameters()

    test = Test(num_input_tokens, num_output_tokens, input_word_dict,
                max_input_seq_len, output_word_dict, max_output_len)

    interface = Interface(test)
    interface.start()



