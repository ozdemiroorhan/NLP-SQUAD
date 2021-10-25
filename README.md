# CHATBOT-SQUAD

## Requirements
- Tensorflow=2.4.0
- CUDA=11.2
- CUDNN=11.4

## Dataset
The Stanford Question Answering Dataset (SQuAD), which is derived from Wikipedia, can be used for question answering chatbot. The SQuAD includes:
- 107,785 question-answer pairs depend on 536 articles. 

- Due to a lack of RAM, only 1.000 pairs have been used for training the Seq2Seq model.
- According to the results, question’s answer can be predicted by the model accurately.
- In order to increase the accuracy of the model given data to model should be enhanced. 

## Model:
To train the model, LSTM architecture have been chosen.

