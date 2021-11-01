# CHATBOT-SQUAD

## Requirements
- Tensorflow=2.4.0
- CUDA=11.2
- CUDNN=11.4

## Dataset
The Stanford Question Answering Dataset (SQuAD), which is derived from Wikipedia, can be used for question answering chatbot. The SQuAD includes:
- 107,785 question-answer pairs depend on 536 articles. 

- Due to a lack of RAM, only 10.000 pairs have been used for training of the Seq2Seq model.
- According to the results, given questions to the model can be predicted by the model accurately.
- In order to increase the accuracy of the model given data to model should be enhanced. 

## Model:
To train the model, Seq2Seq architecture have been chosen.

