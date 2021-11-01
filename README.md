# CHATBOT-SQUAD

## Requirements
- Tensorflow=2.4.0
- CUDA=11.2
- CUDNN=11.4
- Python 3.6

## Dataset
The Stanford Question Answering Dataset (SQuAD), which is derived from Wikipedia, can be used for question answering chatbot. The SQuAD includes:
- 107,785 question-answer pairs depend on 536 articles. 

- Due to a lack of RAM, only 10.000 pairs have been used for training of the Seq2Seq model.
- According to the results, given questions to the model can be predicted by the model accurately.
- Given data to the model should be enhanced in order to increase the accuracy.


## Folders:
### Data: 
- Downloaded data is in this folder. Dowloading script is available in Preporeccsing.py. The program automatically will download it under Data folder, if it is exist.
### Image: 
- Model accuracy will be saved in this folder after training process.
### Weight
- Model weight, encoder and decoder model will be saved under this folder.

## Model:
To train the model, Seq2Seq architecture have been chosen.

