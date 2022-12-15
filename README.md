# Question Answering On SQUAD
## Short-Video
https://user-images.githubusercontent.com/87471617/140063943-754d35aa-e806-4234-a5b4-108b2193d962.mp4



## Requirements
- Tensorflow=2.4.0
- CUDA=11.2
- CUDNN=11.4 (necessary only for gpu computing (optional))
- Python 3.6

## Dataset
The Stanford Question Answering Dataset (SQuAD), which is derived from Wikipedia, can be used for question answering chatbot. The SQuAD includes:
- 107,785 question-answer pairs depend on 536 articles.
- Due to a lack of RAM, only 10.000 pairs have been used for training of the Seq2Seq model.
- According to the results, given questions to the model can be predicted by the model accurately.
- Given data to the model should be enhanced in order to increase the accuracy of the model.

## Folders:
### Data: 
- Downloaded data will be saved in this folder. Dowloading script is available in "Preporeccsing file.py". The program will download it in Data folder, if the folder exists.
### Weight:
- Model weight, encoder and decoder model will be saved in this folder.
### Image:
- The history of the model will be saved in this folder.


## Model:
Seq2Seq architecture has been chosen to train the model.

## Running:
- The "main.py" file must be run in a virtual environment with the system requirements. 
