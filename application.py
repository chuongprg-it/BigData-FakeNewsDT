from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import torch
# import seaborn as sns
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

from torchtext import data
# from torcheval.metrics import binary_accuracy

app = Flask(__name__)

#
device = torch.device("cpu")
# Pytorch's nn module has lots of useful feature
import torch.nn as nn

class LSTMNet(nn.Module):

    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout):

        super(LSTMNet,self).__init__()

        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        # LSTM layer process the vector sequences
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True
                           )

        # Dense layer to predict
        self.fc = nn.Linear(hidden_dim * 2,output_dim)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()


    def forward(self,text,text_lengths):
        embedded = self.embedding(text).to(device)

        # Thanks to packing, LSTM don't see padding tokens
        # and this makes our model better
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)

        packed_output,(hidden_state,cell_state) = self.lstm(packed_embedded)

        # Concatenating the final forward and backward hidden states
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)

        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.sigmoid(dense_outputs)

        return outputs
    
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("keepitreal/vietnamese-sbert")
vocab_size = tokenizer.vocab_size
embedding_dim = 300
hidden_dim = 64
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.2

model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
model.load_state_dict(torch.load('I:\CODE\PROJECT_COURSE\BIGDATA\FakeNewDT_Vietnamese\model\model_0.9178659543395042.pth', map_location=device))
model = model.to(device)
model.eval()

def input_test(text, tokenizer, model, device):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(device)

    # Get the length of the input sequence
    length = torch.tensor(ids.shape[1], dtype=torch.long).unsqueeze(0)

    # Evaluate the model on the input text
    with torch.no_grad():
        model.eval()
        predictions = model(ids, length)


    binary_predictions = torch.round(predictions).cpu().numpy()

    return binary_predictions[0][0]

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    if request.method == 'POST':
        input_text = request.form['input_text']
        # Thực hiện kiểm tra fakenews ở đây và trả kết quả
        result = input_test(input_text, tokenizer, model, device)
        if result == 1:
            result = "Tin giả"
        else:
            result = "Tin thật"  # Thay đổi kết quả tùy thuộc vào quá trình kiểm tra
        return render_template('index.html', results=[{'input_text': input_text, 'detection_value': result}])
    
@app.route('/streaming')
def streaming():
    random_data = [
        {'STT': 1, 'Sentence': 'This is a random sentence', 'Value Detection': 'True'},
        # Các dòng dữ liệu khác nếu có
    ]
    return render_template('streaming.html', random_data=random_data)
    
if __name__ == '__main__':
    app.run(debug=True)
