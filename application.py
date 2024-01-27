from flask import Flask, render_template, request
from flask_socketio import SocketIO
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
socketio = SocketIO(app)

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

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
vocab_size = tokenizer.vocab_size
embedding_dim = 300
hidden_dim = 64
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.2

model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
model.load_state_dict(torch.load('./model/model_0.9178659543395042.pth', map_location=device))
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



import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession 

import numpy as np
import transformers
import pandas as pd
# import matplotlib.pyplot as plt
import torch
# import seaborn as sns
from pyspark.sql.types import ArrayType, IntegerType

from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
# import matplotlib.pyplot as plt
import time
from pyspark.sql.functions import *
# import seaborn as sns
import statsmodels.api as sm
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext

from pyspark.ml.feature import StringIndexer, VectorIndexer, StringIndexerModel, IndexToString
from pyspark.sql import SparkSession

scala_version = '2.12'  # your scala version
spark_version = '3.5.0' # your spark version
packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:3.6.0' #your kafka version
]
spark = SparkSession.builder.master("local").appName("kafka-example").config("spark.jars.packages", ",".join(packages)).getOrCreate()

sqlContext = spark.sparkContext

from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, FloatType

# Define the schema for your DataFrame
schema = StructType([
    StructField("post_message", StringType(), True),
])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
# Specify the Kafka parameters
kafka_params = {
    "kafka.bootstrap.servers": "localhost:9092",  # Change this to your Kafka broker
    "subscribe": "FakeNewDT",               # Change this to your Kafka topic
    "startingOffsets": "earliest"
}

stream_model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
device = torch.device("cpu")
stream_model = stream_model.to(device)
stream_model.load_state_dict(torch.load('./model/model_0.9178659543395042.pth',map_location=torch.device('cpu')))
stream_model.eval()

def stream_predict(text, tokenizer, model, device):
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
        return render_template('index.html', result=result, input_text=input_text)
    
@app.route('/streaming')
def streaming():
    kafka_stream_df = (
    spark.read.format("kafka")
    .option("kafka.bootstrap.servers", kafka_params["kafka.bootstrap.servers"])
    .option("subscribe", kafka_params["subscribe"])
    .option("startingOffsets", kafka_params["startingOffsets"])
    .load()
    .selectExpr("CAST(value AS STRING)")
    .select(from_json("value", schema).alias("data"))
    .select("data.*")
    )
# Apply the UDF to create a new column 'predictions'
    data = kafka_stream_df.toPandas()['post_message']
    data = data.head(100)
    import pandas as pd
    from time import sleep
    from IPython.display import display, clear_output

    # Tạo DataFrame trống
    df = pd.DataFrame(columns=['post_message', 'predictions'])
    sent = []
    pred = []
    for i in range(len(data)):
        result = stream_predict(data[i], tokenizer, stream_model, device)
        # df = df.append({'post_message': data[i], 'predictions': result}, ignore_index=True)
        sent.append(data[i])
        pred.append(result)
        render_template(
            'streaming.html',
            sent=sent,
            pred=pred
        )
        # print(sent)
        print(pred)
        # sleep(3)
        # socketio.emit('update', {'sent': sent, 'pred': pred})
        # socketio.sleep(1)
        clear_output(wait=True)
    return render_template('streaming.html', sent=sent, pred=pred)


if __name__ == '__main__':
    app.run(debug=True)
