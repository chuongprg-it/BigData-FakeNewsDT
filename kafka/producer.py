from kafka import KafkaProducer
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from json import dumps
from time import sleep
from random import seed
from random import randint

topic_name = 'test4'
kafka_server = 'localhost:9092'

producer = KafkaProducer(bootstrap_servers=kafka_server,value_serializer = lambda x:dumps(x).encode('utf-8'))


seed(1)
data = pd.read_csv('Test.csv')
for i in range(1000):
    for index, row in data.iterrows():
        # Convert the row to a dictionary and then to JSON
        json_data = row.to_dict()
        producer.send(topic_name, value=json_data)
        # print(str(json_data) + " sent")

producer.flush()