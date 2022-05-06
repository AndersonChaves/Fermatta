from time import sleep
from kafka import KafkaProducer
from json import dumps

producer = KafkaProducer(
    value_serializer=lambda m: dumps(m).encode(),
    bootstrap_servers=['localhost']
)

for i in range(3):
    producer.send('quickstart-events', value='009')
    sleep(0.001)