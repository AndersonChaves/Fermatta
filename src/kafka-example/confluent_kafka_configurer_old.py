from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import Consumer
from confluent_kafka import KafkaException, KafkaError

# from confluent_kafka import avro
# def unpack(string, schema):
#   reader = DatumReader(schema)
#   for position in range(0,11):
#       try:
#           decoder = BinaryDecoder(io.BytesIO(string[position:]))
#           decoded_msg = reader.read(decoder)
#           return decoded_msg
#       except AssertionError:
#           continue
#   raise Exception('Msg cannot be decoded. MSG: {}.'.format(string))

def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg), str(err)))
    else:
        print("Message produced: %s" % (str(msg)))

def produce(topic_name, key, message):
    conf = {'bootstrap.servers': "localhost:9092, localhost:9092",
            'client.id': 'localhost'}
    producer = Producer(conf)
    producer.produce(topic=topic_name, key=key, value=message, callback=acked)
    producer.flush()

def create_topic(topic_name):
    admin_client = AdminClient({
        "bootstrap.servers": "localhost:9092"
    })

    topic_list = []
    topic_list.append(NewTopic(topic=topic_name, num_partitions=1, replication_factor=1))
    admin_client.create_topics(topic_list)

def consume():
    conf = {'bootstrap.servers': "localhost:9092",
            'group.id': "3",
            'auto.offset.reset': 'smallest',
            'enable.auto.commit': False}
    #conf = {'bootstrap.servers': "localhost:9092, localhost:9092",
    #        'group.id': "1",
    #        'auto.offset.reset': 'smallest'}
    consumer = Consumer(conf)
    consumer.subscribe(["mytopic02"])

    raw_msg = consumer.poll(timeout=5.0)
    msg = raw_msg
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            # End of partition event
            print('%% %s [%d] reached end at offset %d\n' %(msg.topic(), msg.partition(), msg.offset()))
        elif msg.error():
            raise KafkaException(msg.error())

    #msg = unpack(raw_msg.value(), value_schema)

    print(raw_msg.value())

    # consumer.consume(3)
    # for message in consumer:
    #     # message value and key are raw bytes -- decode if necessary!
    #     # e.g., for unicode: `message.value.decode('utf-8')`
    #     print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
    #                                          message.offset, message.key,
    #                                          message.value))

# create_topic("mytopic02")
produce("mytopic02", "20", "23")
# consume()