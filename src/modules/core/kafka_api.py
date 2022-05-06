from time import sleep
from kafka import KafkaProducer
from json import dumps, loads
from kafka import KafkaConsumer, TopicPartition

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import KafkaException, KafkaError

class KafkaConnection:
    producer = None

    def __init__(self):
        self.producer = KafkaProducer(
            key_serializer=lambda m: dumps(m).encode(),
            value_serializer=lambda m: dumps(m).encode(),
            bootstrap_servers=['localhost'],
        )

    def get_end_offsets(self, consumer, topic) -> dict:
        print("Getting partitions: ")
        partitions_for_topic = consumer.partitions_for_topic(topic)
        print(partitions_for_topic)
        if partitions_for_topic:
            partitions = []
            for partition in consumer.partitions_for_topic(topic):
                partitions.append(TopicPartition(topic, partition))
            # https://kafka-python.readthedocs.io/en/master/apidoc/KafkaConsumer.html#kafka.KafkaConsumer.end_offsets
            # Get the last offset for the given partitions. The last offset of a partition is the offset of the upcoming message, i.e. the offset of the last available message + 1.
            end_offsets = consumer.end_offsets(partitions)
            return end_offsets

    def delete_topic(self, topic_name):
        admin_client = AdminClient({
            "bootstrap.servers": "localhost:9092"
        })
        fs = admin_client.delete_topics([topic_name], operation_timeout=1)
        # Wait for operation to finish.
        for topic, f in fs.items():
            try:
                f.result()  # The result itself is None
                print("Topic {} deleted".format(topic))
            except Exception as e:
                print("Failed to delete topic {}: {}".format(topic, e))
        sleep(5)

    # confluent_kafka based create topic
    def create_topic(self, topic_name):
        self.delete_topic(topic_name)
        admin_client = AdminClient({
            "bootstrap.servers": "localhost:9092"
        })
        new_topic = NewTopic(topic=topic_name, num_partitions=1, replication_factor=1)
        admin_client.create_topics([new_topic])
        sleep(1)


    def produce(self, topic, value, key=None):
        print("Producing value for topic")
        if key != None:
            self.producer.send(topic, value=value, key=key)
        else:
            self.producer.send(topic, value=value, key=0)
        sleep(3)

    def get_consumer(self, topic_name = None):
        consumer = KafkaConsumer(
            # topic_name,
            bootstrap_servers=["localhost"],
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="1",
            value_deserializer=lambda m: loads(m.decode('ascii'))
        )
        return consumer
        # for message in consumer:
        #     print(message)
        #     print(message.value)


# k = KafkaAPI()
# k.delete_topic('quickstart-events')
# k.create_topic('quickstart-events')
# k.produce('quickstart-events', '004')

if __name__ == "__main__":
    kafka_connection = KafkaConnection()
    topic_name = "mytopic03"
    consumer = kafka_connection.get_consumer(topic_name)
    end_offsets = kafka_connection.get_end_offsets(consumer, topic_name)
    while not end_offsets:
        print("Offset not found")
        sleep(3)
        end_offsets = kafka_connection.get_end_offsets(consumer, topic_name)
    consumer.assign([*end_offsets])
    for key_partition, value_end_offset in end_offsets.items():
        new_calculated_offset = value_end_offset - 4
        new_offset = new_calculated_offset if new_calculated_offset >= 0 else 0
        while(True):
            consumer.seek(key_partition, new_offset)
            msg = consumer.poll(timeout_ms=1000.0)
            if msg is None:
                print("Nothing here")
                continue
            else:
                break

    rain_list = []
    for message in consumer:
        rain_list.append(tuple(message.value[0:2]))
        print("Rain list: ")
        print(rain_list)