from time import sleep
from kafka import TopicPartition
from src.modules.core.kafka_api import KafkaConnection
from src.model_base.simple_regressor import LinearRegressionStrategy


class ContinuousQueryProcessorEngine:
    kafka_connection = None

    def __init__(self):
      self.kafka_connection = KafkaConnection()

    #QUERY REPOSITORY
    # time always in minutes
    def predict_rain_for_next_hours(self, time_window_size_in_minutes):
        print("Starting predictive query")
        while(True):
            self.execute_predictive_query()
            sleep(3)

    def execute_predictive_query(self):
        topic_name = "mytopic03"
        consumer = self.kafka_connection.get_consumer(topic_name)
        topic_partition = TopicPartition('mytopic03', 0)
        consumer.assign([topic_partition])
        consumer.seek_to_end(topic_partition)
        position = consumer.position(topic_partition)
        consumer.seek(topic_partition, max(0, position - 4))

        rain_list = []
        for message in consumer:
            rain_list.append(tuple(message.value[0:2]))
            print("Rain list: ")
            print(rain_list)
            # if len(rain_list) >= 2 and \ # Verifies if current reading is or not new
            #  rain_list[-1] == rain_list[-2]:
            #  rain_list.pop(-1)
            if len(rain_list) >= 4:
                X = [x[0] for x in rain_list[:4]]
                y = self.predict_rain(X)
                print(y)
                rain_list.pop(0)


    def predict_rain(self, X):
        model = LinearRegressionStrategy([0, 1, 2, 3], X)
        model.train()
        print(model.predict([4, 5, 6, 7]))

if __name__ == "__main__":
    cqp = ContinuousQueryProcessorEngine()
    cqp.predict_rain_for_next_hours(3)