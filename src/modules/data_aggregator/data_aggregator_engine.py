from src.modules.core.xml_reader import XMLReader
from src.modules.core.kafka_api import KafkaConnection
from time import sleep
import datetime

class DataAggregatorEngine:
    data_pool = {}
    kafka_api = None

    def __init__(self):
        self.kafka_api = KafkaConnection()

    def start_data_pool(self):
      self.kafka_api.create_topic("mytopic03")
      while(True):
        self.update_data_pool()
        sleep(10)

    def update_data_pool(self):
      self.feed_rain_data_pool()

    def feed_rain_data_pool(self):
      xml_reader = XMLReader()
      xml_reader.read_xml('http://alertario.rio.rj.gov.br/upload/xml/Chuvas.xml')
      rain_data = xml_reader.get_rain_for_station('Rocinha', 'm15')
      record_time = xml_reader.get_record_time()
      read_time = str(datetime.datetime.now())
      self.kafka_api.produce(topic="mytopic03", value=[rain_data, record_time, read_time])

    def get_data_from_size_window(self, stream_identifier, window_size):
        stream = self.data_pool[stream_identifier]['Rocinha'][-window_size:]
        return stream

if __name__ == '__main__':
  data_aggregator = DataAggregatorEngine()
  data_aggregator.start_data_pool()