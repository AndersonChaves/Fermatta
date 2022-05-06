import multiprocessing
from time import sleep
from src.modules.data_aggregator.data_aggregator_engine import DataAggregatorEngine
from src.modules.continuous_query_processor.continuous_query_processor_engine import ContinuousQueryProcessorEngine


if __name__ == '__main__':
  data_aggregator_engine = DataAggregatorEngine()
  continuous_query_processor_engine = ContinuousQueryProcessorEngine()

  data_aggregator_process = multiprocessing.Process(target=data_aggregator_engine.start_data_pool, args=())
  #continuous_query_processor_process = multiprocessing.Process(
  #  target=continuous_query_processor_engine.predict_rain_for_next_hours, args=(3,))

  #continuous_query_processor_engine.predict_rain_for_next_hours(3)

  data_aggregator_process.start()
  # sleep(3)
  #continuous_query_processor_process.start()


  data_aggregator_process.join()
  continuous_query_processor_process.join()
