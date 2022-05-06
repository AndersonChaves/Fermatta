from functools import reduce
from .series_generator import SeriesGenerator
from .configuration_manager import ConfigurationManager
from .dataset_manager import DatasetManager
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import pandas as pd
import numpy as np
import ray

class ConvolutionalModelInvoker:
    def __init__(self):
        self.configuration_manager = ConfigurationManager("Q1/query1.config")
        self.dataset_manager = DatasetManager()
        self.temporal_length = int(self.get_parameter_value("temporal_length"))
        self.number_of_samples = int(self.get_parameter_value("number_of_samples"))
        self.offset = int(self.get_parameter_value("offset"))
        self.tiles = pd.read_csv(self.get_parameter_value("tile_specifications_file"))

    def get_parameter_value(self, parameter):
        return self.configuration_manager.get_configuration_value(parameter)

    def invoke_convolutional_model_for_tile(self, tile, data, learner):

        # Get coordinate limits for every tile i
        ti = self.tiles[self.tiles.name == 'Tile' + str(tile + 1)].iloc[0][1:5].values
        lat1, lat2, lon1, lon2 = int(ti[0]), int(ti[1]), int(ti[2]), int(ti[3])

        tile_frame_series = SeriesGenerator().tile_frame_series_generator(data[self.offset:, lat1:lat2 + 1, lon1:lon2 + 1],
                                                                               self.temporal_length)
        # FOR EVERY FRAME SERIES IN TILE
        rmse_by_frame = []
        for s, (frame_series_input, frame_series_output) in enumerate(tile_frame_series):
            #print("Evaluating error for frame series", s + 1, " of tile ", tile + 1)
            if s == self.number_of_samples:
                break

            # gets a prediction from a model ensemble, the average pred of different models
            predicted_frame_series = self.averaging_ensemble(frame_series_input, [learner])

            # computes the rmse for the frame.
            last_output_frame    = frame_series_output[:,-1:,:,:]
            last_predicted_frame = predicted_frame_series[:, -1, :, :, :]
            last_output_frame = last_output_frame.reshape(last_predicted_frame.shape)
            loss = tf.sqrt(tf.math.reduce_mean(tf.losses.mean_squared_error(last_output_frame, last_predicted_frame)))
            #print("RMSE Calculated for series ", s + 1, " of tile ", tile + 1, ": ", loss)
            #print("Expected: Shape-", last_output_frame.shape, "Data:", last_output_frame)
            #print("Predicted: Shape-", last_predicted_frame.shape, "Data:", last_predicted_frame.numpy())
            rmse_by_frame.append(loss.numpy())
        average_rmse = reduce(lambda a, b: a + b, rmse_by_frame) / len(rmse_by_frame)
        return average_rmse

    def averaging_ensemble(self, frame_series, learner_list, weights=None):
        if weights is None:
            weights = [1 for i in range(len(learner_list))]

        y_m = []
        length_x = frame_series.shape
        for i, learner in enumerate(learner_list):
            result_y, length_x = self.invoke_candidate_model(learner, frame_series)
            y_m.append(result_y)

        sum = tf.zeros_like(length_x)

        # The average prediction of different models
        # Isolate this part to adapt to model stacking
        for i, y in enumerate(y_m):
            sum = sum + y * weights[i]
        total = reduce(lambda a, b: a + b, weights) # sum all weights
        return sum / total

    def invoke_candidate_model(self, learner, query):
        #model = ray.get(learner.get_model.remote())
        shape = ray.get(learner.get_shape.remote())
        # number of iterations in x and y axis
        x_size, y_size = int(shape[2]), int(shape[3])  # Size of the models input frame

        # Duplicates the dataset when necessary to fit the models input
        temp_query = self.dataset_manager.resizeDataset(x_size, y_size, query)

        # How many times does the model fit on that dimension: query_dim / model_dim
        length_x = int(temp_query.shape[2] / int(shape[2]))
        length_y = int(temp_query.shape[3] / int(shape[3]))

        yp = np.zeros((temp_query.shape))
        x = query.shape
        list = []
        for x in range(length_x):
            for y in range(length_y):
                # Get the corresponding query data and predict
                lat_i, lat_e = x * int(shape[2]), (x + 1) * int(shape[2])
                lon_i, lon_e = y * int(shape[3]), (y + 1) * int(shape[3])
                prediction = learner.invoke.remote(temp_query[:, :, lat_i:lat_e, lon_i:lon_e, :])
                list.append((lat_i, lat_e, lon_i, lon_e, prediction))

        for e in list:
            # Get the corresponding query data and predict
            yp[:, :, e[0]:e[1], e[2]:e[3], :] = ray.get(e[4])

        output_frame_series = yp[:, :, :query.shape[2], :query.shape[3], :]
        #output_frame        = output_frame_series[:,9, :, :, :]
        # Cuts the predicted section corresponding to the original query only
        return output_frame_series, x