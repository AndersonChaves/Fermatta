# %tensorflow_version 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

#[RAY] Import Ray
import ray

from functools import reduce
import itertools
import pandas as pd
import numpy as np
from dataset_manager import DatasetManager
from models_manager import ModelsManager
from series_generator import SeriesGenerator
from convolutional_model_invoker import ConvolutionalModelInvoker
import time

import copy

MAXIMUM_COST = 999999

def normalize_data(model_tiles):
    # Normalize execution rmse
    tile_identifier_list = ["Tile" + str(i+1) for i in range(model_tiles.shape[1] - 1)]
    normalized_model_tiles = copy.deepcopy(model_tiles)
    max_value_rmse = normalized_model_tiles[tile_identifier_list].max().max()
    normalized_model_tiles[tile_identifier_list] = normalized_model_tiles[tile_identifier_list].div(max_value_rmse)
    return normalized_model_tiles

class DJEnsemble:
    def __init__(self, configuration_manager, time_weight):
        # [RAY] Start Ray.
        ray.init()

        self.configuration_manager = configuration_manager

        self.dataPath                  =             self.get_parameter_value("query")
        self.convolutional_models_path =             self.get_parameter_value("convolutional_models_path") + "/"
        self.temporal_length           =         int(self.get_parameter_value("temporal_length"))
        self.threshold                 =       float(self.get_parameter_value("threshold"))
        self.temporal_models_path      =             self.get_parameter_value("temporal_models_path") + "/"
        self.region                    =        eval(self.get_parameter_value("max_tile_area"))
        self.min_tile_length           =         int(self.get_parameter_value("min_tile_length"))
        self.number_of_samples         =         int(self.get_parameter_value("number_of_samples"))
        self.offset                    =         int(self.get_parameter_value("offset"))
        self.error_weight              =   1 - time_weight
        self.time_weight               =       time_weight
        self.tiles                     = pd.read_csv(self.get_parameter_value("tile_specifications_file"))
        self.model_tiles_rmse          = pd.read_csv(self.get_parameter_value("model_estimated_errors_by_tile_file"))
        self.model_tiles_time          = pd.read_csv(self.get_parameter_value("model_estimated_time_by_tile_file"))
        self.output_file_name          = self.get_parameter_value("output_file_name")

        self.multiply_number_of_executions = self.get_parameter_value("multiply_number_of_executions") == 'S'
        self.normalize_values = self.get_parameter_value("normalize_values") == 'S'
        self.dataset_manager = DatasetManager()
        self.models_manager  = ModelsManager()
        self.models_manager.include_convolutional_models_from_directory(self.convolutional_models_path)
        self.models_manager.include_temporal_models_from_directory(self.temporal_models_path)
        self.models_manager.load_models()

    def get_parameter_value(self, parameter):
        return self.configuration_manager.get_configuration_value(parameter)

    def ensemble(self):
        self.start = time.time()
        result_by_tile = []
        # --------------------------------- READ DATASET ---------------------------------
        data            = self.dataset_manager.loadDataset(self.dataPath)

        # -------------------------- CALCULATE ALLOCATION COSTS --------------------------
        best_allocation, best_cost_by_tile, best_cost = self.get_lower_cost_combination()
        print('Best Allocation: ', best_allocation)
        print('Best Estimated Cost: ', best_cost)

        # --------------------------------- GET MODELS ---------------------------------
        print("Loading models...")
        learner_list = self.models_manager.get_models_from_list_of_names(best_allocation)

        # --------------------------------- EVALUATE ERROR ----------------------------
        print("Evaluating errors...")
        for tile in range(self.tiles.shape[0]):
            print("==>Evaluating error for tile ", tile + 1, " of ", self.tiles.shape[0])
            print("Model: ", ray.get(learner_list[tile].get_name.remote()))
            start = time.time()
            average_error = self.parallel_calculate_error_for_tile(tile, data, learner_list[tile])

            print("Total time for tile evaluation: ", round(time.time() - start, 2), " seconds")
            #print("Average tile error: ", average_error, "\n")
            result_by_tile.append(average_error)

        # Returns the total average rmses and the tiles rmses
        #total_rmse = sum(result_by_tile) / self.tiles.shape[0]
        # [RAY] get Results
        results = result_by_tile
        total_rmse = sum(results) / self.tiles.shape[0]

        self.write_results(self.output_file_name, best_allocation, best_cost_by_tile, best_cost, result_by_tile, total_rmse)
        return total_rmse, results # [RAY] result_by_tile

    def write_results(self, output_file_name, best_allocation, best_cost_by_tile, best_cost, result_by_tile, total_rmse):
        end = time.time()
        file = open(output_file_name + "-" + str(self.time_weight) + '.out', 'w')
        file.write("Best Allocation: \n")
        for model in best_allocation:
            file.write(model + '\n')
        file.write("Best cost by tile: \n")
        for cost in best_cost_by_tile:
            file.write(str(cost).replace('.', ',') + '\n')
        file.write("Best Cost: " + str(best_cost) + '\n')
        file.write("Total RMSE: " + str(total_rmse) + '\n')
        file.write("Total time: " + str(end - self.start) + '\n')
        file.write("Results by tile: \n")
        for result in result_by_tile:
            file.write(str(result).replace('.', ',') + '\n')

    #def calculate_error_for_tile(self, tile, data, learner):
    #    if learner.is_temporal_model:  # For LSTM models
    #        return self.invoke_temporal_model_for_tile(tile, data, learner)
    #    return self.invoke_convolutional_model_for_tile(tile, data, learner) # For Convolutional models

    def parallel_calculate_error_for_tile(self, tile, data, learner):
        if ray.get(learner.is_temporal_model.remote()):  # For LSTM models
            return self.invoke_temporal_model_for_tile(tile, data, learner)
        convolutional_model_invoker = ConvolutionalModelInvoker()
        error = convolutional_model_invoker.invoke_convolutional_model_for_tile(tile, data, learner)
        return error # For Convolutional models

    def invoke_temporal_model_for_tile(self, tile, data, learner):
        ti = self.tiles[self.tiles.name == 'Tile' + str(tile + 1)].iloc[0][1:5].values
        lat1, lat2, lon1, lon2 = int(ti[0]), int(ti[1]), int(ti[2]), int(ti[3])
        rmse_by_region = []
        for i in range(lat1 - 1, lat2):
            for j in range(lon1 - 1, lon2):
                cut_start  = self.offset
                cut_ending = self.offset + self.number_of_samples + self.temporal_length
                X, y = SeriesGenerator().split_series_into_set_of_fixed_size_series(data[cut_start:cut_ending, i, j],
                                                                       self.temporal_length, 1)
                X = X.reshape((self.number_of_samples, self.temporal_length, 1))
                output = learner.invoke.remote(X)
                region_rmse = tf.sqrt(tf.math.reduce_mean(tf.losses.mean_squared_error(ray.get(output), y)))
                rmse_by_region.append(region_rmse.numpy())
        average_rmse = reduce(lambda a, b: a + b, rmse_by_region) / len(rmse_by_region)
        return average_rmse

    def get_coringa_for_tile(self, i):
        group = self.tiles['Grupo'][i]
        coringa_name = 'best_model_C' + str(group)
        return coringa_name

    def get_w(self):
        # W means how much each tile represents of the query
        W = []
        area_query = (self.region[1] - self.region[0]) * (self.region[3] - self.region[2])

        for j in range(self.tiles.shape[0]):
            area_tile = (self.tiles.iloc[j].X_max - self.tiles.iloc[j].X_min) * (
                        self.tiles.iloc[j].Y_max - self.tiles.iloc[j].Y_min)
            W.append(area_tile / area_query)
        return W

    def estimate_cost_for_allocation(self, option, models_names_list, normalized_model_tiles, W):
        cost = 0.0
        for i, model_name in enumerate(option):
            if model_name in models_names_list:
                cost = cost + W[i] * (self.estimate_cost_for_model_in_tile(model_name, "Tile" + str(i+1),
                                                                           normalized_model_tiles, W))
        return cost, option

    def estimate_cost_for_model_in_tile(self, model_name, tile_name, normalized_model_tiles, W):
        selected_model_tiles_rmse = self.normalized_model_tiles_rmse[normalized_model_tiles.models == model_name]
        selected_model_tiles_time = self.normalized_model_tiles_time[normalized_model_tiles.models == model_name]
        normalized_estimated_error = selected_model_tiles_rmse[tile_name].iloc[0]
        executions = 1
        if self.multiply_number_of_executions:
            executions = self.calculate_number_of_executions(model_name, tile_name)
        estimated_time = selected_model_tiles_time[tile_name].iloc[0] * executions

        return ((self.error_weight * normalized_estimated_error + self.time_weight * estimated_time))

    def calculate_number_of_executions(self, model_name, tile_name):
        #Get tile dimensions
        ti = self.tiles[self.tiles.name == tile_name].iloc[0][1:5].values
        lat1, lat2, lon1, lon2 = int(ti[0]), int(ti[1]), int(ti[2]), int(ti[3])
        tile_len_x = lat2 - lat1 + 1
        tile_len_y = lon2 - lon1 + 1

        # Get model dimensions
        model_lat_size  = self.models_manager.get_latitude_input_size(model_name)
        model_long_size = self.models_manager.get_longitude_input_size(model_name)

        length_lat  = int(tile_len_x // model_lat_size) + (tile_len_x % model_lat_size)
        length_long = int(tile_len_y // model_long_size) + (tile_len_x % model_long_size)

        return length_lat * length_long

    def get_lower_cost_combination(self):
        models_names_list = self.get_names_of_available_models()
        # Montar alocação de menor custo
        tile_set = range(len(self.tiles))
        best_allocation = []
        best_cost_list = []
        best_cost = 0
        for i in tile_set:
            lower_cost, lower_cost_model = self.get_lower_cost_model_for_tile("Tile" + str(i+1), models_names_list)
            best_cost += lower_cost
            best_allocation.append(lower_cost_model)
            best_cost_list.append(lower_cost)
        return (best_allocation, best_cost_list, best_cost)

    def get_lower_cost_model_for_tile(self, tile, model_names):
        print("Analyzing best model for tile " + tile + ':')
        W = self.get_w()
        lower_cost = MAXIMUM_COST
        best_model = ''
        for model in model_names:
            self.normalized_model_tiles_rmse = self.model_tiles_rmse
            self.normalized_model_tiles_time = self.model_tiles_time
            if self.normalize_values:
                self.normalized_model_tiles_rmse = normalize_data(self.model_tiles_rmse)
                self.normalized_model_tiles_time = normalize_data(self.model_tiles_time)
            model_cost = self.estimate_cost_for_model_in_tile(model, tile, self.normalized_model_tiles_rmse, W)
            if model_cost < lower_cost and model_cost != -10:
                lower_cost, best_model = model_cost, model
        if best_model == '':
            raise Exception("No model found ")
        print(best_model + ": " + str(lower_cost))
        return lower_cost, best_model

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

    def select_tiles_by_max_length(self, min_tile_length, tile_set):
        filtered_tiles = []
        for tile in tile_set:#AKI Tile names
            if (self.get_tile_x_length(tile) < min_tile_length) or \
                (self.get_tile_y_length(tile) < min_tile_length):
                filtered_tiles.append(tile)
            else:
                print("Tile min lenth:", tile)
        return filtered_tiles

    def get_tile_x_length(self, i):
        return (self.tiles.iloc[i].X_max - self.tiles.iloc[i].X_min) + 1

    def get_tile_y_length(self, i):
        return (self.tiles.iloc[i].Y_max - self.tiles.iloc[i].Y_min) + 1

    def invoke_candidate_model(self, learner, query):
        model = learner.model
        # number of iterations in x and y axis
        x_size, y_size = int(model.input.shape[2]), int(model.input.shape[3])  # Size of the models input frame

        # Duplicates the dataset when necessary to fit the models input
        temp_query = self.dataset_manager.resizeDataset(x_size, y_size, query)

        # How many times does the model fit on that dimension: query_dim / model_dim
        length_x = int(temp_query.shape[2] / int(model.input.shape[2]))
        length_y = int(temp_query.shape[3] / int(model.input.shape[3]))

        yp = np.zeros((temp_query.shape))
        x = query.shape
        for x in range(length_x):
            for y in range(length_y):
                # Get the corresponding query data and predict

                lat_i, lat_e = x * int(model.input.shape[2]), (x + 1) * int(model.input.shape[2])
                lon_i, lon_e = y * int(model.input.shape[3]), (y + 1) * int(model.input.shape[3])
                prediction = model.predict(temp_query[:, :, lat_i:lat_e, lon_i:lon_e, :])
                yp[:, :, lat_i:lat_e, lon_i:lon_e, :] = prediction
        output_frame_series = yp[:, :, :query.shape[2], :query.shape[3], :]
        #output_frame        = output_frame_series[:,9, :, :, :]
        # Cuts the predicted section corresponding to the original query only
        return output_frame_series, x

    def invoke_coringa_model(self, learner, frame_series):
        length_x = frame_series.shape[2]
        length_y = frame_series.shape[3]

        yp = np.zeros((frame_series.shape))
        x = frame_series.shape
        for x in range(length_x):
            for y in range(length_y):
                # Get the corresponding query data and predict
                lat_i, lat_e = x , x + 1
                lon_i, lon_e = y , y + 1

                input = frame_series[:, :, lat_i:lat_e, lon_i:lon_e, :]
                input = np.reshape(input, (1, self.temporal_length, 1))

                print("Invoking model ", learner.get_name())
                print("Input: ", input)
                yp[:, :, lat_i:lat_e, lon_i:lon_e, :] = learner.get_model().predict(input)
                print("Output: ", yp[:, :, lat_i:lat_e, lon_i:lon_e, :])

        # Cuts the predicted section corresponding to the original query only
        return yp[:, :, :frame_series.shape[2], :frame_series.shape[3], :], x

    def get_names_of_available_models(self):
        names_list = self.models_manager.get_names_of_models_in_dir(self.convolutional_models_path)
        names_list = names_list + self.models_manager.get_names_of_models_in_dir(self.temporal_models_path)
        return names_list