import math
import numpy as np
import h5py

class DatasetManager:
    def resizeDataset(self, x, y, data):
        while not (data.shape[2] % x == 0 and data.shape[3] % y == 0):
            if data.shape[2] % x != 0:
                # The size of the dimension that must be extended
                #  so that the final size is multiple of the models input.
                # Ex. mx=3, tx = 10: ceil(10/3)*3-10=4*3-10=2
                ext_x = abs(math.ceil(data.shape[2] / x) * x - data.shape[2])

                # Duplicates the last column of the data to fit the models input size
                # Ex. x y ===> x y y
                #     z w      z w w
                data = np.concatenate((data, data[:, :, -ext_x:, :, :]), axis=2)

            if data.shape[3] % y != 0:
                ext_y = abs(math.ceil(data.shape[3] / y) * y - data.shape[3])
                data = np.concatenate((data, data[:, :, :, -ext_y:, :]), axis=3)
        return data

    def loadDataset(self, dataPath):
        if dataPath == 'RainData.h5':
            return self.loadRainDataset(dataPath)
        else:
            return self.loadTemperatureDataset(dataPath)

    def loadRainDataset(self, dataPath):
        with h5py.File(dataPath) as f:
            dataset = f['data'][...][:, 20, :, :]
            # dataset = dataset.transpose((0, 2, 1))
        return dataset

    def loadTemperatureDataset(self, dataPath):
        with h5py.File(dataPath) as f:
            dataset = f['real'][...]
        return dataset