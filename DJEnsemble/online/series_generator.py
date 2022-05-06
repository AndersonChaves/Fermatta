import numpy as np

class SeriesGenerator:
    # Generate series of specified length for convolutional models
    def tile_frame_series_generator(self, data, temporal_length):
        for i in range(data.shape[0] - temporal_length):
            X = np.zeros((1, temporal_length, data.shape[1], data.shape[2], 1), dtype=float)
            Y = np.zeros((1, temporal_length, data.shape[1], data.shape[2], 1), dtype=float)
            X[0, :, :, :, 0] = data[i:i + temporal_length, :, :]
            Y[0, :, :, :, 0] = data[i + 1:i + temporal_length + 1, :, :]
            yield X, Y

    # Former split_sequence function
    def split_series_into_set_of_fixed_size_series(self, sequence, n_steps_in, n_steps_out):
        # X is the series inputs, y is the expected outputs for each series
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequence):
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)