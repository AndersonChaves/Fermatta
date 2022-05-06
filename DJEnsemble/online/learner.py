import ray
from tensorflow.keras.models import model_from_json

@ray.remote
class Learner():
    name = ''
    model = None
    _is_temporal_model = False

    def __init__(self, model_directory, model_name, pis_temporal_model=False):
        self.name = model_name
        json_file = open(model_directory + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(model_directory + model_name + '.h5')
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self._is_temporal_model = pis_temporal_model
        self.model = model

    def invoke(self, X):
        return self.get_model().predict(X)

    def get_model(self):
        return self.model

    def get_shape(self):
        return self.model.input.shape

    def set_model(self, model):
        self.model = model

    def get_name(self):
        return self.name

    def is_temporal_model(self):
        return self._is_temporal_model