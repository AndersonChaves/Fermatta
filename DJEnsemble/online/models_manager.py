from learner import Learner
from tensorflow.keras.models import model_from_json
import fnmatch
import ray
import os

class ModelsManager():
    temporal_models      = []
    convolutional_models = []

    def load_models(self):
        print("Loading temporal models")
        self.temporal_models      = self.load_all_temporal_models()
        print("Loading convolutional models")
        self.convolutional_models = self.load_all_convolutional_models() 
        
    def load_all_temporal_models(self):
        model_list = []
        for file in os.listdir(self.temporal_models_directory):
            if file.endswith(".h5"):
                model_name = file[:-3]
                model_list.append(self.load_model(model_name))
        return model_list

    def load_all_convolutional_models(self):
        models_names = self.get_names_of_models_in_dir(self.convolutional_models_directory)
        models_list = [self.load_model(model) for model in models_names]
        return models_list

    def load_model(self, model_name):
        if model_name.startswith('best_model'):
            return self.load_temporal_model(model_name)
        else:
            return self.load_convolutional_model(model_name)

    def load_temporal_model(self, model_name):
        return Learner.remote(self.temporal_models_directory, model_name, pis_temporal_model=True)

    def load_convolutional_model(self, model_name):
        return Learner.remote(self.convolutional_models_directory, model_name, pis_temporal_model=False)

    def include_convolutional_models_from_directory(self, models_path):
        self.convolutional_models_directory = models_path

    def include_temporal_models_from_directory(self, models_path):
        self.temporal_models_directory = models_path

    def get_names_of_models_in_dir(self, models_path):
        models = fnmatch.filter(os.listdir(models_path), '*.h5')
        for i, m in enumerate(models):
            models[i] = m.split('.')[0]
        return models

    def get_latitude_input_size(self, model_name):
        model = self.get_model_from_name(model_name)
        if model.is_temporal_model:
            return 1
        else:
            return int(model.get_model().input.shape[2])

    def get_longitude_input_size(self, model_name):
        model = self.get_model_from_name(model_name)
        if model.is_temporal_model:
            return 1
        else:
            return int(model.get_model().input.shape[3])

    def get_model_from_name(self, model_name):
        all_learners = self.temporal_models + self.convolutional_models
        if all_learners == []:
            print("No models loaded. Inform models directory and run function load_models")
            raise(Exception)
        for learner in all_learners:
            #learner = ray.get(learner_id)
            name = ray.get(learner.get_name.remote())
            if name == model_name:
                return learner
        print("Model not found")
        raise(Exception)

    def get_models_from_list_of_names(self, best_allocation):
        models_list = []
        for model_name in best_allocation:
            models_list.append(self.get_model_from_name(model_name))
        return models_list

    def get_models(self):
        return self.temporal_models + self.convolutional_models
