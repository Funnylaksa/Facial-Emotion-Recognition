from keras.models import model_from_json
import numpy as np


import tensorflow
import keras

config = tensorflow.ConfigProto(
device_count={'GPU': 1},
intra_op_parallelism_threads=1,
allow_soft_placement=True
)

config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tensorflow.Session(config=config)
keras.backend.set_session(session)

class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        with session.as_default():
            with session.graph.as_default():
                    self.preds = self.loaded_model.predict(img)
                    return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
