import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

class KeyPointClassifier:
    def __init__(self, model_path='/home/spring/hand/hand-gesture-recognition-using-mediapipe/model/keypoint_classifier/keypoint_classifier.hdf5'):
        self.model = tf.keras.models.load_model(model_path)

    def __call__(self, landmark_list):
      x = np.expand_dims(landmark_list, axis=0)

      pred = self.model.predict(x, verbose=0)

      class_id = np.argmax(pred)
      confidence = np.max(pred)

      if confidence < 0.8:
        return 8

      return class_id
