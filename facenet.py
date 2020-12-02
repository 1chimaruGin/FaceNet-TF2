import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3
from tensorflow.keras import layers, optimizers, metrics, Sequential, Model


base_model = {
  'InceptionResNetV2': InceptionResNetV2,
  'Resnet50': ResNet50,
  'InceptionV3': InceptionV3
}

class FaceNet():
  def __init__(self, config):
    super(FaceNet, self).__init__()
    self.model = self.create_model(config)

  def create_model(self, config):
    base_network = base_model[config.backbone](
      input_shape=(config.input_size, config.input_size, 3),
      include_top=False,
      weights='imagenet'
    )
    base_network.trainable = True
    inputs = base_network.input
    x = base_network(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Dense(config.embedding_size, name='embedding')(x)
    model = Model(inputs=inputs, outputs=embedding)
    model.summary()
    return model
