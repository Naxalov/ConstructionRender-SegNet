import tensorflow as tf
import os
new_model = tf.keras.models.load_model(os.getcwd()+'/model/ConstructionRender.h5')

# Check its architecture

new_model.summary()
