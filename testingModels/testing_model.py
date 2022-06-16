import cv2
import numpy as np
import tensorflow as tf
img_height = 180
img_width = 180
model = tf.keras.models.load_model("flower.h5") #loading the save model
class_names=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'] #list of the classes

flower_path = "../flowers/Testing/dandelion.png"  #image path

#convert the image into array to match into the tensor
img = tf.keras.utils.load_img(
    flower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)#predict #array
score = tf.nn.softmax(predictions[0])#Computes softmax activations.

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

