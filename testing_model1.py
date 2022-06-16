import cv2
import numpy as np
import tensorflow as tf
img_height = 180
img_width = 180
model = tf.keras.models.load_model("fruits_vegetable.h5") #loading the save model
class_names=['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
             'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
             'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika',
             'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach',
             'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
 #list of the classes

fruit_path = "fruits/test/banana/Image_1.jpg"  #image path

#convert the image into array to match into the tensor
img = tf.keras.utils.load_img(
    fruit_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch


print(img_array)
predictions = model.predict(img_array)#predict #array
score = tf.nn.softmax(predictions[0])#Computes softmax activations.


print(np.argmax(score))#return the index
print(class_names[np.argmax(score)]) #return class name prediction
print(np.max(score)) #return confidence score

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

