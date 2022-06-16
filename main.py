from tkinter import *
from tkinter import filedialog
import os
from PIL import ImageTk, Image
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


def browse():
    global filename
    filename=filedialog.askopenfilename(initialdir=os.getcwd(), title="Select fruit Image", filetypes=(("JPG File","*.jpg"),("PNG file","*.png")))
    fruit_path = filename  # image path

    # convert the image into array to match into the tensor
    img = tf.keras.utils.load_img(
        fruit_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)  # predict #array
    score = tf.nn.softmax(predictions[0])  # Computes softmax activations.

    print(np.argmax(score))  # return the index
    result=class_names[np.argmax(score)] # return class name prediction
    lbl1.config(text=str(result))
main=Tk()
main.geometry("400x400")
main.resizable("false","false")

btnButton=Button(main, text="Browse Image", command=browse)
btnButton.pack()

lbl1=Label(main, text="hello")
lbl1.pack()

main.mainloop()






