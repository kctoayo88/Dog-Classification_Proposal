from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import filedialog
import tkinter as tk
 

# Load, reshape and show it 
def load_image(img_path, show=True):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

# Give predicted result
def predicted_result(num_of_class):   
    if num_of_class == 0:
        messageshow(' Akita! ')
    elif num_of_class == 1:
        messageshow(' Corgi! ')
    elif num_of_class == 2:
        messageshow(' Shiaba! ')
    else:
        messageshow(' I do not know it! ')   

# Open predicted image 
def openimage():
    root = tk.Tk()
    fpath = filedialog.askopenfilename()
    root.destroy()
    print (fpath)
    return fpath

def messageshow(Type):
    root = tk.Tk()
    root.withdraw()
    tk.messagebox.showinfo(title=' What type of this dog? ', message=Type)

if __name__ == '__main__':

    # load model
    model = load_model('3_model.h5')

    # Give the image path
    prdicted_image_path = openimage()

    # load a single image
    new_image = load_image(prdicted_image_path)
    
    # check prediction
    pred = model.predict(new_image)
    print(pred)

    #Show the prediction
    num_of_class = np.argmax(pred,axis=1)
    predicted_result(num_of_class)

    
  


