#Import necessary libraries
from flask import Flask, render_template, request
import tensorflow as tf
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = r'./potatoes.h5';
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_potato_dieas(potato_plant):
    
  test_image = load_img(potato_plant, target_size = (256, 256)) # load image 
  print("@@ Got Image for prediction")
  
  test_image =img_to_array(test_image) # convert image to np array and normalize
  test_image = test_image/255
  print('/255= ',test_image)
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  print('NP=',test_image)
  result = model.predict(test_image) # predict diseased plant or not
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result, axis=1)    #have a rounded number like 0,1,2

  
  print(pred)
  #if pred==0:
     #   return "Potato - Healthy and Fresh ", 'Potato-Healthy.html'
  
  if pred==0:
      return "Potato - Early Blight Disease", 'Potato-Early_Blight.html'
      
  
  elif pred==1:
         return "Potato - Late_Blight", 'Potato-Late_blight.html'
      
        
  elif pred==2:
        return "Potato - Healthy and Fresh ", 'Potato-Healthy.html'
     

    

# Create flask instance
app = Flask(__name__,template_folder='templates')

# render index.html page
@app.route("/",methods=['GET'])
def home():
        return render_template('index1.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        file_name = file.filename        
        print("@@ Input posted = ", file_name)
         
        file_path = os.path.join(file_name)
        file.save(file_path)


        print("@@ Predicting class......")
        potato_plant = file_path
        pred, output =pred_potato_dieas(potato_plant=file_path)
        
        return render_template(output , pred_output = pred ,user_image = file_path)
    
# For local system
if __name__ == "__main__":
    app.run(debug=True,port=8080) 

    
    