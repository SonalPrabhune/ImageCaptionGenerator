# Dependencies
from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
#import cv2
from keras.models import load_model
import ImageCaptionGenerator

# Your API definition
app = Flask(__name__)
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    icgen = ImageCaptionGenerator.ImageCaptionGenerator
    try:
        
        jsonRequest = request.json
        print(jsonRequest)

        pic = jsonRequest["body"]
        #file = request.files.get('imagefile', '')
        #print("The file is",file)
        ##r = request
        ### convert string of image data to uint8
        ##nparr = np.fromstring(r.data, np.uint8)
        ### decode image
        ##img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        ##file = request.files['image'].read() ## byte file
        #npimg = np.fromstring(file, np.uint8)
        #print(npimg)
        ##img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        ##cv2.imwrite(ImageCaptionGenerator.images_path+'im-received.jpg', npimg)

        #npimg.dump(ImageCaptionGenerator.images_path+'im-received.jpg')
        #pic = 'im-received.jpg'                
        predicted_caption = icgen.generateCaption(icgen, pic)
        
        return jsonify({'Caption for the picture uploaded is ': str(predicted_caption)})

    except:
        return jsonify({'trace': traceback.format_exc()})
    