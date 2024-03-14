from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2

import imutils
import sklearn
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

#joblib.load('./models/breast_cancer_model.pkl')
# Loading Models
# covid_model = load_model('./models/covid.h5')
# breastcancer_model = pickle.load(open('./models/breast_cancer_model.pkl','rb'))
alzheimer_model = load_model('./models/alzheimer_model.h5')
pneumonia_model = load_model('./models/pneumonia_model.h5')
skin_model=load_model('./models/skin.h5')
breastcancer_model=load_model('./models/BreastCancerModel.h5')


# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

############################################# BRAIN TUMOR FUNCTIONS ################################################

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('services.html')


@app.route('/Skin.html')
def skin():
    return render_template('Skin.html')


# @app.route('/services.html')
# def services():
#     return render_template('services.html')


@app.route('/breastcancer.html')
def brain_tumor():
    return render_template('breastcancer.html')


# @app.route('/contact.html')
# def contact():
#     return render_template('contact.html')


@app.route('/alzheimer.html')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/pneumonia.html')
def pneumonia():
    return render_template('pneumonia.html')


# @app.route('/about.html')
# def about():
#     return render_template('about.html')


########################### Result Functions ########################################


# @app.route('/resultc', methods=['POST'])
# def resultc():
#     if request.method == 'POST':
#         firstname = request.form['firstname']
#         lastname = request.form['lastname']
#         email = request.form['email']
#         phone = request.form['phone']
#         gender = request.form['gender']
#         age = request.form['age']
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             flash('Image successfully uploaded and displayed below')
#             img = cv2.imread('static/uploads/'+filename)
#             img = cv2.resize(img, (224, 224))
#             img = img.reshape(1, 224, 224, 3)
#             img = img/255.0
#             pred = covid_model.predict(img)
#             if pred < 0.5:
#                 pred = 0
#             else:
#                 pred = 1
#             # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
#             return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

#         else:
#             flash('Allowed image types are - png, jpg, jpeg')
#             return redirect(request.url)


# @app.route('/resultbc', methods=['POST'])
# def resultbc():
#     if wrequest.method == 'POST':
#         firstname = request.form['firstname']
#         lastname = request.form['lastname']
#         email = request.form['email']
#         phone = request.form['phone']
#         gender = request.form['gender']
#         age = request.form['age']
#         cpm = request.form['concave_points_mean']
#         am = request.form['area_mean']
#         rm = request.form['radius_mean']
#         pm = request.form['perimeter_mean']
#         cm = request.form['concavity_mean']
#         pred = breastcancer_model.predict(
#             np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
#         # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Breast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
#         return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

@app.route('/resulta', methods=['GET', 'POST'])
def resulta():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3)
            img = img/255.0
            pred = alzheimer_model.predict(img)
            pred = pred[0].argmax()
            print(pred)
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resulta.html', filename=filename, fn=firstname, ln=lastname, age=age, r=0, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')
@app.route('/resultbc', methods=['GET', 'POST'])
def resultbc():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (48, 48, 3))
            img = np.expand_dims(img, axis=0)
            img = img/255.0
            pred = breastcancer_model.predict(img)
            predc = np.asscalar(np.argmax(pred, axis=1))
            indices = {0: 'Benign', 1: 'IDC'}
            accuracy = round(pred[0][predc] * 100, 2)
            label = indices[accuracy]
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resultbc.html', filename=filename, fn=firstname, ln=lastname, age=age, r={'pred':label}, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')
@app.route('/resultsc', methods=['GET', 'POST'])
def resultsc():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img,(28,28))/255
            img = np.expand_dims(img,axis=0)
            
            pred = skin_model.predict(img)
            pred = np.argmax(pred)
            label = {
            0:'Actinic keratoses',
            1:'Basal cell carcinoma',
            2:'Seborrhoeic Keratosis',
            3:'Dermatofibroma',
            4:'Melanocytic nevi',
            6:'Melanoma',
            5:'Vascular lesions'
        }
            print(r)
            r=label[pred]
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resultsc.html', filename=filename, fn=firstname, ln=lastname, age=age, r={'r':r}, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')


@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)



# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
