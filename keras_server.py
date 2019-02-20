# import the necessary packages
import tensorflow as tf
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask,render_template,request,redirect,flash
import io

# initialize our Flask application and the Keras model
app = Flask(__name__)
# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = r'_5#y2L"F4Q8zsdfsdfec]/'
model = ResNet50(weights="imagenet")
graph = tf.get_default_graph()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

load_model()

@app.route("/")
def hello():
    return render_template("main.html")

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        
        if 'image' not in request.files:
            flash('No file')
            return redirect('/')
        image = request.files['image']
        if image and allowed_file(image.filename):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                    preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

           # loop over the results and add them to the list of
           # returned predictions
            for (_, label, prob) in results[0]:
                r = {"label": label.replace("_"," "), "probability": round(float(prob)*100)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True
            return render_template("main.html",data=data)
        return redirect('/')
