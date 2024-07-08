import os
from pathlib import Path

import numpy as np
from processUtil import Process, preprocess_image

from flask import Flask, request, render_template, flash, jsonify
from datetime import datetime as dt

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'
# Upload directory
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def init_app():
    flask_app = Flask(__name__, static_folder="static")

    # APP CONFIGURATIONS
    flask_app.config['SECRET_KEY'] = 'YourSecretKey'
    flask_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    with flask_app.app_context():
        process = Process()
        process.loadCSVDataAndPreProcess()
    return flask_app


app = init_app()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/get_lutein_value', methods=['POST'])
def success():
    global result
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            # return "redirect(request.url)"
            result = {"error": "Invalid file / File is corrupted!"}
            return jsonify(result)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            # return "redirect(request.url)"
            result = {"error": "No file selected!"}
            return jsonify(result)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save the uploaded file as name of current time
            dt_now = dt.now().strftime("%Y%m%d%H%M%S%f")
            filename = dt_now + ".jpg"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # PATH to static files
            img_dir = "./static/uploads/"
            path = img_dir + filename
            print("File Path:" + path)
            example_image = preprocess_image(path)
            prediction = Process.image_model.predict(np.array([example_image]))
            print("Predicted Lutein Value:", prediction)

            # f = request.files['file']
            # # f.save(f.filename)
            # print("path:" + app.root_path + f.name)
            # # print("app.config:" + app.config)
            # # filepath = app.root_path + '\\static'
            # # flowerFile = tempfile.TemporaryFile()
            # # in_memory_file = io.BytesIO()
            # # f.save(in_memory_file)
            # filedir = os.path.join(app.root_path, 'static')
            # filepath = os.path.join(filedir, 'flower.jpg')
            # os.mkdir(filepath)
            # print("File Path:" + filepath)
            # f.save(filepath)
            # # flowerFile.write(in_memory_file.getvalue())
            # # flowerFile.close()
            # # print("path:" + app.root_path + app.config['STATIC_FOLDER'])
            # # print("flowerFile:" + flowerFile.name)
            # # loadCSVDataAndPreProcess()
            # # Example prediction
            # example_image = preprocess_image(filepath)
            # prediction = model.predict(np.array([example_image]))
            # print("Predicted Lutein Value:" + prediction)
            # # return render_template("Acknowledgement.html", name = f.filename)
            cleaned_value = str(prediction).replace("[[", "").replace("]]", "")
            predicted_value = "Predicted Lutein Value: " + cleaned_value
            Path(path).unlink(missing_ok=True)
            result = {"lutein_value": cleaned_value}
        return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
