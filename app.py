from flask import Flask, render_template, request, redirect, url_for
import logging
from src.set_logger import set_logger
from src.load_model import load_model
import pandas as pd
from src.config import get_config, get_base_path
import joblib
import os


def load_encoder():
    global encoder
    base_path = get_base_path()
    config = get_config('params.yaml')
    encoder_path = os.path.join(base_path, config['model_dirs'])
    encoder_path = os.path.join(encoder_path, config['encoder'])
    return joblib.load(encoder_path)


logger = logging.getLogger('flask_app')
app = Flask(__name__, template_folder=r'webapp/template', static_folder='webapp/static')
model = load_model('params.yaml')
encoder = load_encoder()


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            age = request.form.get("age")
            region = request.form.get("region")
            bmi = request.form.get("bmi")
            children = request.form.get("children")
            sex = request.form.get("gender")
            smoker = request.form.get("smoker")
            df = pd.DataFrame([[sex, smoker, region]], columns=['sex', 'smoker', 'region'])
            df = encoder.transform(df).toarray()
            df = pd.DataFrame(df, columns=encoder.get_feature_names_out(['sex', 'smoker', 'region']))
            data = pd.DataFrame([[children, age, bmi]], columns=['children', 'age_bin', 'bmi_bin'])
            data = pd.concat([data, df], axis=1)
            output = model.predict(data)
            return render_template(r"index.html", output=round(output[0], 3))
        except Exception as e:
            logger.info("Error while prediction")
            logger.exception(e)
            return redirect(url_for('page_not_found'))
    return render_template(r"index.html")


@app.errorhandler(404)
@app.route('/error', methods=["POST", "GET"])
def page_not_found(e):
    return render_template(r"error_404.html"), 404


if __name__ == "__main__":
    # set_logger('flask_app')
    app.run(debug=True)
