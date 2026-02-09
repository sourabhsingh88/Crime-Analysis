from flask import render_template, request, jsonify, make_response
from markupsafe import Markup

import pickle
import pandas as pd
import subprocess
import math

from app import app


# =========================
# MODEL LOADING (SAFE)
# =========================

# Working model
kmeanclus = pickle.load(open('./Prediction/kmean.pkl', 'rb'))

# Legacy / broken models (disabled)
kprotoclus = None
rdcls = None
lr = None


# =========================
# UTILITIES
# =========================

class ExponentialSmoothing:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, data):
        self.data = data

    def predict(self, year):
        smoothed = [self.data[0]]
        for i in range(1, len(self.data)):
            smoothed.append(self.alpha * self.data[i] + (1 - self.alpha) * smoothed[i - 1])

        if year < len(smoothed):
            return smoothed[year]

        last = smoothed[-1]
        for _ in range(len(smoothed), year):
            last = self.alpha * self.data[-1] + (1 - self.alpha) * last
        return last


def projection(v1, v2, yr1, yr2, year):
    t = yr2 - yr1
    n = year - yr2
    r = math.pow(v2 / v1, 1 / t) - 1
    return v2 * math.pow(1 + r, n)

@app.route('/timeseriescr')
def timeseriescr():
    return render_template('time-series-forcasting.html')

@app.route('/timeseriesipc')
def timeseriesipc():
    return render_template('total-ipc-forecasting.html')


# =========================
# BASIC PAGES
# =========================

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('About.html')


@app.route('/feedback')
def feedback():
    return render_template('Feedback.html')


# =========================
# KMEANS (WORKING)
# =========================

@app.route('/Kmeans')
def Kmeans():
    return render_template("K-Means.html")


@app.route('/KMeansanalysis', methods=['POST'])
def KMeansanalysis():
    features = [x for x in request.form.values()]
    df = pd.read_csv("Datasets/kmeansflask2.csv")

    years = df.loc[
        (df["STATE/UT"] == features[0].upper()) &
        (df["DISTRICT"] == features[1])
    ]['YEAR'].values

    clusters = []

    for y in years:
        row = df.loc[
            (df["STATE/UT"] == features[0].upper()) &
            (df["DISTRICT"] == features[1]) &
            (df["YEAR"] == y)
        ].values

        final_features = [[x for x in row[0] if isinstance(x, float)]]
        clusters.append(kmeanclus.predict(final_features)[0])

    high = clusters.count(0)
    low = clusters.count(1)
    moderate = clusters.count(2)

    if high > low and high > moderate:
        label = "RED ZONE"
    elif low > high and low > moderate:
        label = "GREEN ZONE"
    elif moderate > high and moderate > low:
        label = "ORANGE ZONE"
    else:
        label = "Crime Rate Varies a Lot"

    return render_template('K-Means.html', prediction_text0=label)


# =========================
# RANDOM FOREST (DISABLED)
# =========================

@app.route('/Randomfrstcls')
def Randomfrstcls():
    return render_template("RandomForestClassifer.html")


@app.route('/randomfrstcls', methods=['POST'])
def randomfrstcls():
    return "RandomForest model unavailable (legacy pickle)", 503


# =========================
# LINEAR REGRESSION (DISABLED)
# =========================

@app.route('/LinearReg')
def LinearReg():
    return render_template("linear-regression.html")


@app.route('/linearreg', methods=["POST"])
def linearreg():
    return "Linear Regression model unavailable (legacy pickle)", 503


# =========================
# ANALYTICS PAGES
# =========================

@app.route('/analysis')
def analysis():
    return render_template('Analysis.html')


@app.route('/analysis2')
def analysis2():
    return render_template('Analysis2.html')


@app.route('/analysis3')
def analysis3():
    return render_template('Analysis3(maps).html')


@app.route('/datadisp')
def datadisp():
    return render_template('datadisplay.html')


@app.route('/foliummap')
def foliummap():
    return render_template("final.html")


# =========================
# CRIME FEED
# =========================

@app.route('/crimefeed')
def crimefeed():
    response = make_response(render_template('crimefeed.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# =========================
# EXTERNAL SCRIPT RUNNER
# =========================

@app.route("/run-file")
def run_file():
    subprocess.Popen(["python", "folium-map/index.py"])
    return jsonify({"message": "File execution initiated"})


# =========================
# HEALTH CHECK
# =========================

@app.route('/health')
def health():
    return jsonify({
        "status": "UP",
        "kmeans": True,
        "random_forest": False,
        "linear_regression": False
    })
