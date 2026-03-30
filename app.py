from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
except Exception as e:
    print("Model loading error:", e)
    model = None
    vectorizer = None

items = pd.read_csv(os.path.join(BASE_DIR, "20191226-items.csv"))

reviews = pd.read_csv(
    os.path.join(BASE_DIR, "20191226-reviews.csv"),
    usecols=["asin", "body"]
)

review_col = "body"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        product = request.form.get("product", "").strip()

        if product == "":
            error = "Please enter a product name."
            return render_template("index.html", result=result, error=error)

        matched_items = items[items["title"].str.contains(product, case=False, na=False)]

        if matched_items.empty:
            error = "No matching product found."
            return render_template("index.html", result=result, error=error)

        asin_list = matched_items["asin"].unique()
        product_reviews = reviews[reviews["asin"].isin(asin_list)].copy()

        product_reviews = product_reviews.dropna(subset=[review_col])

        if product_reviews.empty:
            error = "No reviews found."
            return render_template("index.html", result=result, error=error)

        if model is None or vectorizer is None:
            error = "Model not found. Run train_model.py first."
            return render_template("index.html", result=result, error=error)

        X = vectorizer.transform(product_reviews[review_col])
        predictions = model.predict(X)

        product_reviews["sentiment"] = predictions
        result = product_reviews["sentiment"].value_counts().to_dict()

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)

app = Flask(__name__)
app.debug = True
