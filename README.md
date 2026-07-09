# 🏠 House Price Predictor

A Streamlit app that trains three regression models (Linear Regression, Random
Forest, Gradient Boosting) live on a housing dataset, picks the best performer
by R², and lets you search for houses in a city within your budget or compare
average prices across cities.

**Live demo:** _add your Streamlit Cloud link here after deploying_

## How it works
- `app.py` — loads `house_price_train_dataset.csv`, trains all three models on app start (cached), and serves the prediction/comparison UI.
- `house_price_train_dataset.csv` — training data (city, locality, distance from city centre, area, BHK, land price/sqft, total price).

## Run locally
```bash
git clone <your-repo-url>
cd house-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack
Python, Streamlit, scikit-learn, pandas, matplotlib
