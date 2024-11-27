from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
from flask import Flask, render_template, request
from markupsafe import Markup
import requests
from datetime import datetime
import base64
import hmac
import hashlib
import time
import json

# Loading plant disease classification model
app = Flask(__name__)

# Gemini API functions (unchanged)
@app.route('/gemini-prices', methods=['GET'])
def get_gemini_prices():
    endpoint = "pricefeed"
    prices = gemini_request(endpoint)

    if prices:
        return render_template('gemini-prices.html', prices=prices)
    else:
        return "Failed to fetch prices."

def get_usd_to_inr_rate():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        if response.status_code == 200:
            data = response.json()
            return data["rates"]["INR"]
        else:
            print("Error fetching exchange rate.")
            return None
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return None

def get_btc_price():
    try:
        endpoint = "v1/pubticker/btcusd"
        response = requests.get(f"https://api.gemini.com/{endpoint}")
        usd_to_inr_rate = get_usd_to_inr_rate()
        if response.status_code == 200 and usd_to_inr_rate:
            data = response.json()
            btc_price_usd = float(data.get('last', 0))
            btc_price_inr = btc_price_usd * usd_to_inr_rate
            return {
                "price_usd": btc_price_usd,
                "price_inr": round(btc_price_inr, 2)
            }
        else:
            return {"price_usd": "N/A", "price_inr": "N/A"}
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        return {"price_usd": "N/A", "price_inr": "N/A"}


def gemini_request(endpoint, payload=None):
    base_url = "https://api.gemini.com/v1"
    url = f"{base_url}/{endpoint}"
    gemini_api_key = config.gemini_api_key
    gemini_api_secret = config.gemini_api_secret.encode()

    if payload is None:
        payload = {}
    payload['nonce'] = int(time.time() * 1000)
    encoded_payload = base64.b64encode(json.dumps(payload).encode())
    signature = hmac.new(gemini_api_secret, encoded_payload, hashlib.sha384).hexdigest()

    headers = {
        'Content-Type': 'text/plain',
        'Content-Length': '0',
        'X-GEMINI-APIKEY': gemini_api_key,
        'X-GEMINI-PAYLOAD': encoded_payload.decode(),
        'X-GEMINI-SIGNATURE': signature
    }

    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error in Gemini API call: {e}")
        return None


# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# Custom functions for calculations
def weather_fetch(city_name, state_code=None, country_code="IN"):
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    
    if state_code:
        location = f"{city_name},{state_code},{country_code}"
    else:
        location = f"{city_name},{country_code}"
    
    complete_url = f"{base_url}q={location}&appid={api_key}"
    response = requests.get(complete_url)
    
    try:
        response.raise_for_status()
        data = response.json()

        if data.get("cod") != "404":
            temperature = round(data["main"]["temp"] - 273.15, 2)
            humidity = data["main"]["humidity"]
            return temperature, humidity
        else:
            return None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

# Render home page
@app.route('/')
def home():
    title = 'Krushimitra - Home'
    return render_template('index.html', title=title)

# Render crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'Krushimitra - Crop Recommendation'
    return render_template('crop.html', title=title)

# Render fertilizer recommendation form page
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Krushimitra - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

# Render crop recommendation result page
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Krushimitra - Crop Recommendation'

    if request.method == 'POST':
        # User inputs
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")
        lat = 19.0760  # Replace with actual lat for the city
        lon = 72.8777  # Replace with actual lon for the city

        # Fetch weather data
        weather_data = weather_fetch(city)
        
        # Fetch BTC price from Gemini API
        btc_price = get_btc_price()

        if weather_data:
            temp, humidity = weather_data
            # Run ML prediction
            data = np.array([[N, P, K, temp, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            # Return weather data as a dictionary
            weather_data_dict = {'temperature': temp, 'humidity': humidity}

            return render_template(
                'crop-result.html',
                prediction=final_prediction,
                title=title,
                weather=weather_data_dict,  # Pass weather data as a dictionary
                btc_price=btc_price,
                city = city # Pass the BTC price (both USD and INR) to the template
            )
        else:
            return render_template('try_again.html', title=title, weather=weather_data)


#New route for fertilizer recommendation
@app.route('/fert-recommend', methods=['POST'])
def fert_recommend():
    title = 'Krushimitra - Fertilizer Recommendation'
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        crop = request.form['cropname']

        N_status = 'NHigh' if N > 100 else 'Nlow'
        P_status = 'PHigh' if P > 100 else 'Plow'
        K_status = 'KHigh' if K > 100 else 'Klow'

        recommendation = f"""
            <h3>Recommendation for {crop}:</h3>
            <p><b>Nitrogen:</b> {N_status}<br/>{fertilizer_dic[N_status]}</p>
            <p><b>Phosphorous:</b> {P_status}<br/>{fertilizer_dic[P_status]}</p>
            <p><b>Potassium:</b> {K_status}<br/>{fertilizer_dic[K_status]}</p>
        """
        return render_template('fertilizer-result.html', recommendation=Markup(recommendation), title=title)

if __name__ == '__main__':
    app.run(debug=False)
