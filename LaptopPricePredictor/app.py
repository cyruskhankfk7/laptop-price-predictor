from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
with open('laptoppricepredictor.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture Form Inputs
        company = request.form['company']
        typename = request.form['typename']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
        ips = 1 if request.form['ips'] == 'Yes' else 0
        ppi = float(request.form['ppi'])
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']

        # 2. Get the exact columns the model was trained on
        # This prevents the "Feature names unseen at fit time" error
        expected_cols = model.feature_names_in_

        # 3. Create a DataFrame with 1 row, filled with 0s, using the model's exact columns
        query_df = pd.DataFrame(0, index=[0], columns=expected_cols)

        # 4. Fill in the numeric values (Ensure these match your exact training column names)
        # Note: If your numeric columns had different capitalizations during training (e.g. 'ram' instead of 'Ram'), update them here.
        if 'Ram' in expected_cols: query_df['Ram'] = ram
        if 'Weight' in expected_cols: query_df['Weight'] = weight
        if 'Touchscreen' in expected_cols: query_df['Touchscreen'] = touchscreen
        if 'Ips' in expected_cols: query_df['Ips'] = ips
        if 'ppi' in expected_cols: query_df['ppi'] = ppi
        if 'HDD' in expected_cols: query_df['HDD'] = hdd
        if 'SSD' in expected_cols: query_df['SSD'] = ssd

        # 5. Fill in the Categorical variables (One-Hot Encoding)
        # We construct the column names just like pd.get_dummies did during training

        company_col = f"Company_{company}"
        if company_col in expected_cols:
            query_df[company_col] = 1

        type_col = f"TypeName_{typename}"
        if type_col in expected_cols:
            query_df[type_col] = 1

        # The error mentioned 'CPU_name_...'
        cpu_col = f"CPU_name_{cpu}"
        if cpu_col not in expected_cols:
            cpu_col = f"Cpu brand_{cpu}"  # Fallback just in case
        if cpu_col in expected_cols:
            query_df[cpu_col] = 1

        # Guessing the prefix for GPU and OS based on standard datasets
        gpu_col = f"Gpu brand_{gpu}"
        if gpu_col in expected_cols:
            query_df[gpu_col] = 1

        os_col = f"os_{os}"
        if os_col in expected_cols:
            query_df[os_col] = 1

        # 6. Make prediction
        prediction = model.predict(query_df)[0]

        # Apply np.exp() since the target variable was likely log-transformed
        predicted_price = int(np.exp(prediction))

        return render_template('index.html', prediction_text=f"Predicted Laptop Price: ₹ {predicted_price:,}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)