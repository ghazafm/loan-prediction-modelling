from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pandas as pd

app = Flask(__name__)

# Memuat model yang telah dilatih
model = joblib.load('Model/xgboost_model.pkl')

# Kolom numerik dan kategorikal yang digunakan dalam model
numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
categorical_features = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Label encoder dan scaler yang digunakan saat pelatihan
label_encoders = LabelEncoder()
label_encoders = joblib.load('Model/encoder.pkl')  # Assuming you have a dictionary of encoders
scaler = joblib.load('Model/scaler.bin')

# Halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan data dari form di halaman web
        data = request.form.to_dict()
        
        # Konversi tipe data numerik
        for key in numeric_features:
            data[key] = float(data[key])
        
        # Konversi data ke DataFrame
        df = pd.DataFrame([data])

        # Preprocessing data seperti pada saat pelatihan
        for feature in categorical_features:
            le = label_encoders[feature]
            df[feature] = le.transform(df[feature])
        df[numeric_features] = scaler.transform(df[numeric_features])

        # Melakukan prediksi
        prediction = model.predict(df)
        result = 'Layak' if prediction[0] == 0 else 'Tidak Layak'

        # Kembalikan hasil prediksi ke template
        return render_template('index.html', prediction_text=f'Prediksi: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
