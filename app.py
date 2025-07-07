from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    education = int(request.form['education'])
    income = float(request.form['income'])
    loan_amount = float(request.form['loan'])
    credit = int(request.form['credit'])
    property_area = int(request.form['property'])

    data = np.array([[gender, married, education, income, loan_amount, credit, property_area]])
    result = model.predict(data)[0]

    prediction = "✅ Loan Approved" if result == 1 else "❌ Loan Rejected"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
