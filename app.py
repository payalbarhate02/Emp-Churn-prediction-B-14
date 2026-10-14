import pickle
from flask import Flask, render_template_string, request
import pandas as pd

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naive_bayes.pkl', 'rb') as f:
  model = pickle.load(f)

# Embedded HTML/CSS Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #cbd5e1;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 650px;
        }
        h2 {
            margin-top: 0;
            color: var(--primary);
            text-align: center;
            font-size: 24px;
            margin-bottom: 24px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 6px;
        }
        input, select {
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 24px;
            padding: 16px;
            background-color: #e0e7ff;
            border-left: 5px solid var(--primary);
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            color: #312e81;
        }
        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            button {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Employee Prediction Portal</h2>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Education">Education</label>
                    <select name="Education" id="Education" required>
                        <option value="Bachelors">Bachelors</option>
                        <option value="Masters">Masters</option>
                        <option value="PHD">PHD</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="JoiningYear">Joining Year</label>
                    <input type="number" name="JoiningYear" id="JoiningYear" placeholder="e.g. 2018" required>
                </div>
                <div class="form-group">
                    <label for="City">City</label>
                    <select name="City" id="City" required>
                        <option value="Bangalore">Bangalore</option>
                        <option value="Pune">Pune</option>
                        <option value="New Delhi">New Delhi</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="PaymentTier">Payment Tier</label>
                    <select name="PaymentTier" id="PaymentTier" required>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" name="Age" id="Age" placeholder="e.g. 28" required>
                </div>
                <div class="form-group">
                    <label for="Gender">Gender</label>
                    <select name="Gender" id="Gender" required>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="EverBenched">Ever Benched</label>
                    <select name="EverBenched" id="EverBenched" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="ExperienceInCurrentDomain">Experience in Domain</label>
                    <input type="number" name="ExperienceInCurrentDomain" id="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
                </div>
                <button type="submit">Predict</button>
            </div>
        </form>

        {% if prediction_text %}
        <div class="result-card">
            {{ prediction_text }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def home():
  prediction_text = ''
  if request.method == 'POST':
    try:
      # Extracting form inputs mapped to model feature names
      education = request.form['Education']
      joining_year = int(request.form['JoiningYear'])
      city = request.form['City']
      payment_tier = int(request.form['PaymentTier'])
      age = int(request.form['Age'])
      gender = request.form['Gender']
      ever_benched = request.form['EverBenched']
      experience = int(request.form['ExperienceInCurrentDomain'])

      # Create DataFrame matching original feature order[cite: 1]
      input_data = pd.DataFrame([[
          education,
          joining_year,
          city,
          payment_tier,
          age,
          gender,
          ever_benched,
          experience,
      ]], columns=[
          'Education',
          'JoiningYear',
          'City',
          'PaymentTier',
          'Age',
          'Gender',
          'EverBenched',
          'ExperienceInCurrentDomain',
      ])

      prediction = model.predict(input_data)
      prediction_text = f'Prediction Result: {prediction[0]}'
    except Exception as e:
      prediction_text = f'Error in prediction: {str(e)}'

  return render_template_string(
      HTML_TEMPLATE, prediction_text=prediction_text
  )


if __name__ == '__main__':
  app.run(debug=True)
