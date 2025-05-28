from flask import Flask,render_template, request
import pickle
import numpy as np
import pandas as pd
  

# Initialize the Flask app
app = Flask(__name__)

with open ('classification_model.pkl','rb')as f:
  model= pickle.load(f)

with open('Scaler.pkl', 'rb') as f:
    scaler= pickle.load(f)


edu_order = {
    'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6,
    '12th': 7, 'HS-grad': 8, 'Some-college': 9, 'Assoc-voc': 10, 'Assoc-acdm': 11,
    'Bachelors': 12, 'Masters': 13, 'Prof-school': 14, 'Doctorate': 15
}
sex_map = {'Male': 0, 'Female': 1}

@app.route('/')
def home():
    return render_template('form.html',prediction_text='')

@app.route ('/predict',methods=['POST'])
def index():

    input_data={
        'age':int(request.form['age']),
        "education":request.form['education'],
        'sex':request.form['sex'],
        "capital-gain":int(request.form["capital-gain"]),
       	'capital-loss':int(request.form['capital-loss']),
        'hours-per-week':int(request.form['hours-per-week']),
        'marital_Married-civ-spouse':1 if request.form['marital_Married-civ-spouse']== 'yes'else 0,
        'occupation_Exec-managerial':1 if request.form['occupation_Exec-managerial']== 'yes'else 0
    }
# Create DataFrame
    input_df = pd.DataFrame([input_data])


    input_df['education'] = input_df['education'].map(edu_order)
    input_df['sex'] = input_df['sex'].map(sex_map)

# Scale the input
    numeric_columns = ['age', 'education', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'marital_Married-civ-spouse', 'occupation_Exec-managerial']


    input_df [numeric_columns] = scaler.transform(input_df[numeric_columns])
# Make prediction
    result=model.predict(input_df)

    # Render result
    return render_template('index.html', prediction_text=f'Predicted class: {result}')

if __name__ == '__main__':
    app.run(debug=True)
