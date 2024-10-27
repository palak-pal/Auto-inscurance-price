
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import logging
from sqlalchemy import create_engine
from sqlalchemy import create_engine, Column, Integer, String, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import mysql.connector



app = Flask(__name__)


DATABASE_URI = 'mysql+pymysql://root:admin@localhost/auto_insurance_premium_db'  # Update with your credentials
engine = create_engine(DATABASE_URI)

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

# Load the pre-trained model
model = joblib.load('insurance_model.pkl')

@app.route('/')
def home():
    # Render the index.html template
    return render_template('auto_insurance_premium_prediction.html')

# Route to update the model with new data
@app.route('/update-model', methods=['POST'])
def update_model():
    from train_model import train_model  # Import the function to retrain the model
    train_model()
    global model
    model = joblib.load('insurance_model.pkl')  # Reload updated model
    return "Model updated successfully!"


column_mapping = {
    'gender': 'Gender',
    'age': 'Age',
    'Rent Own': 'Rent Own',
    'City': 'City',
    'state': 'State',
    'zip': 'Zip',
    'vehical_year': 'Vehicle Year',
    'vehicle_make': 'Vehicle Make',
    'vehicle_model': 'Vehicle Model',
    'insurance_company': 'Insurance Company',
    'Do You Have A Valid License': 'Do You Have A Valid License'
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data
        data = request.get_json()

        # Map data to the expected columns
        model_data = {column_mapping[key]: value for key, value in data.items() if key in column_mapping}
        
        # Convert to DataFrame
        df = pd.DataFrame([model_data])

        # Check for missing columns that the model expects
        expected_columns = set(column_mapping.values())
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            return jsonify({'error': f'Columns are missing: {missing_columns}'}), 400

        # Make prediction
        prediction = model.predict(df)
        predicted_premium = prediction[0]  # Assuming model returns a single value
        
        return jsonify({'predicted_premium': predicted_premium})

    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': 'Error during prediction'}), 500
    





@app.route('/bulk_import', methods=['POST'])
def bulk_import():
    file_path = request.json.get('file_path')
    
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv('transposed_file.csv')
        
        # Save DataFrame to MySQL database
        df.to_sql('user_inputs', con=engine, if_exists='append', index=False)
        return jsonify({'message': 'Bulk data imported successfully!'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    except FileNotFoundError:
        return jsonify({'error': 'File not found. Please check the file path.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
