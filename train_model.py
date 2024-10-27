import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from datetime import datetime

# Load the CSV files
file_path_1 = 'transposed_file.csv'
file_path_2 = 'Car Data.csv'

# Reading both CSV files
data_1 = pd.read_csv(file_path_1)
data_2 = pd.read_csv(file_path_2)

# Print column names to debug
print("Columns in transposed_file.csv:", data_1.columns.tolist())
print("Columns in Car Data.csv:", data_2.columns.tolist())

# Merge datasets (if necessary)
data = pd.concat([data_1, data_2], axis=1)

# Fill missing values for numeric and categorical columns separately
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Fill numeric columns with mean and categorical columns with mode
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Check for duplicate columns and remove them if necessary  
data = data.loc[:, ~data.columns.duplicated()]  

# Print merged DataFrame columns for verification
print("Columns after merging:", data.columns.tolist())

# Check for duplicate columns and remove them if necessary
data = data.loc[:, ~data.columns.duplicated()]

# Function to calculate age from birthdate
def calculate_age(birthdate):
    today = datetime.now()
    birthdate = pd.to_datetime(birthdate)  # Changed to use the passed argument
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

# Check if 'DOB' column exists
if 'DOB' in data.columns:  # Changed to 'DOB' to match your dataset
    # Calculate age and create a new column 'Age'
    data['Age'] = data['DOB'].apply(calculate_age)
else:
    print("'DOB' column not found in the dataset.")

# Defining the features (X) and target variable (y)
X = data.drop(columns=['Premium'])  # Replace 'Premium' with the actual column name if different
y = data['Premium']

# Check if 'Age' and other features exist
print("Features available for training:", X.columns.tolist())

# Define the categorical and numerical columns
categorical_cols = ['Zip', 'City', 'State', 'Gender', 'Rent Own', 
                    'Insurance Company', 'Vehicle Make', 
                    'Vehicle Model', 'Do You Have A Valid License']
numerical_cols = ['Age', 'Vehicle Year']

# Check if categorical and numerical columns are present in X
missing_categorical = [col for col in categorical_cols if col not in X.columns]
missing_numerical = [col for col in numerical_cols if col not in X.columns]

if missing_categorical or missing_numerical:
    print("Missing categorical columns:", missing_categorical)
    print("Missing numerical columns:", missing_numerical)

# Preprocessing pipeline
preprocessor = ColumnTransformer(  
    transformers=[  
        ('num', StandardScaler(), numerical_cols),  
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  
    ]) 


# Define models and their hyperparameters  
models = {  
    'RandomForest': {  
        'model': RandomForestRegressor(random_state=42),  
        'hyperparameters': {  
            'regressor__n_estimators': [100, 200, 500],  
            'regressor__max_depth': [10, 20, None],  
            'regressor__min_samples_split': [2, 5, 10],  
            'regressor__min_samples_leaf': [1, 2, 4],  
            'regressor__max_features': ['auto', 'sqrt', 'log2'],  
        }  
    },  
    'GradientBoosting': {  
        'model': GradientBoostingRegressor(random_state=42),  
        'hyperparameters': {  
            'regressor__n_estimators': [100, 200, 500],  
            'regressor__max_depth': [10, 20, None],  
            'regressor__min_samples_split': [2, 5, 10],  
            'regressor__min_samples_leaf': [1, 2, 4],  
            'regressor__max_features': ['auto', 'sqrt', 'log2'],  
        }  
    }  
} 

best_model = None
best_score = float('inf')

for model_name, model_config in models.items():  
    pipeline = Pipeline(steps=[  
        ('preprocessor', preprocessor),  
        ('regressor', model_config['model'])  
    ]) 

# Split the dataset into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
 
# Set up Grid Search with cross-validation  
grid_search = GridSearchCV(pipeline, model_config['hyperparameters'], cv=5, scoring='neg_mean_squared_error')  # Use model_config['hyperparameters']  
grid_search.fit(X_train, y_train)  

# Print the best parameters found by Grid Search  
print(f"Best Parameters for {model_name}:", grid_search.best_params_) 

# Evaluate the best estimator  
best_model = grid_search.best_estimator_  
y_pred = best_model.predict(X_test)  

# Print evaluation metrics  
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  

print(f"Mean Squared Error for {model_name}:", mse)  
print(f"R^2 Score for {model_name}:", r2)  


 # Save the best model if it has the lowest MSE
if mse < best_score:
        best_score = mse
        best_model = best_model

# Train the model and save it
def train_model():
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'insurance_model.pkl')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
