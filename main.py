from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Function to load model
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load Models
xgboost_model = load_model('xgboost-SMOTE.pkl')
naive_bayes_model = load_model('nb_model.pkl')  
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
extra_trees_model = load_model('et_model.pkl')

# Preprocess given data into a dataframe
def preprocess_data(transaction_dict):
    # Create the base input dictionary with exact feature names from training
    input_dict = {
        'amt': transaction_dict['amt'],
        'zip': transaction_dict['zip'],
        'lat': transaction_dict['lat'],
        'long': transaction_dict['long'],
        'unix_time': transaction_dict['unix_time'],
        'merch_lat': transaction_dict['merch_lat'],
        'merch_long': transaction_dict['merch_long'],
        'gender_M': transaction_dict['gender_M'],
        'category_food_dining': transaction_dict['category_food_dining'],
        'category_gas_transport': transaction_dict['category_gas_transport'],
        'category_grocery_net': transaction_dict['category_grocery_net'],
        'category_grocery_pos': transaction_dict['category_grocery_pos'],
        'category_health_fitness': transaction_dict['category_health_fitness'],
        'category_home': transaction_dict['category_home'],
        'category_kids_pets': transaction_dict['category_kids_pets'],
        'category_misc_net': transaction_dict['category_misc_net'],
        'category_misc_pos': transaction_dict['category_misc_pos'],
        'category_personal_care': transaction_dict['category_personal_care'],
        'category_shopping_net': transaction_dict['category_shopping_net'],
        'category_shopping_pos': transaction_dict['category_shopping_pos'],
        'category_travel': transaction_dict['category_travel']
    }

    customer_df = pd.DataFrame([input_dict])
    return customer_df


# Get predictions and probabilities from all models and return average
def get_prediction(transaction_dict):
    preprocessed_data = preprocess_data(transaction_dict)

    # Get predictions from all models
    predictions = {
        'XGBoost': xgboost_model.predict(preprocessed_data)[0],
        'Naive Bayes': naive_bayes_model.predict(preprocessed_data)[0],
        'Random Forest': random_forest_model.predict(preprocessed_data)[0],
        'Decision Tree': decision_tree_model.predict(preprocessed_data)[0],
        'Extra Trees': extra_trees_model.predict(preprocessed_data)[0],
    }
    
    # Get probabilities from models that support predict_proba
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(preprocessed_data)[0][1],
        'Naive Bayes': naive_bayes_model.predict_proba(preprocessed_data)[0][1],
        'Random Forest': random_forest_model.predict_proba(preprocessed_data)[0][1],
        'Decision Tree': decision_tree_model.predict_proba(preprocessed_data)[0][1],
        'Extra Trees': extra_trees_model.predict_proba(preprocessed_data)[0][1],
    }
    
    return predictions, probabilities

# Endpoint to get predictions and probabilities
@app.post("/predict")
async def predict(data: dict):
    prediction, probabilities = get_prediction(data)

    # Convert NumPy types to Python types
    prediction = {model: int(pred) for model, pred in prediction.items()}
    probabilities = {model: float(prob) for model, prob in probabilities.items()}

    return {
        "prediction": prediction,
        "probability": probabilities
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
