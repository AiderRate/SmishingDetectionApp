import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def predict_messages():
    # Load the trained model
    with open('model_pickle','rb') as f:
        mp = pickle.load(f)

    # Load the TF-IDF vectorizer used for training
    with open('vectorizer_pickle', 'rb') as f:
        vectorizer = pickle.load(f)

    # Load the CSV file containing text messages
    input_messages_df = pd.read_csv('input_messages.csv')

    # Extract text messages from the input DataFrame
    input_messages = input_messages_df['Message']

    # Transform the text messages into features using the same TfidfVectorizer used for training
    input_features = vectorizer.transform(input_messages)

    # Predict labels for the input messages
    predictions = mp.predict(input_features)

    # Map numerical predictions to labels (0: ham, 1: smishing, 2: spam)
    prediction_labels = ['ham' if pred == 0 else 'smishing' if pred == 1 else 'spam' for pred in predictions]

    # Create a DataFrame with original messages and predictions
    output_df = pd.DataFrame({'Message': input_messages, 'Prediction': prediction_labels})

    # Write the DataFrame to a new CSV file
    output_df.to_csv('output_predictions.csv', index=False)

# Call the function to execute the predictions
predict_messages()