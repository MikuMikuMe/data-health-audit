# data-health-audit

Creating a Python program to automate the detection and rectification of anomalies and inconsistencies in large datasets involves various tasks such as loading data, detecting anomalies, handling missing values, and more. Below is a complete Python program that serves as a basic framework for this purpose. It uses libraries like pandas for data manipulation and scikit-learn for anomaly detection.

Before running this program, ensure you have installed all the necessary packages: `pandas`, `numpy`, and `scikit-learn`.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file.
    :return: Loaded pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully from %s", file_path)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def detect_anomalies(data):
    """
    Detect anomalies using Isolation Forest.

    :param data: Pandas DataFrame.
    :return: DataFrame with an anomaly score for each record.
    """
    try:
        model = IsolationForest(contamination=0.01, random_state=42)
        data_numeric = data.select_dtypes(include=[np.number])
        data['anomaly_score'] = model.fit_predict(data_numeric)
        logging.info("Anomalies detected using Isolation Forest")
        return data
    except Exception as e:
        logging.error("Error detecting anomalies: %s", e)
        raise

def handle_missing_values(data):
    """
    Fill missing values with the mean of each column.

    :param data: Pandas DataFrame.
    :return: DataFrame with missing values handled.
    """
    try:
        imputer = SimpleImputer(strategy='mean')
        data.iloc[:, :] = imputer.fit_transform(data)
        logging.info("Missing values handled using mean imputation")
        return data
    except Exception as e:
        logging.error("Error handling missing values: %s", e)
        raise

def rectify_anomalies(data):
    """
    Remove anomalies based on anomaly scores.

    :param data: Pandas DataFrame with anomaly scores.
    :return: Cleaned DataFrame.
    """
    try:
        clean_data = data[data['anomaly_score'] != -1]
        clean_data.drop(columns='anomaly_score', inplace=True)
        logging.info("Anomalies rectified by removing flagged records")
        return clean_data
    except Exception as e:
        logging.error("Error rectifying anomalies: %s", e)
        raise

def main():
    # Path to the CSV file
    file_path = 'path/to/your/dataset.csv'
    
    # Load data
    data = load_data(file_path)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Detect anomalies
    data = detect_anomalies(data)
    
    # Rectify anomalies
    data_cleaned = rectify_anomalies(data)
    
    # Save the cleaned data to a new CSV file
    try:
        data_cleaned.to_csv('cleaned_dataset.csv', index=False)
        logging.info("Cleaned data saved to cleaned_dataset.csv")
    except Exception as e:
        logging.error("Error saving cleaned data: %s", e)

if __name__ == "__main__":
    main()
```

### Key Features and Considerations

1. **Logging:** This program uses the logging module to keep track of each process stage, including loading data, handling missing values, detecting anomalies, and rectifying them.
   
2. **Error Handling:** It wraps key operations in try-except blocks to handle and log any exceptions that occur during execution.

3. **Anomaly Detection:** Utilizes the Isolation Forest algorithm, which is effective for identifying outliers in high-dimensional datasets.

4. **Missing Values Imputation:** Replaces missing values with the mean of each column using `SimpleImputer`.

5. **Flexibility:** This script can be adapted and extended to handle different types of pre-processing as per the dataset needs.

Before running the script, make sure to replace `'path/to/your/dataset.csv'` with the actual path to your CSV dataset. The processed dataset will be saved as `cleaned_dataset.csv`.