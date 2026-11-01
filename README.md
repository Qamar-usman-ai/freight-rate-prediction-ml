# Freight Rate Prediction - Road Logistics Machine Learning

A robust and interpretable machine learning pipeline designed to predict road freight posted rates (`posted_rate`) using historical shipment data, numerical attributes, and route-based text features.

---

## 📂 Project Structure & Files

- `train.py`: The complete end-to-end Python script for data preprocessing, feature engineering, leak-free median imputation, TF-IDF vectorization, model training, and prediction.
- `train-test.csv`: Historical training dataset.
- `validation.csv`: Validation dataset containing shipment features.
- `validation-predictions-template.csv`: Template containing the exact required `load_id` identifiers.
- `requirements.txt`: Python package dependencies.
- `validation_predictions.csv`: The final generated output file containing predictions.

---

## ⚙️ Prerequisites & Installation

Make sure you have **Python 3.x** installed on your system. 

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Qamar-usman-ai/freight-rate-prediction-ml.git](https://github.com/Qamar-usman-ai/freight-rate-prediction-ml.git)
   cd freight-rate-prediction-ml
