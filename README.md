📋📘Income Prediction App

A Deep Neural Network (DNN)-based web application built using TensorFlow and Streamlit that predicts whether a person’s income exceeds a certain threshold (e.g., > $50,000/year) based on demographic and employment-related features.

📋📘Overview

This project aims to classify individuals into income categories based on attributes such as age, education level, relationship status, hours worked per week, and capital gain.
The model is trained using a subset of the Adult Income Dataset and deployed via a simple and interactive Streamlit web app.


🧩📂Tech Stack

Frontend/UI-Streamlit

Model-TensorFlow / Keras

Data Processing-Pandas, NumPy, scikit-learn

Model Storage-Joblib, HDF5

Language-Python 3.10+

🧮Dataset

The dataset includes demographic and work-related attributes used for income prediction.

Column Name	Description

Age-Age of the individual

workclass-Type of employment

fnlwgt-Final sampling weight

education-Education level

education-num--Numerical encoding of education

marital-status--Marital status

occupation-Type of occupation

relationship-Relationship status

race-Race category

sex-Gender

capital-gain-Capital gain

capital-loss-Capital loss

hours-per-week-Average working hours per week

⚙️How It Works

Model Training

The dataset is preprocessed (label encoding, normalization).

A DNN classifier is trained using TensorFlow/Keras.

The trained model and encoders are saved as .h5 and .pkl files.

Streamlit Interface

Users input their details (age, education-num, hours-per-week, etc.).

Input data is transformed using saved encoders and scaler.

The DNN model predicts whether the income is above or below the threshold.

Prediction Output

Displays either:

Income <= $50K

Income > $50K

native-country-Country of origin

target_value-Income category (0 = ≤50K, 1 = >50K)


🏆 Future Improvements

Add support for multiple categorical fields (e.g., workclass, education).

Integrate SHAP for explainable AI (model interpretability).

Deploy the model on Streamlit Cloud or Hugging Face Spaces.

Improve accuracy with hyperparameter tuning or feature engineering.



👨‍💻 Author

M Lalpradhap

🎓 B.Tech Artificial Intelligence and Data Science

🏅 National Gold Medalist in Silambam 

📧 [lalpradhapking@gmail.com]
