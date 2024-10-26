import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def galaxy_cluster_analysis():
    # Load the data
    sky = pd.read_csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv")

    # Drop non-informative columns and select specific features
    features_to_use = ["ra", "dec", "u", "g", "r", "i", "z", "redshift"]
    sky = sky[features_to_use + ["class"]]  # Include the target variable

    # Encode the target variable
    le = preprocessing.LabelEncoder()
    sky["class"] = le.fit_transform(sky["class"])

    # Prepare the features and target
    X = sky.drop("class", axis=1)
    y = sky["class"]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Train a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Streamlit app
    st.markdown(
        """
        <h1 style="text-align: center;">Galaxy Cluster Analysis</h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: center; font-weight: bold; font-size: 18px;">
            Utilize machine learning algorithms to classify galaxies into types such as spiral, elliptical, and irregular based on key features like morphology, brightness, and color. The model continuously refines classifications as new data becomes available, improving accuracy and insight into galactic formation and evolution. <br>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Enter the features to classify an object:")

    # Create input fields for each feature based on the specified features
    features = {}
    for feature in features_to_use:
        features[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

    # Prepare the input data
    input_data = np.array(list(features.values())).reshape(1, -1)

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the class
    if st.button("Predict"):
        prediction = rf.predict(input_data_scaled)
        class_name = le.inverse_transform(prediction)[0]
        st.write(f"The predicted class is: {class_name}")
