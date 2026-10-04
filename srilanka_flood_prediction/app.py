import streamlit as st
import pandas as pd
import sqlite3
import joblib

original_df = pd.read_csv('sri_lanka_flood_risk_modeled.csv')
con = sqlite3.connect('flood_risk.db')
original_df.to_sql('flood_risk', con, if_exists='replace', index=False)


def load_data():
    scaler_data = joblib.load(
        'D:\Python_Projects\machine_learning_projects\Practice_Supervised_Learning\srilanka_flood_prediction\scaled_data.pkl')
    base_model = joblib.load(
        'D:\Python_Projects\machine_learning_projects\Practice_Supervised_Learning\srilanka_flood_prediction\\flood_risk_model.pkl')

    feature_names = [
        "precipitation_sum", "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
        "temperature_2m_max",
        "wind_speed_10m_max",
        "rain_48h",
        "soil_saturation_index",
        "district_Anuradhapura",
        "district_Badulla",
        "district_Batticaloa",
        "district_Colombo",
        "district_Galle",
        "district_Gampaha",
        "district_Hambantota",
        "district_Jaffna",
        "district_Kalutara",
        "district_Kandy",
        "district_Kegalle",
        "district_Kilinochchi",
        "district_Kurunegala",
        "district_Mannar",
        "district_Matale",
        "district_Matara",
        "district_Monaragala",
        "district_Mullaitivu",
        "district_Nuwara Eliya",
        "district_Polonnaruwa",
        "district_Puttalam",
        "district_Ratnapura",
        "district_Trincomalee",
        "district_Vavuniya",
        "province_Eastern",
        "province_North Central",
        "province_North Western",
        "province_Northern",
        "province_Sabaragamuwa",
        "province_Southern",
        "province_Uva",
        "province_Western",
        "climatic_zone_Intermediate",
        "climatic_zone_Wet"
    ]
    return scaler_data, base_model, feature_names


if __name__ == '__main__':

    scaler, model, trained_features = load_data()
    print("SCALER:", type(scaler))
    print("MODEL:", type(model))
    print("FEATURES:", type(trained_features))
    # Create UI
    st.title("Flood Risk Prediction")

    # Select District
    districts = ['Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'Nuwara Eliya', 'Galle', 'Matara', 'Hambantota',
                 'Jaffna', 'Kilinochchi', 'Mannar', 'Vavuniya', 'Mullaitivu', 'Batticaloa', 'Ampara', 'Trincomalee',
                 'Kurunegala', 'Puttalam', 'Anuradhapura', 'Polonnaruwa', 'Badulla', 'Monaragala', 'Ratnapura',
                 'Kegalle']
    district = st.selectbox(label="Select District", options=districts)

    st.subheader("48-Hour Weather Forecast Indicators")
    rain_48h = st.number_input(label="Forecasted 48h Accumulated Rain (mm)", min_value=0.0, step=0.1)
    pred_temp = st.number_input(label="Forecasted Maximum Temperature (°C)", min_value=-10.0, step=0.1)
    pred_wind = st.number_input(label="Forecasted Maximum Wind Speed (m/s)", min_value=0.0, step=0.1)

    st.subheader("Soil Moisture Indicators")
    curr_soil_0_7 = st.slider(label="Current Soil Moisture (0-7cm)", min_value=0.0, max_value=1.0, step=0.01)
    curr_soil_7_28 = st.slider(label="Current Soil Moisture (7-28cm)", min_value=0.0, max_value=1.0, step=0.01)
    curr_soil_sat = st.slider(label="Current Soil Saturation Index", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Predict Flood Risk"):
        input_data = {feat: 0.0 for feat in trained_features}

        input_data['precipitation_sum'] = rain_48h
        input_data['rain_48h'] = rain_48h
        input_data['temperature_2m_max'] = pred_temp
        input_data['wind_speed_10m_max'] = pred_wind
        input_data['soil_moisture_0_to_7cm_mean'] = curr_soil_0_7
        input_data['soil_moisture_7_to_28cm_mean'] = curr_soil_7_28
        input_data['soil_saturation_index'] = curr_soil_sat

        # Set the corresponding district, province, and climatic zone columns to 1
        if f"district_{district}" in input_data:
            input_data[f"district_{district}"] = 1
            # Automatically map provinces/zones based on selected district
        if district == "Colombo":
            input_data["province_Western"] = 1
            input_data["climatic_zone_Wet"] = 1

        # Convert to DataFrame to lock column order completely
        input_df = pd.DataFrame([input_data])[trained_features]

        standardize_cols = [
            'precipitation_sum',
            'soil_moisture_0_to_7cm_mean',
            'soil_moisture_7_to_28cm_mean',
            'temperature_2m_max',
            'wind_speed_10m_max',
            'rain_48h',
            'soil_saturation_index'
        ]

        # Standardize the input features
        input_df[standardize_cols] = scaler.transform(
            input_df[standardize_cols]
        )

        # Predict
        prediction = model.predict(input_df)

        st.write(f"Predicted Flood Status: **{prediction}**")

    #
