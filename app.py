import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open("solar_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="🔆",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and subtitle with styling
st.markdown(
    """
    <h1 style='text-align: center; color: #FF8C00; font-weight: 900;'>
    🔆 Solar Power Prediction App
    </h1>
    <p style='text-align: center; font-size:18px; color:#333; margin-bottom: 2rem;'>
    Enter environmental and system details below to predict solar power output in kW.
    </p>
    """,
    unsafe_allow_html=True,
)

# Styled container for inputs - vertical layout
with st.container():
    st.markdown(
        """
        <div style="
            background: #fff7e6;
            padding: 25px 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(255, 140, 0, 0.2);
            margin-bottom: 2rem;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        ">
        """, unsafe_allow_html=True)

    temperature = st.number_input(
        "🌡️ Temperature (°C)", 0.0, 50.0, 30.0, step=0.1,
        help="Ambient temperature affecting solar efficiency"
    )
    humidity = st.number_input(
        "💧 Humidity (%)", 0.0, 100.0, 50.0, step=0.1,
        help="Relative humidity in the environment"
    )
    cloud_coverage = st.number_input(
        "☁️ Cloud Coverage (%)", 0.0, 100.0, 20.0, step=0.1,
        help="Cloudiness impacting sunlight"
    )
    ambient_pressure = st.number_input(
        "📈 Ambient Pressure (hPa)", 900.0, 1100.0, 1010.0, step=0.1,
        help="Atmospheric pressure around the solar panel"
    )
    irradiance = st.number_input(
        "☀️ Irradiance (W/m²)", 0.0, 1200.0, 850.0, step=0.1,
        help="Solar irradiance or sunlight intensity"
    )
    wind_speed = st.number_input(
        "🌬️ Wind Speed (m/s)", 0.0, 20.0, 5.0, step=0.1,
        help="Wind speed near the panels"
    )
    sunshine_hours = st.number_input(
        "🌞 Sunshine Hours", 0.0, 12.0, 10.0, step=0.1,
        help="Duration of sunlight in hours"
    )
    panel_tilt_angle = st.number_input(
        "🔧 Panel Tilt Angle (°)", 0.0, 90.0, 30.0, step=0.1,
        help="Tilt angle of the solar panels"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# Predict button centered with some space
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("⚡ Predict Solar Power", help="Click to get solar power prediction")

# Output box with styling
if predict_button:
    with st.spinner("Calculating..."):
        input_data = np.array([[temperature, irradiance, humidity, wind_speed, cloud_coverage,
                                sunshine_hours, ambient_pressure, panel_tilt_angle]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

    st.markdown(
        f"""
        <div style="
            background: #e0f7fa;
            border-left: 6px solid #00bcd4;
            padding: 20px 25px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 188, 212, 0.3);
            margin-top: 2rem;
            font-size: 22px;
            font-weight: 700;
            text-align: center;
            color: #007c91;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        ">
            🔋 Predicted Solar Power Output:<br><span style='font-size: 36px;'>{prediction:.2f} kW</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.balloons()

# Footer note
st.markdown(
    """
    <hr>
    <p style="font-size:12px; text-align:center; color:gray;">
    Developed by AKHILA TUMU
    </p>
    """,
    unsafe_allow_html=True,
)
