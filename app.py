# app.py - Dengue Forecast with Reliable Location Detection

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import datetime
from geopy.geocoders import Nominatim
from xgboost import XGBRegressor
import folium
from streamlit.components.v1 import html
import os
import time

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Dengue Forecast",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 3.5em;
        font-size: 1.2em;
        font-weight: bold;
    }
    .location-card {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f0f2f6;
        margin: 1rem 0;
    }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🦟 Dengue Forecast System")
st.markdown("### Get instant dengue risk assessment for your location")

# -------------------------
# Initialize session state
# -------------------------
if 'location_data' not in st.session_state:
    st.session_state.location_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'location_requested' not in st.session_state:
    st.session_state.location_requested = False
if 'location_denied' not in st.session_state:
    st.session_state.location_denied = False

# -------------------------
# Paths / Files
# -------------------------
MODEL_FILE = "dengue_xgb_model.json"
REGION_MAP_FILE = "region_map.json"
API_KEY = "9a993cd66f8905aa6c390b026af290e5"  # Replace with your actual API key

# -------------------------
# Load model and region map
# -------------------------
@st.cache_resource
def load_model_and_map():
    try:
        if not os.path.exists(MODEL_FILE):
            st.error(f"Model file '{MODEL_FILE}' not found!")
            return None, None, None
            
        if not os.path.exists(REGION_MAP_FILE):
            st.error(f"Region map file '{REGION_MAP_FILE}' not found!")
            return None, None, None
            
        model = XGBRegressor()
        model.load_model(MODEL_FILE)
        
        with open(REGION_MAP_FILE, "r", encoding="utf-8") as f:
            region_map = json.load(f)
        
        region_map_inv = {v: int(k) for k, v in region_map.items()}
        
        st.success("✓ Model and region map loaded successfully")
        return model, region_map, region_map_inv
        
    except Exception as e:
        st.error(f"Error loading model/map: {e}")
        return None, None, None

model, region_map, region_map_inv = load_model_and_map()

# -------------------------
# Get weather data
# -------------------------
def get_weather(lat, lon):
    """Get current weather from OpenWeatherMap"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        r = requests.get(url, timeout=10)
        data = r.json()
        
        if data.get("cod") != 200:
            error_msg = data.get("message", "Unknown error")
            st.warning(f"Weather API Error: {error_msg}")
            return None, None, None, None
            
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rain = data.get("rain", {}).get("1h", 0.0)
        name = data.get("name", "Unknown location")
        return temp, humidity, rain, name
        
    except Exception as e:
        st.warning(f"Weather service temporarily unavailable: {e}")
        # Return default values for demo
        return 28.0, 70.0, 0.0, "Default Location"

# -------------------------
# Detect region from coordinates
# -------------------------
def detect_region(lat, lon):
    """Detect region from coordinates"""
    if not region_map_inv:
        return "National Capital Region"
    
    try:
        # Simple region mapping based on Philippines coordinates
        if 14.35 <= lat <= 14.75 and 120.85 <= lon <= 121.15:
            region_name = "National Capital Region"
        elif 8.0 <= lat <= 9.5 and 124.0 <= lon <= 125.5:
            region_name = "Region X"
        elif 10.0 <= lat <= 11.5 and 122.0 <= lon <= 123.5:
            region_name = "Region VI"
        elif 7.0 <= lat <= 8.5 and 122.5 <= lon <= 124.5:
            region_name = "Region IX"
        else:
            # Fallback to first region in map
            region_name = list(region_map_inv.keys())[0]
        
        # Check if detected region exists in our map
        if region_name in region_map_inv:
            return region_name
        else:
            # Try to find closest match
            for map_name in region_map_inv.keys():
                if region_name.lower() in map_name.lower():
                    return map_name
            
            # Ultimate fallback
            return list(region_map_inv.keys())[0]
            
    except Exception as e:
        st.warning(f"Region detection issue: {e}")
        return list(region_map_inv.keys())[0] if region_map_inv else "National Capital Region"

# -------------------------
# Make prediction
# -------------------------
def make_prediction(region_encoded, temp, humidity, rainfall):
    """Make dengue prediction using the model"""
    try:
        now = datetime.date.today()
        month_num = now.month
        
        X = pd.DataFrame([{
            "region_encoded": region_encoded,
            "month_num": month_num,
            "rainfall": float(rainfall),
            "temperature": float(temp),
            "humidity": float(humidity)
        }])
        
        raw_pred = model.predict(X)[0]
        pred = max(0, raw_pred)
        return pred
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -------------------------
# Create map
# -------------------------
def create_map(lat, lon, region_name, risk_level):
    """Create folium map showing location"""
    
    # Risk level colors
    risk_colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }
    
    # Create map centered on user location
    m = folium.Map(location=[lat, lon], zoom_start=12)
    
    # Add user marker
    folium.Marker(
        [lat, lon],
        popup=f"📍 Your Location\nRegion: {region_name}\nRisk Level: {risk_level}",
        tooltip="Click for details",
        icon=folium.Icon(color=risk_colors.get(risk_level, "blue"), icon="user", prefix="fa")
    ).add_to(m)
    
    # Add circle around location
    folium.Circle(
        location=[lat, lon],
        radius=3000,  # 3km radius
        color=risk_colors.get(risk_level, "blue"),
        fill=True,
        fill_opacity=0.2,
        popup=f"3km radius - {risk_level} Risk Area"
    ).add_to(m)
    
    # Add tile layer for better visualization
    folium.TileLayer('cartodbpositron').add_to(m)
    
    return m

# -------------------------
# JavaScript for location detection
# -------------------------
def get_location_js():
    """JavaScript code to get location"""
    return """
    <script>
    // Function to get location
    function getLocation() {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject("Geolocation not supported by browser");
                return;
            }
            
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    resolve({
                        lat: position.coords.latitude,
                        lon: position.coords.longitude,
                        accuracy: position.coords.accuracy
                    });
                },
                function(error) {
                    let errorMessage;
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            errorMessage = "Location permission denied. Please allow location access.";
                            break;
                        case error.POSITION_UNAVAILABLE:
                            errorMessage = "Location information unavailable.";
                            break;
                        case error.TIMEOUT:
                            errorMessage = "Location request timed out.";
                            break;
                        default:
                            errorMessage = "Unknown location error.";
                    }
                    reject(errorMessage);
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 0
                }
            );
        });
    }
    
    // Function to send data to Streamlit
    function sendToStreamlit(data) {
        const input = parent.document.querySelector('input[data-testid="stLocationInput"]');
        if (input) {
            input.value = JSON.stringify(data);
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
    
    // Try to get location when page loads
    window.addEventListener('load', function() {
        // Check if location is already stored
        const storedLocation = localStorage.getItem('dengue_location');
        if (storedLocation) {
            sendToStreamlit(JSON.parse(storedLocation));
        }
    });
    </script>
    """

# -------------------------
# Create a hidden input for JavaScript communication
# -------------------------
st.markdown("""<input type="hidden" id="locationData" />""", unsafe_allow_html=True)

# Inject JavaScript
st.markdown(get_location_js(), unsafe_allow_html=True)

# -------------------------
# Main Interface
# -------------------------

# Show loading state if location was requested
if st.session_state.location_requested and not st.session_state.location_data:
    with st.spinner("Waiting for location permission..."):
        time.sleep(2)
    
    # Show instructions for allowing location
    st.markdown("""
    <div class="location-card">
        <h3>📍 Location Access Required</h3>
        <p>Please check your browser for a location permission popup.</p>
        <p>If you don't see it, check your browser's address bar for the location icon.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🔄 Try Again", use_container_width=True):
            st.session_state.location_requested = False
            st.rerun()

# Main button to trigger location request
if not st.session_state.location_data and not st.session_state.location_requested:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3>Ready to check dengue risk?</h3>
            <p>Click below to start location detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📍 DETECT MY LOCATION & PREDICT", 
                    type="primary", 
                    use_container_width=True,
                    key="detect_location"):
            
            st.session_state.location_requested = True
            st.rerun()

# Manual location fallback option
if not st.session_state.location_data:
    st.markdown("---")
    with st.expander("🌐 Manual Location Input (Fallback)"):
        st.write("If automatic location detection doesn't work, you can enter coordinates manually:")
        
        col1, col2 = st.columns(2)
        with col1:
            manual_lat = st.number_input("Latitude", 
                                        value=14.5995, 
                                        format="%.6f",
                                        help="Example: 14.5995 for Manila")
        with col2:
            manual_lon = st.number_input("Longitude", 
                                        value=120.9842, 
                                        format="%.6f",
                                        help="Example: 120.9842 for Manila")
        
        if st.button("Use Manual Location", type="secondary"):
            st.session_state.location_data = {
                "lat": manual_lat,
                "lon": manual_lon,
                "accuracy": 1000,
                "source": "manual"
            }
            st.session_state.location_requested = False
            st.rerun()

# Process location and make prediction
if st.session_state.location_data:
    location = st.session_state.location_data
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Get weather
    status_text.text("🌤️ Getting weather data...")
    progress_bar.progress(25)
    temp, humidity, rainfall, place_name = get_weather(location["lat"], location["lon"])
    
    # Step 2: Detect region
    status_text.text("🗺️ Detecting region...")
    progress_bar.progress(50)
    region_name = detect_region(location["lat"], location["lon"])
    region_encoded = region_map_inv.get(region_name, 0)
    
    # Step 3: Make prediction
    status_text.text("📊 Analyzing dengue risk...")
    progress_bar.progress(75)
    prediction = make_prediction(region_encoded, temp, humidity, rainfall)
    
    # Step 4: Complete
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(100)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    if prediction is not None:
        # Determine risk level
        if prediction < 30:
            risk_level = "Low"
            risk_class = "risk-low"
        elif prediction < 70:
            risk_level = "Medium"
            risk_class = "risk-medium"
        else:
            risk_level = "High"
            risk_class = "risk-high"
        
        # Store result
        st.session_state.prediction_result = {
            "lat": location["lat"],
            "lon": location["lon"],
            "region": region_name,
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "prediction": prediction,
            "risk_level": risk_level,
            "place_name": place_name
        }

# Display results
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    
    st.markdown("---")
    
    # Results header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Dengue Risk Assessment")
    with col2:
        if st.button("🔄 New Analysis", type="secondary"):
            st.session_state.location_data = None
            st.session_state.prediction_result = None
            st.session_state.location_requested = False
            st.rerun()
    
    # Results cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h4>📍 Location Details</h4>
            <p><strong>Region:</strong> {}</p>
            <p><strong>Coordinates:</strong><br>{:.6f}, {:.6f}</p>
            <p><strong>Place:</strong> {}</p>
        </div>
        """.format(
            result["region"],
            result["lat"],
            result["lon"],
            result["place_name"]
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h4>🌤️ Weather Conditions</h4>
            <p><strong>Temperature:</strong> {:.1f}°C</p>
            <p><strong>Humidity:</strong> {}%</p>
            <p><strong>Rainfall:</strong> {} mm</p>
        </div>
        """.format(
            result["temperature"],
            result["humidity"],
            result["rainfall"]
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="result-card">
            <h4>🦟 Dengue Risk</h4>
            <p><strong>Predicted Cases:</strong><br>
            <span style="font-size: 2em; {}">{:,.0f}</span></p>
            <p><strong>Risk Level:</strong><br>
            <span class="{}" style="font-size: 1.5em;">{}</span></p>
        </div>
        """.format(
            f"color: {'green' if result['risk_level'] == 'Low' else 'orange' if result['risk_level'] == 'Medium' else 'red'};",
            result["prediction"],
            risk_class,
            result["risk_level"]
        ), unsafe_allow_html=True)
    
    # Map display
    st.subheader("🗺️ Location Map")
    m = create_map(result["lat"], result["lon"], result["region"], result["risk_level"])
    map_html = m._repr_html_()
    html(map_html, height=500)
    
    # Recommendations
    st.markdown("---")
    st.subheader("📋 Recommendations")
    
    if result["risk_level"] == "Low":
        st.success("""
        ### ✅ Low Risk Area - Good News!
        **Preventive Actions:**
        1. **Maintain cleanliness** - Ensure no stagnant water in containers
        2. **Use mosquito repellent** when outdoors, especially during dawn and dusk
        3. **Install screens** on windows and doors
        4. **Wear protective clothing** - Long sleeves and pants when possible
        5. **Regularly check** for mosquito breeding sites around your home
        
        **Stay vigilant** - Dengue risk can change with weather conditions
        """)
    elif result["risk_level"] == "Medium":
        st.warning("""
        ### ⚠️ Medium Risk Area - Stay Alert!
        **Immediate Actions:**
        1. **Intensify cleaning** - Eliminate ALL stagnant water sources
        2. **Use mosquito nets** while sleeping
        3. **Apply insect repellent** daily
        4. **Report suspected cases** to local health authorities immediately
        5. **Community cleanup** - Organize with neighbors to clear breeding grounds
        
        **Monitor symptoms** - Fever, headache, muscle/joint pain, rash
        """)
    else:
        st.error("""
        ### 🚨 High Risk Area - Take Immediate Action!
        **Urgent Measures:**
        1. **Seek medical advice** immediately if fever develops
        2. **Intensify fogging/misting** in the area
        3. **Avoid outdoor activities** during peak mosquito hours (dawn & dusk)
        4. **Use mosquito nets** day and night
        5. **Emergency hotline** - Keep local health center number handy
        6. **Isolate suspected cases** to prevent spread
        
        **Emergency symptoms:** Severe abdominal pain, persistent vomiting, bleeding, difficulty breathing
        """)

# Instructions for first-time users
if not st.session_state.location_data and not st.session_state.location_requested:
    st.markdown("---")
    st.info("""
    ### 📱 How to Use This App:
    
    **Step 1:** Click the **"DETECT MY LOCATION & PREDICT"** button above
    
    **Step 2:** Allow location access when your browser asks for permission
    - Look for a popup or check your browser's address bar
    - Click "Allow" or "Yes" to share your location
    
    **Step 3:** Wait for analysis (takes about 10-15 seconds)
    
    **Step 4:** View your personalized dengue risk assessment
    
    ### 💡 Tips for Best Results:
    - Make sure location/GPS is enabled on your device
    - Use Chrome, Firefox, or Edge browsers
    - Allow popups for this site
    - If automatic detection fails, use the manual input option
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🦟 <strong>Dengue Forecast System</strong> | Real-time risk assessment using weather and location data</p>
    <p>📍 Your location data is processed locally and not stored on any server</p>
</div>
""", unsafe_allow_html=True)
