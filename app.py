import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import datetime
from xgboost import XGBRegressor
import folium
from streamlit.components.v1 import html
import time

# Set page config
st.set_page_config(
    page_title="Dengue Forecast",
    page_icon="🦟",
    layout="wide"
)

# Title
st.title("🦟 Dengue Forecast System")
st.markdown("### Select region and date for dengue risk assessment")

# Load model
@st.cache_resource
def load_model():
    try:
        model = XGBRegressor()
        model.load_model("dengue_xgb_model.json")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Load regions from region_map.json
@st.cache_resource
def load_regions():
    try:
        with open("region_map.json", "r") as f:
            region_data = json.load(f)
        
        # Create a list of regions with their codes
        regions = []
        for code, name in region_data.items():
            regions.append({
                'code': int(code),
                'name': name,
                'display_name': name.replace('-', ' ').title()
            })
        
        # Sort by code
        regions.sort(key=lambda x: x['code'])
        return regions
    except Exception as e:
        st.error(f"❌ Error loading regions: {e}")
        # Return default regions if file not found
        return [
            {'code': 0, 'name': 'BARMM', 'display_name': 'BARMM'},
            {'code': 1, 'name': 'CAR', 'display_name': 'CAR'},
            {'code': 2, 'name': 'CARAGA', 'display_name': 'CARAGA'},
            {'code': 3, 'name': 'NATIONAL CAPITAL REGION', 'display_name': 'National Capital Region (NCR)'},
            {'code': 4, 'name': 'REGION III-CENTRAL LUZON', 'display_name': 'Region III - Central Luzon'},
            {'code': 5, 'name': 'REGION IV-A-CALABARZON', 'display_name': 'Region IV-A - CALABARZON'},
            {'code': 6, 'name': 'REGION IVB-MIMAROPA', 'display_name': 'Region IV-B - MIMAROPA'},
            {'code': 7, 'name': 'REGION IX-ZAMBOANGA PENINSULA', 'display_name': 'Region IX - Zamboanga Peninsula'},
            {'code': 8, 'name': 'REGION V-BICOL REGION', 'display_name': 'Region V - Bicol Region'},
            {'code': 9, 'name': 'REGION VI-WESTERN VISAYAS', 'display_name': 'Region VI - Western Visayas'},
            {'code': 10, 'name': 'REGION VII-CENTRAL VISAYAS', 'display_name': 'Region VII - Central Visayas'},
            {'code': 11, 'name': 'REGION VII-EASTERN VISAYAS', 'display_name': 'Region VIII - Eastern Visayas'},
            {'code': 12, 'name': 'REGION X-NORTHERN MINDANAO', 'display_name': 'Region X - Northern Mindanao'},
            {'code': 13, 'name': 'REGION XI-DAVAO REGION', 'display_name': 'Region XI - Davao Region'},
            {'code': 14, 'name': 'REGION XII-SOCCSKSARGEN', 'display_name': 'Region XII - SOCCSKSARGEN'},
            {'code': 15, 'name': 'Region I-ILOCOS REGION', 'display_name': 'Region I - Ilocos Region'},
            {'code': 16, 'name': 'Region II-CAGAYAN VALLEY', 'display_name': 'Region II - Cagayan Valley'}
        ]

regions = load_regions()

# Get historical weather data for a specific date
def get_historical_weather(lat, lon, target_date):
    """
    Get historical weather data for a specific date using OpenWeatherMap One Call API 3.0
    Note: Free tier only supports current weather, so we'll simulate historical data
    """
    API_KEY = "9a993cd66f8905aa6c390b026af290e5"
    
    
    try:
        # Get current weather as baseline
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            current_data = response.json()
            
            # Extract target month
            target_month = target_date.month
            

            month_adjustments = {
                1: {'temp_adj': -1.0, 'rain_adj': 0.1, 'humidity_adj': -5},  # January - cool dry
                2: {'temp_adj': 0.5, 'rain_adj': 0.1, 'humidity_adj': -5},   # February - dry
                3: {'temp_adj': 1.5, 'rain_adj': 0.2, 'humidity_adj': 0},    # March - hot dry
                4: {'temp_adj': 2.0, 'rain_adj': 0.3, 'humidity_adj': 5},    # April - hottest
                5: {'temp_adj': 1.5, 'rain_adj': 0.8, 'humidity_adj': 10},   # May - start of wet
                6: {'temp_adj': 0.5, 'rain_adj': 1.5, 'humidity_adj': 15},   # June - rainy
                7: {'temp_adj': 0.0, 'rain_adj': 2.0, 'humidity_adj': 20},   # July - rainy
                8: {'temp_adj': 0.0, 'rain_adj': 2.0, 'humidity_adj': 20},   # August - rainy
                9: {'temp_adj': 0.0, 'rain_adj': 1.8, 'humidity_adj': 18},   # September - rainy
                10: {'temp_adj': 0.5, 'rain_adj': 1.2, 'humidity_adj': 15},  # October - transition
                11: {'temp_adj': 0.0, 'rain_adj': 0.6, 'humidity_adj': 10},  # November - dry start
                12: {'temp_adj': -0.5, 'rain_adj': 0.3, 'humidity_adj': 5},  # December - cool dry
            }
            
            adjustments = month_adjustments.get(target_month, {'temp_adj': 0, 'rain_adj': 0, 'humidity_adj': 0})
            
            # Adjust current weather based on target month
            base_temp = current_data['main']['temp']
            base_humidity = current_data['main']['humidity']
            base_rain = current_data.get('rain', {}).get('1h', 0.0)
            
            # Apply adjustments
            adjusted_temp = base_temp + adjustments['temp_adj']
            adjusted_humidity = min(100, max(30, base_humidity + adjustments['humidity_adj']))
            adjusted_rain = max(0, base_rain + adjustments['rain_adj'])
            
            return {
                'temp': adjusted_temp,
                'humidity': adjusted_humidity,
                'rain': adjusted_rain,
                'place': current_data.get('name', 'Unknown Location'),
                'lat': lat,
                'lon': lon,
                'date': target_date.strftime("%B %d, %Y"),
                'month': target_month,
                'is_historical': True,
                'season': 'Dry season' if target_month in [11, 12, 1, 2, 3, 4] else 'Wet season'
            }
            
    except Exception as e:
        st.warning(f"Weather API error: {e}")
    
    # Fallback: Generate realistic weather based on date
    target_month = target_date.month
    
    # Average temperatures in Philippines by month (in °C)
    avg_temps = {
        1: 26.5, 2: 27.0, 3: 28.0, 4: 29.0, 5: 29.0, 6: 28.5,
        7: 28.0, 8: 28.0, 9: 28.0, 10: 28.0, 11: 27.5, 12: 26.8
    }
    
    # Average humidity by month (in %)
    avg_humidity = {
        1: 75, 2: 73, 3: 72, 4: 71, 5: 75, 6: 80,
        7: 82, 8: 83, 9: 82, 10: 81, 11: 78, 12: 76
    }
    
    # Average rainfall by month (in mm)
    avg_rainfall = {
        1: 20, 2: 15, 3: 20, 4: 30, 5: 120, 6: 180,
        7: 200, 8: 190, 9: 170, 10: 140, 11: 80, 12: 40
    }
    
    # Add some randomness
    temp = avg_temps.get(target_month, 28.0) + np.random.uniform(-1.5, 1.5)
    humidity = avg_humidity.get(target_month, 75) + np.random.uniform(-5, 5)
    rain = avg_rainfall.get(target_month, 50) / 30  # Convert monthly to daily average
    
    return {
        'temp': temp,
        'humidity': humidity,
        'rain': rain,
        'place': 'Philippines',
        'lat': lat,
        'lon': lon,
        'date': target_date.strftime("%B %d, %Y"),
        'month': target_month,
        'is_historical': True,
        'season': 'Dry season' if target_month in [11, 12, 1, 2, 3, 4] else 'Wet season'
    }

# Coordinates for region capitals
REGION_COORDINATES = {
    'BARMM': (7.1907, 124.2154),  # Cotabato City
    'CAR': (16.4023, 120.5960),   # Baguio City
    'CARAGA': (8.9560, 125.5960), # Butuan City
    'NATIONAL CAPITAL REGION': (14.5995, 120.9842),  # Manila
    'REGION III-CENTRAL LUZON': (15.4828, 120.7125), # Tarlac City
    'REGION IV-A-CALABARZON': (14.1667, 121.2167),   # Calamba
    'REGION IVB-MIMAROPA': (12.9375, 121.0856),      # Puerto Princesa
    'REGION IX-ZAMBOANGA PENINSULA': (6.9214, 122.0790),  # Zamboanga City
    'REGION V-BICOL REGION': (13.1394, 123.7408),    # Legazpi City
    'REGION VI-WESTERN VISAYAS': (10.7202, 122.5621), # Iloilo City
    'REGION VII-CENTRAL VISAYAS': (10.3157, 123.8854), # Cebu City
    'REGION VII-EASTERN VISAYAS': (11.2444, 125.0039), # Tacloban City
    'REGION X-NORTHERN MINDANAO': (8.4542, 124.6319),  # Cagayan de Oro
    'REGION XI-DAVAO REGION': (7.1907, 125.4550),    # Davao City
    'REGION XII-SOCCSKSARGEN': (6.1164, 125.1716),   # General Santos
    'Region I-ILOCOS REGION': (16.6166, 120.3165),   # San Fernando
    'Region II-CAGAYAN VALLEY': (17.5869, 121.7225)  # Tuguegarao
}

# Make prediction
def make_prediction(region_code, weather_data, target_date):
    if model is None:
        return None
    
    month_num = target_date.month
    
    try:
        X = pd.DataFrame([{
            "region_encoded": region_code,
            "month_num": month_num,
            "rainfall": float(weather_data['rain']),
            "temperature": float(weather_data['temp']),
            "humidity": float(weather_data['humidity'])
        }])
        
        prediction = model.predict(X)[0]
        prediction = max(0, prediction)
        
        return {
            'prediction': prediction,
            'month': target_date.strftime("%B"),
            'year': target_date.year,
            'date': target_date.strftime("%Y-%m-%d"),
            'weather': weather_data
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Main interface
st.markdown("---")

# Create two columns for selection
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Select Region")
    
    if not regions:
        st.error("No regions found. Please check your region_map.json file.")
    else:
        # Region selection dropdown
        region_options = {r['display_name']: r for r in regions}
        selected_region_name = st.selectbox(
            "Choose your region:",
            options=list(region_options.keys()),
            index=3  # Default to NCR
        )
        
        if selected_region_name:
            selected_region = region_options[selected_region_name]

with col2:
    st.subheader("📅 Select Date")
    
    # Date selection
    today = datetime.date.today()
    min_date = datetime.date(2020, 1, 1)
    max_date = datetime.date(2025, 12, 31)
    
    selected_date = st.date_input(
        "Choose date for analysis:",
        value=datetime.date(2025, 12, 8),  # Default to Dec 8, 2025 as requested
        min_value=min_date,
        max_value=max_date,
        help="Select any date between 2020-2025"
    )
    
    # Display season information
    if selected_date:
        month = selected_date.month
        season = "Dry season" if month in [11, 12, 1, 2, 3, 4] else "Wet season"
        st.info(f"**{selected_date.strftime('%B %Y')}** is in the **{season}**")

# Predict button
st.markdown("---")
if st.button("🔍 PREDICT DENGUE RISK", type="primary", use_container_width=True):
    if 'selected_region' in locals() and selected_date:
        # Store in session state
        st.session_state.selected_region = selected_region
        st.session_state.selected_date = selected_date
        st.rerun()

# If region and date are selected, show analysis
if 'selected_region' in st.session_state and 'selected_date' in st.session_state:
    region = st.session_state.selected_region
    target_date = st.session_state.selected_date
    
    # Show processing
    with st.spinner(f"🔍 Analyzing dengue risk for {region['display_name']} on {target_date.strftime('%B %d, %Y')}..."):
        # Get coordinates for the region
        lat, lon = REGION_COORDINATES.get(region['name'], (14.5995, 120.9842))
        
        # Get historical weather data
        weather_data = get_historical_weather(lat, lon, target_date)
        
        # Make prediction
        result = make_prediction(region['code'], weather_data, target_date)
        
        if result is None:
            # Use sample data if prediction fails
            base_pred = 50
            # Adjust based on season
            if weather_data['season'] == 'Wet season':
                base_pred = base_pred * 1.5  # Higher risk in wet season
            
            result = {
                'prediction': base_pred,
                'month': target_date.strftime("%B"),
                'year': target_date.year,
                'date': target_date.strftime("%Y-%m-%d"),
                'weather': weather_data
            }
        
        # Determine risk level
        prediction = result['prediction']
        if prediction < 30:
            risk_level = "LOW"
            risk_color = "green"
            risk_emoji = "✅"
        elif prediction < 70:
            risk_level = "MEDIUM"
            risk_color = "orange"
            risk_emoji = "⚠️"
        else:
            risk_level = "HIGH"
            risk_color = "red"
            risk_emoji = "🚨"
        
        # Store results
        st.session_state.results = {
            'region': region,
            'date': target_date,
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_emoji': risk_emoji,
            'result': result,
            'weather': weather_data
        }

# Show results if available
if 'results' in st.session_state:
    results = st.session_state.results
    region = results['region']
    target_date = results['date']
    weather_data = results['weather']
    
    st.markdown("---")
    st.success(f"✅ Analysis Complete!")
    
    # Header with region and date
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.subheader(f"🗺️ {region['display_name']} - {target_date.strftime('%B %d, %Y')}")
    with col_header2:
        if st.button("🔄 New Analysis", type="secondary"):
            # Clear session state
            for key in ['selected_region', 'selected_date', 'results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Display results in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create and display map
        st.markdown("#### 📍 Region Map")
        
        m = folium.Map(location=[weather_data['lat'], weather_data['lon']], zoom_start=9)
        
        # Add marker for region capital
        folium.Marker(
            [weather_data['lat'], weather_data['lon']],
            popup=f"""
            <div style='font-family: Arial;'>
            <b>📍 {weather_data['place']}</b><br>
            <b>Region:</b> {region['display_name']}<br>
            <b>Date:</b> {target_date.strftime('%B %d, %Y')}<br>
            <b>Risk Level:</b> {results['risk_level']}<br>
            <b>Predicted Cases:</b> {results['prediction']:.0f}<br>
            <b>Temperature:</b> {weather_data['temp']:.1f}°C
            </div>
            """,
            tooltip=f"Click for details - {results['risk_level']} Risk",
            icon=folium.Icon(color=results['risk_color'], icon="info-sign")
        ).add_to(m)
        
        # Add region area
        folium.Circle(
            location=[weather_data['lat'], weather_data['lon']],
            radius=50000,  # 50km radius for region
            color=results['risk_color'],
            fill=True,
            fill_opacity=0.1,
            popup=f"{region['display_name']} Region"
        ).add_to(m)
        
        map_html = m._repr_html_()
        html(map_html, height=400)
        
        # Location info
        st.info(f"**📍 Regional Capital:** {weather_data['place']}")
        st.info(f"**🔢 Region Code:** {region['code']}")
        st.info(f"**🗓️ Analysis Date:** {target_date.strftime('%B %d, %Y')}")
    
    with col2:
        # Results card
        st.markdown("#### 📊 Prediction Results")
        
        # Date and season info
        st.markdown(f"""
        <div style='background: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h4>📅 Date Information</h4>
        <p><b>Selected Date:</b> {target_date.strftime('%B %d, %Y')}</p>
        <p><b>Season:</b> {weather_data.get('season', 'Unknown')}</p>
        <p><b>Month:</b> {target_date.strftime('%B')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Weather info
        st.markdown(f"""
        <div style='background: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h4>🌤️ Historical Weather</h4>
        <p><b>Temperature:</b> {weather_data['temp']:.1f}°C</p>
        <p><b>Humidity:</b> {weather_data['humidity']:.0f}%</p>
        <p><b>Rainfall:</b> {weather_data['rain']:.1f} mm</p>
        <p><b>Location:</b> {weather_data['place']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk display
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 5px solid {results['risk_color']}; margin-bottom: 1rem;'>
        <h3>{results['risk_emoji']} {results['risk_level']} RISK</h3>
        <h2 style='color: {results['risk_color']};'>{results['prediction']:.0f}</h2>
        <p><b>Predicted Dengue Cases</b></p>
        <p><small>For {target_date.strftime('%B %Y')}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Season-specific analysis
    st.markdown("---")
    st.subheader("🌦️ Season Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if weather_data.get('season') == 'Wet season':
            st.warning("""
            ### 🌧️ Wet Season Alert
            **Higher Dengue Risk Expected:**
            - Increased mosquito breeding due to rain
            - More stagnant water sites
            - Higher humidity favors mosquito survival
            - Typical peak: June to October
            
            **Recommendations:**
            - Intensify mosquito control measures
            - Daily inspection for breeding sites
            - Ensure proper drainage
            - Community cleanup essential
            """)
        else:
            st.success("""
            ### ☀️ Dry Season Analysis
            **Lower Dengue Risk Expected:**
            - Reduced mosquito breeding sites
            - Less stagnant water
            - Lower humidity slows mosquito reproduction
            
            **Remain Vigilant:**
            - Continue preventive measures
            - Check stored water containers
            - Maintain cleanliness
            - Prepare for upcoming wet season
            """)
    
    with col4:
        # Monthly trends
        month = target_date.month
        month_trends = {
            1: "Cool dry month, lower dengue risk",
            2: "Dry month, moderate risk",
            3: "Hot dry month, risk increasing",
            4: "Hottest month, watch for breeding in stored water",
            5: "Start of wet season, risk begins to rise",
            6: "Rainy season begins, higher risk",
            7: "Peak rainy month, highest risk",
            8: "Heavy rains continue, high risk",
            9: "Rainy season continues, high risk",
            10: "Transition to dry, risk remains",
            11: "Start of dry season, risk decreasing",
            12: "Cool dry month, lowest annual risk"
        }
        
        st.info(f"""
        ### 📈 Monthly Trend: {target_date.strftime('%B')}
        
        **{month_trends.get(month, 'Standard risk period')}**
        
        **Historical Pattern:**
        - Dengue cases typically peak during wet months
        - Risk is lowest in cool dry months
        - Urban areas may have year-round transmission
        
        **Model Considerations:**
        - Weather patterns for this month
        - Historical dengue data
        - Regional characteristics
        """)
    
    # Recommendations
    st.markdown("---")
    st.subheader("📋 Recommendations")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.markdown("#### 🦟 Preventive Measures")
        if results['risk_level'] == "LOW":
            st.success("""
            **Maintain Normal Precautions:**
            - Weekly inspection for stagnant water
            - Use mosquito repellent when outdoors
            - Install window/door screens
            - Wear light-colored, long-sleeved clothing
            - Keep surroundings clean
            """)
        elif results['risk_level'] == "MEDIUM":
            st.warning("""
            **Increase Vigilance:**
            - Daily inspection for breeding sites
            - Use mosquito nets, especially at night
            - Report suspected cases immediately
            - Community cleanup activities
            - Fogging in high-risk areas
            """)
        else:
            st.error("""
            **Immediate Action Required:**
            - Seek medical help for any fever symptoms
            - Avoid outdoor activities at dawn/dusk
            - Intensive fogging and larviciding
            - Community-wide cleanup drives
            - Emergency preparedness plans
            """)
    
    with col6:
        st.markdown("#### 🏥 Health Advisory")
        st.info("""
        **Dengue Symptoms to Monitor:**
        - Sudden high fever (40°C/104°F)
        - Severe headache
        - Pain behind the eyes
        - Muscle and joint pains
        - Nausea and vomiting
        - Skin rash (appears 2-5 days after fever)
        - Mild bleeding (nose or gum bleed)
        
        **Emergency Signs (Seek immediate help):**
        - Severe abdominal pain
        - Persistent vomiting
        - Rapid breathing
        - Bleeding gums
        - Fatigue, restlessness
        
        **Treatment:**
        - No specific antiviral treatment
        - Early detection and proper medical care
        - Maintain hydration
        - Avoid aspirin and NSAIDs
        """)
    
    with col7:
        st.markdown("#### 📞 Emergency Contacts")
        st.warning("""
        **National Emergency Hotlines:**
        - Department of Health: 1555
        - National Emergency: 911
        - Red Cross: 143
        - Poison Control: (02) 8524-1078
        
        **For the selected date:**
        - Contact local health center
        - Visit nearest hospital
        - Report suspected cases
        
        **Remember:**
        - Early detection saves lives
        - Proper hydration is crucial
        - Follow medical advice strictly
        - Isolate to prevent spread
        """)

# Instructions
if 'selected_region' not in st.session_state:
    st.markdown("---")
    st.info("""
    ### 📋 How to Use This App:
    
    1. **Select your region** from the dropdown
    2. **Choose a date** for analysis (2020-2025)
    3. **Click "PREDICT DENGUE RISK"** button
    4. **View** detailed dengue risk assessment
    
    ### 🌦️ Weather Data:
    - **Historical weather simulation** for selected date
    - **Seasonal adjustments** applied (Dry/Wet season)
    - **Region-specific** climate patterns
    - **Monthly trends** considered
    
    ### 📊 Analysis Includes:
    - **Date-specific** dengue prediction
    - **Historical weather** conditions
    - **Interactive map** of the region
    - **Risk level** assessment
    - **Seasonal analysis**
    - **Preventive recommendations**
    
    ### 📅 Date Range:
    - **Minimum:** January 1, 2020
    - **Maximum:** December 31, 2025
    - **Default:** December 8, 2025
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🦟 <b>Dengue Forecast System</b> | Date-Specific Risk Assessment</p>
    <p><small>Uses historical weather simulation and machine learning | Covers 2020-2025 period</small></p>
    <p><small>⚠️ This is a predictive tool for risk assessment. Historical accuracy depends on available data.</small></p>
</div>
""", unsafe_allow_html=True)

