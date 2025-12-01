import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIGURATION - MOBILE OPTIMIZED
# ============================================
st.set_page_config(
    page_title="Dengue Forecast",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ============================================
# RESPONSIVE DESIGN - Works on Phone & Desktop
# ============================================
st.markdown("""
<style>
    /* Mobile-friendly CSS */
    .main { 
        padding: 10px; 
        max-width: 100%;
    }
    .stMetric { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 10px;
        margin: 5px 0;
    }
    h1 { 
        color: #e74c3c; 
        font-size: 2em; 
        margin-bottom: 5px;
        text-align: center;
    }
    h2 { 
        color: #c0392b; 
        font-size: 1.5em; 
        margin-top: 15px;
    }
    h3 { 
        color: #333; 
        font-size: 1.1em;
    }
    .stButton > button {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
    }
    .stNumberInput, .stSelectbox {
        margin: 5px 0;
    }
    /* Mobile responsive */
    @media (max-width: 640px) {
        h1 { font-size: 1.5em; }
        h2 { font-size: 1.2em; }
        .stMetric { padding: 10px; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HIDDEN: Load and train model
# ============================================
@st.cache_resource
def load_and_train_model():
    try:
        csv_path = "Philippines_Dengue_2016_2024_TEMPLATE.csv"
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error("❌ Dataset file not found!")
        st.stop()

    np.random.seed(42)
    mask = df['dengue_cases'].isna()
    df.loc[mask, 'dengue_cases'] = np.random.randint(50, 500, size=mask.sum())
    df['dengue_deaths'] = df['dengue_deaths'].fillna(0)

    df = df.sort_values(['region', 'year', 'month']).reset_index(drop=True)

    df['prev_cases_1'] = df.groupby('region')['dengue_cases'].shift(1).fillna(0)
    df['prev_cases_2'] = df.groupby('region')['dengue_cases'].shift(2).fillna(0)
    df['prev_cases_3'] = df.groupby('region')['dengue_cases'].shift(3).fillna(0)
    df['prev_cases_4'] = df.groupby('region')['dengue_cases'].shift(4).fillna(0)
    df['prev_cases_5'] = df.groupby('region')['dengue_cases'].shift(5).fillna(0)
    df['prev_cases_6'] = df.groupby('region')['dengue_cases'].shift(6).fillna(0)

    df['ma_3'] = df.groupby('region')['dengue_cases'].rolling(window=3).mean().reset_index(0, drop=True).fillna(0)
    df['ma_6'] = df.groupby('region')['dengue_cases'].rolling(window=6).mean().reset_index(0, drop=True).fillna(0)

    df['month_num'] = df['month'].astype(int)

    rainfall_map = {1: 60, 2: 50, 3: 40, 4: 80, 5: 150, 6: 250,
                    7: 300, 8: 290, 9: 280, 10: 200, 11: 120, 12: 80}
    temp_map = {1: 26, 2: 27, 3: 28, 4: 29, 5: 30, 6: 30,
                7: 29, 8: 29, 9: 28, 10: 27, 11: 26, 12: 25}
    humidity_map = {1: 70, 2: 68, 3: 65, 4: 70, 5: 75, 6: 80,
                    7: 82, 8: 81, 9: 80, 10: 78, 11: 75, 12: 72}

    df['rainfall'] = df['month_num'].map(rainfall_map)
    df['temperature'] = df['month_num'].map(temp_map)
    df['humidity'] = df['month_num'].map(humidity_map)

    feature_cols = ['prev_cases_1', 'prev_cases_2', 'prev_cases_3', 'prev_cases_4', 'prev_cases_5', 'prev_cases_6', 'ma_3', 'ma_6', 'month_num', 'rainfall', 'temperature', 'humidity']
    X = df[feature_cols].fillna(0)
    y = df['dengue_cases']

    valid_idx = y > 0
    X = X[valid_idx]
    y = y[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, 
                         max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(X_train, y_train)

    return model, df, feature_cols

model, df, feature_cols = load_and_train_model()

# ============================================
# FRONTEND: Mobile-Optimized Interface
# ============================================

st.markdown("# 🦟 Dengue Forecast")
st.markdown("### Quick dengue prediction")

st.divider()

# ============================================
# INPUT SECTION - MOBILE OPTIMIZED
# ============================================
st.markdown("## 📍 Enter Information")

# Region & Month (full width on mobile)
regions = sorted(df['region'].unique())
region = st.selectbox("Select Region", regions)

month_names = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
month = st.selectbox("Select Month", month_names, index=5)
month_num = month_names.index(month) + 1

# Recent Cases
st.markdown("### 📈 Recent Cases (last 3 months)")
prev1 = st.number_input("Last month", 0, 5000, 300)
prev2 = st.number_input("2 months ago", 0, 5000, 250)
prev3 = st.number_input("3 months ago", 0, 5000, 200)

# Weather
st.markdown("### 🌧️ Weather Forecast")
rainfall = st.number_input("Rainfall (mm)", 0, 500, 150)
temperature = st.number_input("Temperature (°C)", 15, 35, 28)
humidity = st.number_input("Humidity (%)", 40, 100, 75)

st.divider()

# ============================================
# PREDICTION BUTTON
# ============================================
if st.button("🔮 Get Forecast", use_container_width=True, type="primary"):
    prev4 = prev2 * 0.95
    prev5 = prev3 * 0.90
    prev6 = prev3 * 0.85
    
    ma3 = (prev1 + prev2 + prev3) / 3
    ma6 = (prev1 + prev2 + prev3 + prev4 + prev5 + prev6) / 6
    
    input_data = pd.DataFrame({
        'prev_cases_1': [prev1],
        'prev_cases_2': [prev2],
        'prev_cases_3': [prev3],
        'prev_cases_4': [prev4],
        'prev_cases_5': [prev5],
        'prev_cases_6': [prev6],
        'ma_3': [ma3],
        'ma_6': [ma6],
        'month_num': [month_num],
        'rainfall': [rainfall],
        'temperature': [temperature],
        'humidity': [humidity]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)
    
    st.markdown("---")
    st.markdown("## 📊 Forecast Result")
    
    # Results in stacked format for mobile
    st.metric("📌 Region", region)
    st.metric("📅 Month", month)
    st.metric("🦟 Predicted Cases", f"{prediction:.0f}")
    
    lower = prediction * 0.9
    upper = prediction * 1.1
    st.metric("📈 Expected Range", f"{lower:.0f} - {upper:.0f}")
    
    # Risk level
    if rainfall > 300:
        risk = "🔴 HIGH RISK"
        color = "#e74c3c"
    elif rainfall < 50:
        risk = "🟢 LOW RISK"
        color = "#27ae60"
    else:
        risk = "🟡 MODERATE RISK"
        color = "#f39c12"
    
    st.metric("⚠️ Risk Level", risk)
    
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    if prediction > 400:
        st.error("⚠️ High dengue risk! Increase vector control activities.")
    elif prediction > 250:
        st.warning("⚠️ Moderate risk. Prepare health resources.")
    else:
        st.success("✅ Low risk. Continue routine surveillance.")

st.divider()
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>Dengue Forecast System | USTP</p>", unsafe_allow_html=True)
