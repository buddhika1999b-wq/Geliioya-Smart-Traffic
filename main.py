import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import telepot
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 🛠️ Library Import ---
try:
    import shapefile
    SHP_SUPPORT = True
except ImportError:
    SHP_SUPPORT = False

# --- 🤖 Telegram Bot Configuration ---
BOT_TOKEN = '8602459951:AAEQif4JnTDQjl7gvnVGv0pEw-tXn3b6DKs'
MY_CHAT_ID = '5365836212'
GROUP_CHAT_ID = '-1003967636037'
bot = telepot.Bot(BOT_TOKEN)
DASHBOARD_URL = "https://gelioya-traffic-ai.streamlit.app"

# --- 🧠 AI Training Function ---
@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    temp_df = df.copy()
    time_col = [c for c in temp_df.columns if 'Time' in c][0]
    def extract_hour(time_str):
        try:
            hour = "".join(filter(str.isdigit, str(time_str).split(':')[0]))
            return int(hour) if hour else 0
        except: return 0
    temp_df['Time_Numeric'] = temp_df[time_col].apply(extract_hour)
    
    def apply_custom_logic(row):
        h = row['Time_Numeric']
        d = row['Day_Type']
        if d != 'Sunday' and ((7 <= h <= 8) or (12 <= h <= 14)): return 85 
        if d != 'Sunday' and (16 <= h <= 19): return 90 
        if d == 'Saturday': return 80 
        return row['Weight']

    temp_df['Weight'] = temp_df.apply(apply_custom_logic, axis=1)
    temp_df['Day_Encoded'] = le.fit_transform(temp_df['Day_Type'])
    X = temp_df[['Day_Encoded', 'Time_Numeric']]
    y = temp_df['Weight']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.values, y)
    return model, le

@st.cache_data
def load_data():
    try:
        traffic = pd.read_csv('Weekly_Traffic_Simulation.csv', encoding='latin1')
        parking = pd.read_csv('Parking Slot.csv', encoding='latin1')
        traffic.columns = traffic.columns.str.strip()
        parking.columns = parking.columns.str.strip()
        
        bypass_roads = []
        # මෙන්න මෙතන තමයි වැදගත්ම දේ. GitHub එකේ තියෙන නමම (Case Sensitive) මෙතන තියෙන්න ඕනේ.
        shp_path = "Gelioya_BypassRd.shp" 
        
        if SHP_SUPPORT:
            if os.path.exists(shp_path):
                sf = shapefile.Reader(shp_path)
                for shape in sf.shapes():
                    lons, lats = zip(*shape.points)
                    bypass_roads.append({'lats': list(lats), 'lons': list(lons)})
            else:
                # ෆයිල් එක නැත්නම් sidebar එකේ පෙන්වන්න
                st.sidebar.error(f"❌ File Not Found: {shp_path}")
        return traffic, parking, bypass_roads
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, []

# --- 🖥️ User Interface ---
st.set_page_config(page_title="Gelioya Smart Traffic AI", layout="wide")
st.title("🚦 Gelioya Smart Traffic AI Dashboard")

traffic_data, parking_data, bypass_roads = load_data()

if traffic_data is not None:
    model, encoder = train_model(traffic_data)

    st.sidebar.header("Control Panel")
    day_type = st.sidebar.selectbox("Select Day", traffic_data['Day_Type'].unique())
    time_24 = st.sidebar.slider("Select Time (Hour)", 6, 22, 14)
    time_display = f"{time_24-12 if time_24 > 12 else time_24}:00 {'PM' if time_24 >= 12 else 'AM'}"
    map_theme = st.sidebar.selectbox("Map Style", ["open-street-map", "carto-positron", "carto-darkmatter"])
    show_parking = st.sidebar.checkbox("Show Parking", value=True)
    
    day_enc = encoder.transform([day_type])[0]
    ai_pred = model.predict([[day_enc, time_24]])[0]
    
    st.sidebar.info(f"AI Prediction: {ai_pred:.1f}%")

    # --- 📍 Map ---
    st.subheader(f"📍 Traffic Forecast & Routing: {day_type} at {time_display}")
    filtered_traffic = traffic_data[traffic_data['Day_Type'] == day_type].copy()
    
    fig_map = px.scatter_mapbox(
        filtered_traffic, lat="Latitude", lon="Longitude", color="Traffic_Level",
        hover_name="Road_Segment", size_max=15, zoom=14.5, height=600,
        center={"lat": 7.213, "lon": 80.593},
        color_discrete_map={'High (Red)':'#FF0000', 'Moderate (Orange)':'#FFA500', 'Low (Green)':'#00FF00'}
    )

    # Bypass පාරවල් අඳිනවා (SHP එකෙන් දත්ත ආවා නම් විතරයි)
    is_peak = (7 <= time_24 <= 8) or (12 <= time_24 <= 14) or (16 <= time_24 <= 19) or (day_type == 'Saturday')
    if (ai_pred > 40 or is_peak) and bypass_roads:
        for road in bypass_roads:
            fig_map.add_trace(go.Scattermapbox(
                mode="lines", lat=road['lats'], lon=road['lons'],
                line=dict(width=5, color='#00FFFF'), name="AI Bypass Active"
            ))

    if show_parking and parking_data is not None:
        fig_map.add_trace(go.Scattermapbox(
            lat=parking_data['Lattitude'], lon=parking_data['Longitude'],
            mode='markers', marker=dict(size=12, color='#007BFF'),
            hovertext=parking_data['Slot Name'], name="Parking"
        ))

    fig_map.update_layout(mapbox_style=map_theme, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # 📊 Charts
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("📊 Congestion Analysis")
        fig_chart = px.bar(filtered_traffic, x='Road_Segment', y='Weight', color='Traffic_Level',
                           color_discrete_map={'High (Red)':'red', 'Moderate (Orange)':'orange', 'Low (Green)':'green'})
        st.plotly_chart(fig_chart, use_container_width=True)
    with col2:
        st.subheader("🅿️ Smart Parking Status")
        st.dataframe(parking_data[['Slot Name', 'Capacity estimate']], use_container_width=True)
else:
    st.error("Missing Data Files!")
