import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import telepot
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 🤖 Telegram Bot Configuration ---
BOT_TOKEN = '8602459951:AAEQif4JnTDQjl7gvnVGv0pEw-tXn3b6DKs'
MY_CHAT_ID = '5365836212'
GROUP_CHAT_ID = '-1003967636037'
bot = telepot.Bot(BOT_TOKEN)
DASHBOARD_URL = "https://gelioya-traffic-ai.streamlit.app"

# --- 🧠 AI Training Function with Practical Logic ---
@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    temp_df = df.copy()
    
    # Time Numeric බවට පත් කිරීම [cite: 10]
    time_col = [c for c in temp_df.columns if 'Time' in c][0]
    def extract_hour(time_str):
        try:
            hour = "".join(filter(str.isdigit, str(time_str).split(':')[0]))
            return int(hour) if hour else 0
        except: return 0
    temp_df['Time_Numeric'] = temp_df[time_col].apply(extract_hour)
    
    # 🚨 Practical Time Logic (School/Office/Tuition hours) [cite: 11]
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
    model = RandomForestRegressor(n_estimators=100, random_state=42) # [cite: 9]
    model.fit(X.values, y)
    return model, le

@st.cache_data
def load_data():
    try:
        # Data Acquisition [cite: 4, 5]
        traffic = pd.read_csv('Weekly_Traffic_Simulation.csv', encoding='latin1')
        parking = pd.read_csv('Parking Slot.csv', encoding='latin1')
        traffic.columns = traffic.columns.str.strip()
        parking.columns = parking.columns.str.strip()
        return traffic, parking
    except: return None, None

# --- 🖥️ User Interface ---
st.set_page_config(page_title="Gelioya Smart Traffic AI", layout="wide")
st.title("🚦 Gelioya Smart Traffic AI Dashboard")

traffic_data, parking_data = load_data()

if traffic_data is not None:
    model, encoder = train_model(traffic_data)

    # Sidebar Controls
    st.sidebar.header("Control Panel")
    day_type = st.sidebar.selectbox("Select Day", traffic_data['Day_Type'].unique())
    time_24 = st.sidebar.slider("Select Time (Hour)", 6, 22, 17)
    time_display = f"{time_24-12 if time_24 > 12 else time_24}:00 {'PM' if time_24 >= 12 else 'AM'}"
    map_theme = st.sidebar.selectbox("Map Style", ["open-street-map", "carto-positron", "carto-darkmatter"])
    show_parking = st.sidebar.checkbox("Show Parking", value=True)
    
    day_enc = encoder.transform([day_type])[0]
    ai_pred = model.predict([[day_enc, time_24]])[0]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Community Broadcast")
    st.sidebar.info(f"AI Prediction: {ai_pred:.1f}%")
    
    # Broadcast System [cite: 18]
    if st.sidebar.button("📢 Send Update to Telegram"):
        try:
            status = "🔴 HIGH" if ai_pred > 70 else "🟡 MODERATE" if ai_pred > 40 else "🟢 LOW"
            msg = f"📢 GELIOYA TRAFFIC REPORT\n\n🕒 Time: {time_display}\n📅 Day: {day_type}\n📊 AI Status: {status}\n📈 Congestion: {ai_pred:.1f}%\n\n🔗 {DASHBOARD_URL}"
            for chat_id in [MY_CHAT_ID, GROUP_CHAT_ID]:
                bot.sendMessage(chat_id, msg, parse_mode='Markdown')
            st.sidebar.success("✅ Alert Sent!")
        except Exception as e: st.sidebar.error(f"Error: {e}")

    # --- 📍 Map Section (Plotly Mapbox) [cite: 14] ---
    st.subheader(f"📍 Traffic Forecast & Routing: {day_type} at {time_display}")
    filtered_traffic = traffic_data[traffic_data['Day_Type'] == day_type].copy()
    
    fig_map = px.scatter_mapbox(
        filtered_traffic, lat="Latitude", lon="Longitude", color="Traffic_Level",
        hover_name="Road_Segment", size_max=15, zoom=14.5, height=600,
        center={"lat": 7.213, "lon": 80.593},
        color_discrete_map={'High (Red)':'#FF0000', 'Moderate (Orange)':'#FFA500', 'Low (Green)':'#00FF00'}
    )

    # --- 🛣️ Manual Bypass Road Logic  ---
    # මෙතනට ඔයාගේ Bypass පාරේ නියම Coordinates ටික දාන්න (දැනට මම Sample දාලා තියෙන්නේ)
    bypass_lat = [7.212, 7.215, 7.218, 7.221] 
    bypass_lon = [80.588, 80.590, 80.592, 80.595]
    
    is_peak = (7 <= time_24 <= 8) or (12 <= time_24 <= 14) or (16 <= time_24 <= 19) or (day_type == 'Saturday')
    
    if ai_pred > 40 or is_peak:
        fig_map.add_trace(go.Scattermapbox(
            mode="lines+markers",
            lat=bypass_lat, 
            lon=bypass_lon,
            line=dict(width=5, color='#00FFFF'), 
            name="AI Recommended Bypass",
            text="Recommended Route to Avoid Congestion"
        ))

    # Infrastructure Visualization (Parking) [cite: 5]
    if show_parking and parking_data is not None:
        fig_map.add_trace(go.Scattermapbox(
            lat=parking_data['Lattitude'], lon=parking_data['Longitude'],
            mode='markers+text', marker=dict(size=15, color='#007BFF'),
            text="P", textposition="middle center", textfont=dict(size=10, color="white"),
            hoverinfo='text', hovertext=parking_data['Slot Name'], name="Parking"
        ))

    fig_map.update_layout(mapbox_style=map_theme, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # --- 📊 Lower Section ---
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("📊 Congestion Analysis")
        fig_chart = px.bar(filtered_traffic, x='Road_Segment', y='Weight', color='Traffic_Level',
                           color_discrete_map={'High (Red)':'red', 'Moderate (Orange)':'orange', 'Low (Green)':'green'})
        st.plotly_chart(fig_chart, use_container_width=True)
    
    with col2:
        st.subheader("🅿️ Smart Parking Status")
        p_df = parking_data.copy().rename(columns={'Slot Name': 'Location', 'Capacity estimate': 'Vehicle Capacity'})
        # Dynamic Parking Logic
        p_df['Current Status'] = ["Full ❌" if (i * ai_pred) % 10 > (3 if ai_pred > 50 else 8) else "Available ✅" for i in range(len(p_df))]
        st.dataframe(p_df[['Location', 'Vehicle Capacity', 'Current Status']], use_container_width=True, height=450)
else:
    st.error("Missing Data Files!")
