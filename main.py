import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import telepot
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 🤖 Telegram Configuration ---
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
    
    # 🧠 AI Prediction Logic (Fixed Thresholds)
    day_enc = encoder.transform([day_type])[0]
    ai_raw = model.predict([[day_enc, time_24]])[0]
    
    # CSV එකේ Weights වලට අනුව නිවැරදි Status එක තේරීම
    if ai_raw >= 66:
        status_text, status_emoji, status_col = "HIGH TRAFFIC", "🔴", "red"
    elif ai_raw >= 31:
        status_text, status_emoji, status_col = "MODERATE TRAFFIC", "🟡", "orange"
    else:
        status_text, status_emoji, status_col = "LOW TRAFFIC", "🟢", "green"

    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Traffic Status")
    # සංඛ්‍යා පෙන්වන්නේ නැතිව ලස්සනට Status එක පෙන්වීම
    st.sidebar.info(f"AI STATUS: {status_emoji} {status_text}")
    
    # --- 📢 Telegram Sending Logic ---
    if st.sidebar.button("📢 Send Update to Telegram"):
        try:
            msg = (f"📢 *GELIOYA TRAFFIC REPORT*\n\n"
                   f"🕒 Time: {time_display}\n"
                   f"📅 Day: {day_type}\n"
                   f"📊 AI Status: {status_emoji} {status_text}\n\n"
                   f"🔗 Live Dashboard: {DASHBOARD_URL}")
            
            for chat_id in [MY_CHAT_ID, GROUP_CHAT_ID]:
                bot.sendMessage(chat_id, msg, parse_mode='Markdown')
            st.sidebar.success(f"✅ Sent: {status_text}")
        except Exception as e: 
            st.sidebar.error(f"Telegram Error: {e}")

    # --- 📍 Map Section (Old Favorite Point View) ---
    st.subheader(f"📍 Live AI Traffic Map: {status_text}")
    filtered_traffic = traffic_data[traffic_data['Day_Type'] == day_type].copy()
    
    fig_map = px.scatter_mapbox(
        filtered_traffic, lat="Latitude", lon="Longitude", color="Traffic_Level",
        hover_name="Road_Segment", size_max=12, zoom=15, height=550,
        center={"lat": 7.214, "lon": 80.598},
        color_discrete_map={'High (Red)':'#FF0000', 'Moderate (Orange)':'#FFA500', 'Low (Green)':'#00FF00'}
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # --- 📊 Graph Section (AI Updated) ---
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("📊 AI Congestion Analysis")
        # පාරවල් වල ලිස්ට් එකට AI අගය දාමු
        roads = filtered_traffic['Road_Segment'].unique()
        fig_chart = go.Figure(go.Bar(
            x=roads, 
            y=[ai_raw] * len(roads), 
            marker_color=status_col,
            text=status_text,
            textposition='auto'
        ))
        fig_chart.update_layout(yaxis_title="Congestion Level", yaxis_range=[0, 100], height=400)
        st.plotly_chart(fig_chart, use_container_width=True)
    
    with col2:
        st.subheader("🅿️ Smart Parking Status")
        if parking_data is not None:
            p_df = parking_data.copy().rename(columns={'Slot Name': 'Location', 'Capacity estimate': 'Vehicle Capacity'})
            p_df['Current Status'] = ["Full ❌" if (i * ai_raw) % 10 > 6 else "Available ✅" for i in range(len(p_df))]
            st.dataframe(p_df[['Location', 'Vehicle Capacity', 'Current Status']], use_container_width=True, height=400)
else:
    st.error("Missing Data Files!")
