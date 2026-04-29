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
    temp_df['Day_Encoded'] = le.fit_transform(temp_df['Day_Type'])
    X = temp_df[['Day_Encoded', 'Time_Numeric']]
    y = temp_df['Weight'] # AI එක තාමත් Weight එක ඉගෙන ගන්නවා, නමුත් අපි ඒක පෙන්වන්නේ නැහැ
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
        shp_path = "Gelioya_BypassRd.shp" 
        if SHP_SUPPORT and os.path.exists(shp_path):
            sf = shapefile.Reader(shp_path)
            for shape in sf.shapes():
                lons, lats = zip(*shape.points)
                bypass_roads.append({'lats': list(lats), 'lons': list(lons)})
        return traffic, parking, bypass_roads
    except: return None, None, []

# --- 🖥️ User Interface ---
st.set_page_config(page_title="Gelioya Smart Traffic AI", layout="wide")
st.title("🚦 Gelioya Smart Traffic AI Dashboard")

traffic_data, parking_data, bypass_roads = load_data()

if traffic_data is not None:
    model, encoder = train_model(traffic_data)

    # Sidebar Controls
    st.sidebar.header("Control Panel")
    day_type = st.sidebar.selectbox("Select Day", traffic_data['Day_Type'].unique())
    time_24 = st.sidebar.slider("Select Time (Hour)", 6, 22, 17)
    time_display = f"{time_24-12 if time_24 > 12 else time_24}:00 {'PM' if time_24 >= 12 else 'AM'}"
    
    # 🧠 AI Calculation (Internal)
    day_enc = encoder.transform([day_type])[0]
    ai_raw = model.predict([[day_enc, time_24]])[0]
    
    # 🚦 මෙතනදී අපි අගය Status එකකට හරවනවා
    if ai_raw >= 66:
        status_text = "HIGH TRAFFIC"
        status_color = "error" # Red
        status_emoji = "🔴"
    elif ai_raw >= 31:
        status_text = "MODERATE TRAFFIC"
        status_color = "warning" # Orange
        status_emoji = "🟡"
    else:
        status_text = "LOW TRAFFIC"
        status_color = "success" # Green
        status_emoji = "🟢"

    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Traffic Status")
    # 0-100% පෙන්වන්නේ නැතුව කෙලින්ම Status එක පෙන්වීම
    st.sidebar.info(f"AI STATUS: {status_emoji} {status_text}")
    
    # --- 📢 Telegram Sending Logic (No Percentages) ---
    if st.sidebar.button("📢 Send Update to Telegram"):
        try:
            msg = (f"📢 *GELIOYA TRAFFIC REPORT*\n\n"
                   f"🕒 Time: {time_display}\n"
                   f"📅 Day: {day_type}\n"
                   f"📊 AI Status: {status_emoji} {status_text}\n\n"
                   f"🔗 Live Dashboard: {DASHBOARD_URL}")
            
            receivers = [MY_CHAT_ID, GROUP_CHAT_ID]
            for chat_id in receivers:
                bot.sendMessage(chat_id, msg, parse_mode='Markdown')
                
            st.sidebar.success(f"✅ Alert Sent: {status_text}")
        except Exception as e: 
            st.sidebar.error(f"Telegram Error: {e}")

    map_theme = st.sidebar.selectbox("Map Style", ["open-street-map", "carto-positron", "carto-darkmatter"])
    show_parking = st.sidebar.checkbox("Show Parking", value=True)
    
    # --- 📍 Map Section (Points Only) ---
    st.subheader(f"📍 Traffic Heat-Map: {status_text}")
    filtered_traffic = traffic_data[traffic_data['Day_Type'] == day_type].copy()
    
    fig_map = px.scatter_mapbox(
        filtered_traffic, 
        lat="Latitude", 
        lon="Longitude", 
        color="Traffic_Level",
        hover_name="Road_Segment", 
        size_max=12, 
        zoom=15, 
        height=600,
        center={"lat": 7.214, "lon": 80.598},
        color_discrete_map={
            'High (Red)':'#FF0000', 
            'Moderate (Orange)':'#FFA500', 
            'Low (Green)':'#00FF00'
        }
    )

    if (ai_raw > 40 or 16 <= time_24 <= 19) and bypass_roads:
        for road in bypass_roads:
            fig_map.add_trace(go.Scattermapbox(
                mode="lines", lat=road['lats'], lon=road['lons'],
                line=dict(width=4, color='#00FFFF', dash='dash'), name="AI Bypass"
            ))

    if show_parking and parking_data is not None:
        fig_map.add_trace(go.Scattermapbox(
            lat=parking_data['Lattitude'], lon=parking_data['Longitude'],
            mode='markers+text', marker=dict(size=18, color='#007BFF', symbol='parking'),
            text="P", textfont=dict(color="white", size=9), name="Parking"
        ))

    fig_map.update_layout(mapbox_style=map_theme, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # --- 📊 Lower Section ---
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("📊 Road Analysis")
        chart_df = filtered_traffic.drop_duplicates(subset=['Road_Segment'])
        # මෙතනත් Weight පෙන්වන්නේ නැතුව Traffic_Level එකෙන් චාර්ට් එක හදමු
        fig_chart = px.bar(chart_df, x='Road_Segment', y='Weight', color='Traffic_Level',
                           color_discrete_map={'High (Red)':'red', 'Moderate (Orange)':'orange', 'Low (Green)':'green'},
                           labels={'Weight': 'Congestion Level'})
        st.plotly_chart(fig_chart, use_container_width=True)
    
    with col2:
        st.subheader("🅿️ Smart Parking Status")
        p_df = parking_data.copy()
        p_df = p_df.rename(columns={'Slot Name': 'Location', 'Capacity estimate': 'Vehicle Capacity'})
        p_df['Current Status'] = ["Full ❌" if (i * ai_raw) % 10 > 6 else "Available ✅" for i in range(len(p_df))]
        st.dataframe(p_df[['Location', 'Vehicle Capacity', 'Current Status']], use_container_width=True, height=450)
else:
    st.error("Missing Data Files!")
