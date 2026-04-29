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
GROUP_CHAT_ID = '-1003967636037' # ඔයාගේ අලුත් Supergroup ID එක
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
    
    day_enc = encoder.transform([day_type])[0]
    ai_pred = model.predict([[day_enc, time_24]])[0]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Community Broadcast")
    st.sidebar.info(f"AI Prediction: {ai_pred:.1f}%")
    
    if st.sidebar.button("📢 Send Update to Telegram"):
        try:
            status = "🔴 HIGH" if ai_pred > 70 else "🟡 MODERATE" if ai_pred > 40 else "🟢 LOW"
            msg = (f"📢 *GELIOYA TRAFFIC REPORT*\n\n"
                   f"🕒 Time: {time_display}\n"
                   f"📅 Day: {day_type}\n"
                   f"📊 AI Status: {status}\n"
                   f"📈 Congestion: {ai_pred:.1f}%\n\n"
                   f"🔗 View Map: {DASHBOARD_URL}")
            # Personal සහ Group දෙකටම යවනවා
            bot.sendMessage(MY_CHAT_ID, msg, parse_mode='Markdown')
            bot.sendMessage(GROUP_CHAT_ID, msg, parse_mode='Markdown')
            st.sidebar.success("✅ Alert Sent to Group & Admin!")
        except: st.sidebar.error("Telegram Error!")

    map_theme = st.sidebar.selectbox("Map Style", ["open-street-map", "carto-positron", "carto-darkmatter"])
    show_parking = st.sidebar.checkbox("Show Parking", value=True)
    
    # --- 📍 Map Section (Updated with Traffic Lines) ---
    st.subheader(f"📍 Traffic Forecast & Routing: {day_type} at {time_display}")
    filtered_traffic = traffic_data[traffic_data['Day_Type'] == day_type].copy()
    
    # මුලින්ම හිස් map එකක් හදනවා
    fig_map = go.Figure()

    # සෑම පාරක්ම (Road Segment) වෙන වෙනම Lines ලෙස ඇඳීම
    for road_name in filtered_traffic['Road_Segment'].unique():
        road_subset = filtered_traffic[filtered_traffic['Road_Segment'] == road_name]
        
        # ට්‍රැෆික් වර්ණය තීරණය කිරීම (පළමු පේළිය අනුව)
        t_level = road_subset['Traffic_Level'].iloc[0]
        line_color = '#FF0000' if 'High' in t_level else '#FFA500' if 'Moderate' in t_level else '#00FF00'
        
        # පාරවල් රේඛා ලෙස එකතු කිරීම
        fig_map.add_trace(go.Scattermapbox(
            mode="lines+markers",
            lat=road_subset['Latitude'],
            lon=road_subset['Longitude'],
            line=dict(width=5, color=line_color),
            marker=dict(size=8, color=line_color),
            name=road_name,
            hoverinfo='text',
            text=f"{road_name}: {t_level}"
        ))

    # Bypass Roads (AI මගින් යෝජනා කරන විට)
    if (ai_pred > 40 or 16 <= time_24 <= 19) and bypass_roads:
        for road in bypass_roads:
            fig_map.add_trace(go.Scattermapbox(
                mode="lines", lat=road['lats'], lon=road['lons'],
                line=dict(width=4, color='#00FFFF', dash='dash'), name="AI Bypass Route"
            ))

    # Smart Parking Markers
    if show_parking and parking_data is not None:
        fig_map.add_trace(go.Scattermapbox(
            lat=parking_data['Lattitude'], 
            lon=parking_data['Longitude'],
            mode='markers+text',
            marker=dict(size=18, color='#007BFF', symbol='parking'),
            text="P",
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hoverinfo='text',
            hovertext=parking_data['Slot Name'],
            name="Parking"
        ))

    fig_map.update_layout(
        mapbox=dict(
            style=map_theme,
            center={"lat": 7.213, "lon": 80.593},
            zoom=14.5
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600
    )
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
        p_df = parking_data.copy()
        p_df = p_df.rename(columns={'Slot Name': 'Location', 'Capacity estimate': 'Vehicle Capacity'})

        def get_current_status(prediction, index):
            threshold = 50 
            if prediction > threshold:
                return "Full ❌" if (index * prediction) % 10 > 3 else "Available ✅"
            else:
                return "Full ❌" if (index * prediction) % 10 > 8 else "Available ✅"
        
        p_df['Current Status'] = [get_current_status(ai_pred, i) for i in range(len(p_df))]
        st.dataframe(p_df[['Location', 'Vehicle Capacity', 'Current Status']], use_container_width=True, height=450)
else:
    st.error("Missing Data Files!")
