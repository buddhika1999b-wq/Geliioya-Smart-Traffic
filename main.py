import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import telepot
import time
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 🛠️ Library Import for Map Roads ---
try:
    import shapefile
    SHP_SUPPORT = True
except ImportError:
    SHP_SUPPORT = False

# --- 🤖 Telegram Bot Config & Multilingual Logic ---
BOT_TOKEN = '8602459951:AAEQif4JnTDQjl7gvnVGv0pEw-tXn3b6DKs'
MY_CHAT_ID = '5365836212' # ඔයාගේ Screenshot එකේ තිබුණ ID එක
bot = telepot.Bot(BOT_TOKEN)
user_languages = {}

# ඩෑෂ්බෝඩ් ලින්ක් එක (Presentation එකේදී මේක පාවිච්චි කරන්න පුළුවන්)
DASHBOARD_URL = "http://localhost:8501"

def handle_bot_messages(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    
    if content_type == 'text':
        text = msg['text'].lower()
        if text == '/start':
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text='English 🇬🇧', callback_data='en')],
                [InlineKeyboardButton(text='සිංහල 🇱🇰', callback_data='si')],
                [InlineKeyboardButton(text='தமிழ் 🇮🇳', callback_data='ta')]
            ])
            bot.sendMessage(chat_id, "Welcome! Please select your language / භාෂාව තෝරන්න:", reply_markup=keyboard)
        else:
            lang = user_languages.get(chat_id, 'en')
            if any(word in text for word in ['traffic', 'ට්‍රැෆික්', 'போக்குவரத்து']):
                kb = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text='🌍 Open Dashboard', url=DASHBOARD_URL)]
                ])
                res = {
                    'en': "📊 AI Forecast: Check real-time congestion scores on the dashboard.",
                    'si': "📊 AI පුරෝකථනය: වත්මන් තත්ත්වය දැනගැනීමට Dashboard එක බලන්න.",
                    'ta': "📊 AI முன்னறிவிப்பு: டாஷ்போர்டைப் பார்க்கவும்."
                }
                bot.sendMessage(chat_id, res[lang], reply_markup=kb)
            else:
                bot.sendMessage(chat_id, "Please use /start to select language or ask about 'Traffic'.")

    elif content_type == 'callback_query':
        query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
        user_languages[from_id] = query_data
        bot.answerCallbackQuery(query_id, text="Language Updated")
        bot.sendMessage(from_id, f"✅ Language set! Type 'Traffic' to get the live link.")

if 'bot_active' not in st.session_state:
    MessageLoop(bot, {'chat': handle_bot_messages, 'callback_query': handle_bot_messages}).run_as_thread()
    st.session_state['bot_active'] = True

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
st.set_page_config(page_title="Geliioya Smart Traffic AI", layout="wide")
st.title("🚦 Geliioya Smart Traffic AI & Interactive Bot")

traffic_data, parking_data, bypass_roads = load_data()

if traffic_data is not None:
    model, encoder = train_model(traffic_data)

    # Sidebar
    st.sidebar.header("Control Panel")
    day_type = st.sidebar.selectbox("Select Day", traffic_data['Day_Type'].unique())
    time_24 = st.sidebar.slider("Select Time (Hour)", 6, 22, 17)
    time_display = f"{time_24-12 if time_24 > 12 else time_24}:00 {'PM' if time_24 >= 12 else 'AM'}"
    
    day_enc = encoder.transform([day_type])[0]
    ai_pred = model.predict([[day_enc, time_24]])[0]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Community Broadcast")
    st.sidebar.info(f"AI Prediction: {ai_pred:.1f}%")
    
    # 📢 Telegram Alert Section
    if st.sidebar.button("📢 Send Update to Telegram"):
        try:
            status = "🔴 HIGH" if ai_pred > 70 else "🟡 MODERATE" if ai_pred > 40 else "🟢 LOW"
            msg = f"📢 *GELIOYA TRAFFIC REPORT*\n\n🕒 Time: {time_display}\n📅 Day: {day_type}\n📊 AI Status: {status}\n📈 Score: {ai_pred:.1f}%\n\n🚗 Click below to view the interactive map!"
            
            # Dashboard Link Button
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text='🚀 View Live Dashboard', url=DASHBOARD_URL)]
            ])
            
            bot.sendMessage(MY_CHAT_ID, msg, parse_mode='Markdown', reply_markup=kb)
            st.sidebar.success("Message Sent Successfully!")
        except Exception as e:
            st.sidebar.error(f"Telegram Error: Make sure you have messaged the bot first! (Error: {e})")

    map_theme = st.sidebar.selectbox("Map Style", ["open-street-map", "carto-positron", "carto-darkmatter"])
    show_parking = st.sidebar.checkbox("Show Parking", value=True)
    
    # Map Section
    st.subheader(f"📍 Traffic Simulation: {day_type} at {time_display}")
    filtered_traffic = traffic_data[traffic_data['Day_Type'] == day_type].copy()
    
    fig_map = px.scatter_map(
        filtered_traffic, lat="Latitude", lon="Longitude", color="Traffic_Level",
        hover_name="Road_Segment", size_max=15, zoom=14.5, height=600,
        color_discrete_map={'High (Red)':'#FF0000', 'Moderate (Orange)':'#FFA500', 'Low (Green)':'#00FF00'}
    )

    # Bypass Road Display Logic (50% threshold)
    if (ai_pred > 50 or 16 <= time_24 <= 19) and bypass_roads:
        for road in bypass_roads:
            fig_map.add_trace(go.Scattermap(mode="lines", lat=road['lats'], lon=road['lons'],
                line=dict(width=5, color='#00FFFF'), name="AI Bypass"))

    if show_parking and parking_data is not None:
        fig_map.add_trace(go.Scattermap(lat=parking_data['Lattitude'], lon=parking_data['Longitude'],
            mode='markers', marker=go.scattermap.Marker(size=14, color='#1E90FF', symbol='parking'),
            text=parking_data['Slot Name'], name="Parking"))

    fig_map.update_layout(map_style=map_theme, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # Graphs and Tables
    col1, col2 = st.columns([1, 1]) if show_parking else (st.container(), None)
    with col1:
        st.subheader("📊 Intensity Graph")
        fig_chart = px.bar(filtered_traffic, x='Road_Segment', y='Weight', color='Traffic_Level',
                           height=400, color_discrete_map={'High (Red)':'red', 'Moderate (Orange)':'orange', 'Low (Green)':'green'})
        st.plotly_chart(fig_chart, use_container_width=True)
    if show_parking and col2:
        with col2:
            st.subheader("🅿️ Parking Status")
            display_df = parking_data.copy()
            # Dynamic Parking Status based on AI Traffic Prediction
            display_df['Status'] = ["❌ Full" if ai_pred > 65 and i % 2 == 0 else "✅ Available" for i in range(len(display_df))]
            st.dataframe(display_df[['Slot Name', 'Capacity estimate', 'Status']], use_container_width=True, height=400)
else:
    st.error("Missing Data Files (CSV or Shapefiles)!")