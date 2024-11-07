import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ตั้งค่าหน้าเว็บของ Streamlit
st.set_page_config(
    page_title="ทำนายรางวัลที่ 1 ของหวยไทย",
    page_icon="🎉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Set desired colors
text_color = "#000000"  # Text color
result_bg_color = "#F5FFFA"  # Result background color

# Apply CSS for background and text colors
st.markdown(
    f"""
    <style>
    h1, h2, h3, p, div {{
        color: {text_color} !important;
    }}
    .result-container {{
        background-color: {result_bg_color};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        opacity: 0.9;
        border: 2px solid {text_color};
    }}
    div.stButton > button:first-child {{
        background-color: #FF5733; /* สีพื้นหลังของปุ่ม */
        color: #FFFFFF; /* สีของตัวหนังสือ */
        padding: 10px 20px;
        font-size: 18px;
        border-radius: 8px;
        border: 2px solid #C70039;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transition: 0.3s;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #C70039; /* สีเมื่อ hover บนปุ่ม */
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.5);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# โหลดโมเดลที่บันทึกไว้
@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model_rf = load_model()

# โหลดข้อมูลเพื่อใช้ในการทดสอบ
@st.cache_data
def load_data():
    data = pd.read_csv('lottery.csv')
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['prize_1st_lag1'] = data['prize_1st'].shift(1)
    data['prize_1st_lag2'] = data['prize_1st'].shift(2)
    data.dropna(inplace=True)
    return data

data = load_data()

# ส่วนของแอป Streamlit
st.title("🎉 ทำนายรางวัลที่ 1 ของหวยไทย 🎉")
st.write("แอปนี้ใช้ทำนายรางวัลที่ 1 ของหวยไทย โดยใช้ข้อมูลที่ผ่านมาและโมเดลที่ฝึกไว้")

# เพิ่มข้อความแนะนำ
st.markdown("""
### วิธีการใช้งาน:
1. เลือกปีและเดือนของงวดที่ต้องการทำนาย
2. กรอกค่ารางวัลที่ 1 ของงวดที่แล้วและงวดก่อนหน้านั้น
3. กดปุ่ม 'ทำนาย' เพื่อดูผลการทำนาย
""")

# เลือกปี, เดือน และงวดสำหรับทำนาย
year = st.selectbox("เลือกปี:", sorted(data['year'].unique()), help="เลือกปีที่ต้องการทำนายรางวัล")
month = st.selectbox("เลือกเดือน:", sorted(data['month'].unique()), help="เลือกเดือนที่ต้องการทำนายรางวัล")
day = st.selectbox
