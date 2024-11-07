import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ตั้งค่าหน้าเว็บของ Streamlit
st.set_page_config(
    page_title="ทำนายรางวัลที่ 1 ของหวยไทย",
    page_icon="🎉",
    layout="centered",  # ใช้ layout แบบ 'centered' หรือ 'wide'
    initial_sidebar_state="expanded"  # กำหนดสถานะของ Sidebar
)

# Set background image URL
background_image_url = "https://img.freepik.com/free-photo/painting-mountain-lake-with-mountain-background_188544-9126.jpg?t=st=1730980649~exp=1730984249~hmac=e53c8de6cf9711b8c39b2daf4af1ec986daa4885c8686345ab09a8d94e0713a1&w=2000"

# Set desired colors
text_color = "#FFFFFF"  # Text color
result_bg_color = "#F5FFFA"  # Result background color

# Apply CSS for background and text colors
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        height: 100vh;
    }}
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
day = st.selectbox("เลือกงวด:", [1, 16], help="เลือกวันของงวด (1 หรือ 16)")

# เลือกค่า lag (ค่าของรางวัลที่ผ่านมาที่ต้องใส่)
prize_1st_lag1 = st.number_input("กรุณากรอกค่ารางวัลที่ 1 ของงวดที่แล้ว:", min_value=0, help="กรอกค่ารางวัลที่ 1 ของงวดก่อนหน้า")
prize_1st_lag2 = st.number_input("กรุณากรอกค่ารางวัลที่ 1 ของงวดก่อนหน้านั้น:", min_value=0, help="กรอกค่ารางวัลที่ 1 ของงวดก่อนหน้านั้น")

# ทำนายผล
if st.button("ทำนาย 🎯"):
    input_data = pd.DataFrame([[year, month, day, prize_1st_lag1, prize_1st_lag2]], 
                              columns=['year', 'month', 'day', 'prize_1st_lag1', 'prize_1st_lag2'])
    prediction = model_rf.predict(input_data)[0]
    st.success(f"การทำนายรางวัลที่ 1 สำหรับวันที่ {day} เดือน {month} ปี {year} คือ: **{int(prediction)}**")

# เพิ่มส่วนแสดงข้อมูลประเมินผล
st.markdown("### ประเมินผลของโมเดล:")

# ประเมินผลโมเดล
X_test = data[['year', 'month', 'day', 'prize_1st_lag1', 'prize_1st_lag2']]
y_test = data['prize_1st']
y_pred = model_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# แสดงกราฟการกระจาย (Optional)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted Values")

st.pyplot(fig)
