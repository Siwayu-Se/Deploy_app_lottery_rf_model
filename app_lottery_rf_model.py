%%writefile app_lottery_lr_model.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os

# ตั้งค่าหน้าเว็บของ Streamlit
st.set_page_config(
    page_title="ทำนายรางวัลที่ 1 ของหวยไทย (Linear Regression)",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# โหลดโมเดลจาก Google Drive
@st.cache_resource
def load_model_from_gdrive():
    url = 'https://drive.google.com/uc?id=1G8bYPU7e8w5I32FgvWPXjR6v1e5ylRxC'
    output = 'model_lr.pkl'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    with open(output, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model_from_gdrive()

# ส่วนของแอป Streamlit
st.title("📈 ทำนายรางวัลที่ 1 ของหวยไทย (Linear Regression)")
st.write("แอปนี้ใช้โมเดล Linear Regression ที่ฝึกมาแล้วในการทำนายรางวัลที่ 1 โดยไม่ใช้ฐานข้อมูล")

st.markdown("""
### วิธีการใช้งาน:
1. เลือกปี เดือน และวันของงวด
2. กรอกค่ารางวัลที่ 1 ของงวดที่แล้ว และงวดก่อนหน้านั้น
3. กดปุ่ม 'ทำนาย' เพื่อดูผลลัพธ์
""")

# อินพุตจากผู้ใช้
year = st.number_input("ปี (เช่น 2025):", min_value=2000, max_value=2100, value=2025)
month = st.number_input("เดือน (1-12):", min_value=1, max_value=12, value=4)
day = st.selectbox("วันของงวด:", [1, 16])

prize_1st_lag1 = st.number_input("รางวัลที่ 1 ของงวดก่อนหน้า:", min_value=0)
prize_1st_lag2 = st.number_input("รางวัลที่ 1 ของสองงวดก่อน:", min_value=0)

# ทำนาย
if st.button("ทำนาย 🎯"):
    input_df = pd.DataFrame([[year, month, day, prize_1st_lag1, prize_1st_lag2]],
                            columns=['year', 'month', 'day', 'prize_1st_lag1', 'prize_1st_lag2'])
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 การทำนายรางวัลที่ 1 สำหรับ {day}/{month}/{year} คือ: **{int(prediction)}**")
