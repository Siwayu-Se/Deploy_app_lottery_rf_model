import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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

# ระบุสีกรอบที่ต้องการ
border_color = "#FF5733"  # ระบุสีกรอบตามต้องการ เช่น สีแดง

# ทำนายผล
if st.button("ทำนาย 🎯"):
    input_data = pd.DataFrame([[year, month, day, prize_1st_lag1, prize_1st_lag2]], 
                              columns=['year', 'month', 'day', 'prize_1st_lag1', 'prize_1st_lag2'])
    prediction = model_rf.predict(input_data)[0]
    
    # สร้าง HTML สำหรับกรอบที่มีสีที่กำหนด
    prediction_text = f"การทำนายรางวัลที่ 1 สำหรับวันที่ {day} เดือน {month} ปี {year} คือ: **{int(prediction)}** บาท"
    
    # ครอบทุกผลลัพธ์ภายในกรอบเดียว
    st.markdown(f"""
    <div style="padding: 30px; border: 2px solid {border_color}; border-radius: 10px; background-color: #f0f0f0;">
        <h3>ผลการทำนาย:</h3>
        <h4>{prediction_text}</h4>
        
        <h3>ประเมินผลของโมเดล:</h3>
        <p>MAE (Mean Absolute Error): {mean_absolute_error(data['prize_1st'], model_rf.predict(data[['year', 'month', 'day', 'prize_1st_lag1', 'prize_1st_lag2']]))):.2f}</p>
        <p>RMSE (Root Mean Squared Error): {np.sqrt(mean_squared_error(data['prize_1st'], model_rf.predict(data[['year', 'month', 'day', 'prize_1st_lag1', 'prize_1st_lag2']])))):.2f}</p>
        
        <h3>กราฟการกระจาย:</h3>
        <img src="data:image/png;base64,{st.pyplot(fig)}" alt="scatter plot"/>
    </div>
    """, unsafe_allow_html=True)

# แสดงกราฟการกระจาย (Optional)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(data['prize_1st'], model_rf.predict(data[['year', 'month', 'day', 'prize_1st_lag1', 'prize_1st_lag2']]))
ax.plot([data['prize_1st'].min(), data['prize_1st'].max()], [data['prize_1st'].min(), data['prize_1st'].max()], color='red', lw=2)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted Values")

st.pyplot(fig)
