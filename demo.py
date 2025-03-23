import streamlit as st
import pip
pip.main(['install', '-r', 'tensorflow'])

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Tải mô hình 3 lớp ẩn
try:
    model_3_hidden = load_model('model_3_hidden.h5')
except FileNotFoundError:
    st.error("Không tìm thấy file 'model_3_hidden.h5'. Vui lòng đảm bảo file này tồn tại trong thư mục hiện tại.")
    st.stop()

# Phạm vi giá trị của các đặc trưng (ước lượng)
feature_ranges = {
    'Gender': (0, 1),
    'Age': (4, 96),
    'ALT': (5.0, 165.08),
    'AST': (5.0, 85.9),
    'CHO': (2.0, 8.41),
    'CL': (90.0, 110.0),
    'CRE': (45.05, 587.18),
    'EGFR': (9.63, 120.0),
    'GLU': (3.45, 28.65),
    'HDL': (0.5, 2.0),
    'HDL1': (0.5, 2.0),
    'LDL': (1.0, 5.0),
    'NA': (130.0, 145.0),
    'TRI': (0.5, 7.16)
}


# Hàm chuẩn hóa thủ công
def manual_scaling(input_data, feature_ranges):
    scaled_data = {}
    for feature, value in input_data.items():
        min_val, max_val = feature_ranges[feature]
        # Chuẩn hóa: (x - min) / (max - min)
        scaled_value = (value - min_val) / (max_val - min_val)
        # Giới hạn giá trị trong khoảng [0, 1]
        scaled_value = max(0, min(1, scaled_value))
        scaled_data[feature] = scaled_value
    return scaled_data


# Hàm dự đoán Diabetes
def predict_diabetes(input_data, model, feature_ranges):
    # Chuẩn hóa thủ công
    scaled_data = manual_scaling(input_data, feature_ranges)
    # Chuyển dữ liệu thành DataFrame
    input_df = pd.DataFrame([scaled_data],
                            columns=['Gender', 'Age', 'ALT', 'AST', 'CHO', 'CL', 'CRE', 'EGFR', 'GLU', 'HDL', 'HDL1',
                                     'LDL', 'NA', 'TRI'])
    # Dự đoán
    prediction = model.predict(input_df, verbose=0)
    probability = prediction[0][0]
    predicted_class = 1 if probability > 0.5 else 0
    return predicted_class, probability


# Giao diện Streamlit
st.title("Dự đoán bệnh tiểu đường với mô hình ANN (3 lớp ẩn)")
st.write("Nhập thông tin bệnh nhân để dự đoán nguy cơ tiểu đường. Điền đầy đủ các chỉ số xét nghiệm bên dưới:")

# Form nhập liệu
st.subheader("Thông tin bệnh nhân")
gender = st.selectbox("Giới tính", [0, 1], format_func=lambda x: "Nam" if x == 0 else "Nữ")
age = st.number_input("Tuổi", min_value=0, max_value=120, value=65, step=1)
alt = st.number_input("ALT (U/L)", min_value=0.0, value=12.31, step=0.01)
ast = st.number_input("AST (U/L)", min_value=0.0, value=18.45, step=0.01)
cho = st.number_input("Cholesterol (mmol/L)", min_value=0.0, value=5.12, step=0.01)
cl = st.number_input("Chloride (mmol/L)", min_value=0.0, value=99.45, step=0.01)
cre = st.number_input("Creatinine (µmol/L)", min_value=0.0, value=88.14, step=0.01)
egfr = st.number_input("EGFR (mL/min/1.73m²)", min_value=0.0, value=85.87, step=0.01)
glu = st.number_input("Glucose (mmol/L)", min_value=0.0, value=10.31, step=0.01)
hdl = st.number_input("HDL (mmol/L)", min_value=0.0, value=1.23, step=0.01)
hdl1 = st.number_input("HDL1 (mmol/L)", min_value=0.0, value=1.21, step=0.01)
ldl = st.number_input("LDL (mmol/L)", min_value=0.0, value=3.45, step=0.01)
na = st.number_input("Sodium (mmol/L)", min_value=0.0, value=138.45, step=0.01)
tri = st.number_input("Triglycerides (mmol/L)", min_value=0.0, value=2.45, step=0.01)

# Kiểm tra dữ liệu đầu vào
if st.button("Dự đoán"):
    if any(val < 0 for val in [alt, ast, cho, cl, cre, egfr, glu, hdl, hdl1, ldl, na, tri]):
        st.error("Các chỉ số xét nghiệm không được âm. Vui lòng kiểm tra lại!")
    else:
        input_data = {
            'Gender': gender,
            'Age': age,
            'ALT': alt,
            'AST': ast,
            'CHO': cho,
            'CL': cl,
            'CRE': cre,
            'EGFR': egfr,
            'GLU': glu,
            'HDL': hdl,
            'HDL1': hdl1,
            'LDL': ldl,
            'NA': na,
            'TRI': tri
        }

        pred, prob = predict_diabetes(input_data, model_3_hidden, feature_ranges)
        st.subheader("Kết quả dự đoán")
        st.write(f"**Dự đoán:** {'Có tiểu đường' if pred == 1 else 'Không có tiểu đường'}")
        st.write(f"**Xác suất:** {prob:.4f}")

        # Thêm thông tin bổ sung
        if pred == 1:
            st.warning("Bệnh nhân có nguy cơ cao bị tiểu đường. Đề nghị tham khảo ý kiến bác sĩ để kiểm tra thêm.")
            if glu >= 7.0:
                st.write(
                    f"- **Lưu ý:** Chỉ số Glucose ({glu} mmol/L) cao hơn ngưỡng bình thường (≥7.0 mmol/L), có thể là dấu hiệu của tiểu đường.")
            if tri >= 1.7:
                st.write(
                    f"- **Lưu ý:** Chỉ số Triglycerides ({tri} mmol/L) cao hơn ngưỡng bình thường (≥1.7 mmol/L), có thể liên quan đến nguy cơ tim mạch.")
        else:
            st.success(
                "Bệnh nhân có nguy cơ thấp bị tiểu đường. Tuy nhiên, nên duy trì lối sống lành mạnh và kiểm tra định kỳ.")