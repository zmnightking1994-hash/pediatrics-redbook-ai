import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# 1. إعدادات الصفحة والواجهة
st.set_page_config(page_title="Pediatric AI Radiologist", layout="wide")

st.markdown("""
    <style>
    .stAlert { border-radius: 12px; border: 2px solid #3498db; }
    .dosage-card { 
        background-color: #ffffff; padding: 15px; border-radius: 10px; 
        border-right: 5px solid #27ae60; margin-bottom: 10px; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1); color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل الموديل
@st.cache_resource
def load_ai_model():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            return f"خطأ في التحميل: {e}"
    return None

ai_brain = load_ai_model()

st.title("🩺 مساعد طبيب الأطفال: Red Book + AI")
st.markdown("---")

# 3. بروتوكول Red Book 2024
RED_BOOK = {
    "first_line": "Amoxicillin (80–90 mg/kg per day in 2 divided doses)",
    "max": "4 g/day", "duration": "5–7 days", "page": "646"
}

# 4. رفع ومعالجة الصورة
uploaded_file = st.file_uploader("ارفع صورة الأشعة...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.2])
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    with col1:
        st.subheader("🔍 التحليل البصري")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)
        st.image(cv2.addWeighted(img, 0.6, heatmap, 0.4, 0), use_container_width=True)

    with col2:
        st.subheader("📋 نتيجة الذكاء الاصطناعي")
        if ai_brain is None:
            st.error("ملف model.h5 غير موجود.")
        elif isinstance(ai_brain, str):
            st.error(ai_brain)
        else:
            # تجربة أبعاد مختلفة لحل خطأ Kernel shape
            sizes_to_try = [(64, 64), (150, 150), (180, 180), (224, 224)]
            prediction = None
            
            for size in sizes_to_try:
                try:
                    img_resized = cv2.resize(img, size) / 255.0
                    img_input = np.expand_dims(img_resized, axis=0)
                    prediction = ai_brain.predict(img_input)[0][0]
                    break # إذا نجح التحليل نخرج من الحلقة
                except:
                    continue
            
            if prediction is not None:
                if prediction > 0.5:
                    st.error(f"🚨 إيجابي: احتمالية التهاب رئوي {prediction*100:.1f}%")
                    st.markdown(f"""
                    <div class="dosage-card"><strong>💊 العلاج (Red Book):</strong> {RED_BOOK['first_line']}</div>
                    <div class="dosage-card"><strong>⏱️ المدة:</strong> {RED_BOOK['duration']}</div>
                    <div class="dosage-card"><strong>📖 المرجع:</strong> صفحة {RED_BOOK['page']}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ سليم: الرئة طبيعية بنسبة {(1-prediction)*100:.1f}%")
                    st.balloons()
            else:
                st.error("❌ تعذر تحليل الصورة. تأكد أن الموديل يتوافق مع صور RGB بمقاسات قياسية.")

st.markdown("---")
st.caption("تنبيه: أداة مساعدة تقنية للطبيب، والقرار النهائي سريري.")
