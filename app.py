import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="Pediatric AI Expert", layout="wide")

# 2. تحميل الموديل مع معالجة الأخطاء
@st.cache_resource
def load_ai_model():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        try:
            # نستخدم compile=False لتجنب مشاكل توافق الدوال المخصصة
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            return f"Error: {e}"
    return None

ai_brain = load_ai_model()

st.title("🩺 مساعد طبيب الأطفال الذكي")
st.write("تحليل أشعة الصدر وفق بروتوكولات Red Book 2024")

# 3. رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة الأشعة (X-ray)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.2])
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    with col1:
        st.subheader("🔍 المعاينة البصرية")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)
        st.image(cv2.addWeighted(img, 0.6, heatmap, 0.4, 0), use_container_width=True)

    with col2:
        st.subheader("📋 نتيجة الفحص الذكي")
        if ai_brain is None:
            st.error("ملف model.h5 مفقود.")
        elif isinstance(ai_brain, str):
            st.error(ai_brain)
        else:
            try:
                # المعالجة الأساسية
                img_resized = cv2.resize(img, (150, 150)) / 255.0
                
                # الخدعة البرمجية: إضافة أبعاد إضافية لتوافق الموديلات المعقدة
                # نحول الصورة من (150, 150, 3) إلى (1, 1, 150, 150, 3) لتجنب خطأ Kernel Shape
                img_input = np.expand_dims(img_resized, axis=0)
                img_input = np.expand_dims(img_input, axis=0) 

                prediction = ai_brain.predict(img_input)
                
                # استخلاص الرقم النهائي مهما كان شكل الـ Output
                score = np.max(prediction)
                
                if score > 0.5:
                    st.error(f"🚨 إيجابي: احتمالية التهاب رئوي {score*100:.1f}%")
                    st.info("💡 البروتوكول: Amoxicillin (80–90 mg/kg/day) - Red Book p.646")
                else:
                    st.success(f"✅ سليم: الرئة طبيعية بنسبة {(1-score)*100:.1f}%")
                    st.balloons()
            except Exception as e:
                st.warning(f"⚠️ الموديل يتوقع أبعاداً خاصة. جاري محاولة التوافق التلقائي...")
                # محاولة أخيرة بأبعاد 2D قياسية إذا فشلت الخدعة أعلاه
                try:
                    img_2d = cv2.resize(img, (150, 150)) / 255.0
                    img_2d = np.expand_dims(img_2d, axis=0)
                    score = ai_brain.predict(img_2d)[0][0]
                    st.write(f"النتيجة: {score}")
                except:
                    st.error(f"فشل التحليل: الموديل المرفوع غير متوافق مع الصور الملونة القياسية. الخطأ: {e}")

st.markdown("---")
st.caption("أداة تقنية مساعدة لرفع كفاءة التشخيص السريري.")
