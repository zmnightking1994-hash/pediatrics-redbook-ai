import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="AI Pediatric Radiologist", layout="wide")

# 2. تحميل الموديل (تأكد أن ملف model.h5 مرفوع في GitHub)
@st.cache_resource
def load_ai_model():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        try:
            # compile=False ضرورية لتجنب تعارض الإصدارات
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            return f"Error: {e}"
    return None

ai_brain = load_ai_model()

st.title("🩺 مساعد طبيب الأطفال الذكي")
st.markdown("---")

uploaded_file = st.file_uploader("ارفع صورة الأشعة (X-ray)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.2])
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    with col1:
        st.subheader("🔍 المعاينة البصرية")
        # تحسين الصورة للعرض فقط
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)
        st.image(cv2.addWeighted(img, 0.6, heatmap, 0.4, 0), use_container_width=True)

    with col2:
        st.subheader("📋 نتيجة الفحص الذكي")
        if ai_brain is None:
            st.error("⚠️ ملف model.h5 غير موجود في المستودع.")
        elif isinstance(ai_brain, str):
            st.error(ai_brain)
        else:
            try:
                # المقاس الذي يتوقعه الموديل بناءً على رسالة الخطأ
                img_resized = cv2.resize(img, (150, 150)) / 255.0
                
                # الحل الجذري: إضافة بُعد واحد فقط (Batch Dimension)
                # المصفوفة ستصبح (1, 150, 150, 3) وهذا يطابق هيكل الموديل (4 أبعاد)
                img_input = np.expand_dims(img_resized, axis=0) 

                prediction = ai_brain.predict(img_input)
                score = float(np.max(prediction)) 
                
                if score > 0.5:
                    st.error(f"🚨 إيجابي: احتمالية التهاب رئوي {score*100:.1f}%")
                    st.markdown("""
                        <div style="background-color: #fff; padding: 15px; border-radius: 10px; border-right: 5px solid #e74c3c; color: #2c3e50;">
                            <strong>💊 بروتوكول Red Book 2024:</strong><br>
                            الخيار الأول: Amoxicillin (80–90 mg/kg/day).
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ سليم: الرئة طبيعية بنسبة {(1-score)*100:.1f}%")
                    st.balloons()
            except Exception as e:
                st.error(f"❌ خطأ فني في التحليل: {e}")

st.markdown("---")
st.caption("أداة تقنية مساعدة للطبيب - النتيجة النهائية تخضع للتقييم السريري.")
