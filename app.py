import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="AI Pediatric Radiologist", layout="wide")

# 2. تحميل الموديل بحذر
@st.cache_resource
def load_ai_model():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        try:
            # استخدام compile=False لتجنب مشاكل الإصدارات
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            return f"Error: {e}"
    return None

ai_brain = load_ai_model()

st.title("🩺 مساعد طبيب الأطفال الذكي")
st.markdown("---")

# 3. رفع الصورة والمعالجة
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
            st.error("ملف model.h5 غير موجود.")
        elif isinstance(ai_brain, str):
            st.error(ai_brain)
        else:
            try:
                # تغيير الحجم للمقاس المطلوب 150x150
                img_resized = cv2.resize(img, (150, 150)) / 255.0
                
                # --- الحل السحري لمشكلة الـ Kernel Shape ---
                # الموديل يتوقع (Batch, Time, Height, Width, Channels)
                # سنضيف بُعدين إضافيين للصورة لتصبح 5D
                img_input = np.expand_dims(img_resized, axis=0) # تصبح (1, 150, 150, 3)
                img_input = np.expand_dims(img_input, axis=0) # تصبح (1, 1, 150, 150, 3) وهو المطلوب!

                prediction = ai_brain.predict(img_input)
                score = np.max(prediction) # استخراج أعلى قيمة ثقة
                
                if score > 0.5:
                    st.error(f"🚨 نتيجة إيجابية: احتمالية التهاب رئوي {score*100:.1f}%")
                    st.markdown("""
                        <div style="background-color: #fff; padding: 15px; border-radius: 10px; border-right: 5px solid #27ae60;">
                            <strong>💊 بروتوكول Red Book 2024:</strong><br>
                            Amoxicillin (80–90 mg/kg/day) - مرتين يومياً لمدة 5-7 أيام.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ نتيجة سليمة: الرئة طبيعية بنسبة {(1-score)*100:.1f}%")
                    st.balloons()
            except Exception as e:
                st.error(f"⚠️ خطأ في توافق الموديل: {e}")
                st.info("نصيحة: الموديل المرفوع قد يكون مصمماً لبيانات معقدة جداً.")

st.markdown("---")
st.caption("أداة مساعدة للطبيب - تعتمد النتيجة النهائية على التقييم السريري.")
