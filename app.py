import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="Pediatric AI Expert", layout="wide")

# 2. تحميل الموديل مع تجاهل إعدادات التدريب الأصلية لتجنب التعارض
@st.cache_resource
def load_ai_model():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        try:
            # استخدام compile=False ضروري جداً هنا لحل مشاكل التوافق
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            return f"Error: {e}"
    return None

ai_brain = load_ai_model()

st.title("🩺 مساعد طبيب الأطفال: Red Book + AI")

# 3. واجهة رفع الملفات
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
            st.error("ملف model.h5 غير موجود في GitHub.")
        elif isinstance(ai_brain, str):
            st.error(ai_brain)
        else:
            try:
                # تحضير الصورة بالمقاس الذي يطلبه الموديل (150x150)
                img_resized = cv2.resize(img, (150, 150)) / 255.0
                
                # --- التعديل الجوهري لحل خطأ Kernel Shape ---
                # الموديل يتوقع 5 أبعاد: (Batch, Frames, Height, Width, Channels)
                # سنضيف بُعدين إضافيين للصورة:
                img_input = np.expand_dims(img_resized, axis=0) # البعد الرابع (Batch)
                img_input = np.expand_dims(img_input, axis=0) # البعد الخامس (Frames/Time)
                
                # الآن أصبح شكل المدخلات (1, 1, 150, 150, 3) وهذا سيحل الخطأ
                prediction = ai_brain.predict(img_input)
                score = np.max(prediction) 
                
                if score > 0.5:
                    st.error(f"🚨 إيجابي: احتمالية التهاب رئوي {score*100:.1f}%")
                    st.markdown("""
                        <div style="background-color: #f8d7da; padding: 15px; border-radius: 10px; border: 1px solid #f5c6cb;">
                            <strong>💉 بروتوكول Red Book 2024:</strong><br>
                            Amoxicillin (80–90 mg/kg/day) مقسمة على جرعتين.<br>
                            المرجع: صفحة 646.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ سليم: الرئة تظهر طبيعية بنسبة {(1-score)*100:.1f}%")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ خطأ فني: {e}")
                st.info("هذا الموديل يتطلب معالجة خاصة للأبعاد.")

st.markdown("---")
st.caption("أداة مساعدة تقنية للطبيب - النتيجة النهائية تخضع للتقييم السريري.")
