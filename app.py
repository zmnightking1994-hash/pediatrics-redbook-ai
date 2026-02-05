import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# 1. إعدادات الواجهة
st.set_page_config(page_title="AI Pediatric Radiologist", layout="wide")

# 2. تحميل الموديل بدون الطبقات التدريبية (لحل مشاكل التوافق)
@st.cache_resource
def load_ai_model():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        try:
            # استخدام compile=False ضروري جداً لتجاوز أخطاء الـ Kernels الأصلية
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
                # المقاس الذي يتوقعه الموديل
                img_resized = cv2.resize(img, (150, 150)) / 255.0
                
                # --- الحل النهائي لمشكلة 5D vs 4D ---
                # الموديل يطلب (None, None, 150, 150, 3)
                # سنقوم بإنشاء مصفوفة خماسية الأبعاد يدوياً
                # البعد الأول: Batch (1)
                # البعد الثاني: Sequence/Depth (1)
                img_5d = img_resized[np.newaxis, np.newaxis, :, :, :]
                
                # تنفيذ التوقع
                prediction = ai_brain.predict(img_5d)
                score = np.max(prediction) 
                
                if score > 0.5:
                    st.error(f"🚨 إيجابي: احتمالية التهاب رئوي {score*100:.1f}%")
                    st.markdown("""
                        <div style="background-color: #fff; padding: 10px; border-radius: 5px; border-right: 5px solid #e74c3c;">
                            <strong>💊 بروتوكول Red Book 2024:</strong><br>
                            Amoxicillin (80–90 mg/kg/day) مقسمة على جرعتين.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ سليم: الرئة طبيعية بنسبة {(1-score)*100:.1f}%")
                    st.balloons()
            except Exception as e:
                # محاولة أخيرة إذا كان الموديل يتوقع أبعاداً مختلفة قليلاً
                st.warning("جاري محاولة ضبط الأبعاد تلقائياً...")
                try:
                    img_4d = img_resized[np.newaxis, :, :, :]
                    score = ai_brain.predict(img_4d)[0][0]
                    st.write(f"النتيجة: {score}")
                except:
                    st.error(f"❌ الموديل المرفوع لا يتوافق مع الصور الفردية. خطأ: {e}")

st.markdown("---")
st.caption("أداة مساعدة تقنية للطبيب.")
