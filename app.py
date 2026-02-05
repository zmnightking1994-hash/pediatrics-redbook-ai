import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# 1. إعداد الصفحة وتنسيق الواجهة الطبية
st.set_page_config(page_title="AI Pediatric Radiologist", layout="wide")

st.markdown("""
    <style>
    .stAlert { border-radius: 12px; border: 2px solid #3498db; }
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; text-align: center; }
    .dosage-card { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border-right: 5px solid #27ae60; 
        margin-bottom: 10px; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. وظيفة تحميل الموديل الذكي
@st.cache_resource
def load_ai_model():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            return f"Error loading model: {e}"
    return None

ai_brain = load_ai_model()

st.title("🩺 مساعد طبيب الأطفال الذكي: Red Book + AI")
st.markdown("---")

# 3. قاعدة بيانات البروتوكولات (Red Book 2024)
RED_BOOK_GUIDELINES = {
    "Streptococcus pneumoniae": {
        "pattern": "Lobar Consolidation (تصلد فصي واضح)",
        "first_line": "Amoxicillin (80–90 mg/kg per day in 2 divided doses)",
        "max_dose": "4 g/day",
        "duration": "5–7 days",
        "alternative": "Ceftriaxone or Ampicillin.",
        "page": "646-648"
    }
}

# 4. واجهة رفع الصور والمعالجة
uploaded_file = st.file_uploader("ارفع صورة أشعة الصدر (X-ray)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.2])
    
    # تحويل الملف المرفوع لمصفوفة صور
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    with col1:
        st.subheader("🔍 التحليل البصري")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)
        st.image(cv2.addWeighted(img, 0.6, heatmap, 0.4, 0), caption="تحديد مناطق الكثافة شعاعياً", use_container_width=True)

    with col2:
        st.subheader("📋 نتيجة الذكاء الاصطناعي")
        
        if ai_brain is None:
            st.error("⚠️ لم يتم العثور على ملف model.h5 في المستودع. يرجى رفعه لـ GitHub بنفس الاسم.")
        elif isinstance(ai_brain, str):
            st.error(ai_brain)
        else:
            # معالجة الصورة للموديل (نستخدم مقاس 150x150 وهو الشائع)
            img_resized = cv2.resize(img, (150, 150)) / 255.0
            img_input = np.expand_dims(img_resized, axis=0)
            
            prediction = ai_brain.predict(img_input)[0][0]
            
            if prediction > 0.5:
                st.error(f"🚨 إيجابي: احتمالية وجود التهاب رئوي {prediction*100:.1f}%")
                
                # عرض البروتوكول العلاجي
                data = RED_BOOK_GUIDELINES["Streptococcus pneumoniae"]
                st.markdown("### 💊 الخطة العلاجية الموصى بها:")
                
                dosage_info = f"""
                <div class="dosage-card"><strong>🦠 المسبب المرجح:</strong> Streptococcus pneumoniae</div>
                <div class="dosage-card"><strong>📍 النمط الشعاعي:</strong> {data['pattern']}</div>
                <div class="dosage-card"><strong>💉 العلاج:</strong> {data['first_line']}</div>
                <div class="dosage-card"><strong>⏱️ المدة المتوقعة:</strong> {data['duration']}</div>
                <div class="dosage-card"><strong>📖 المرجع:</strong> Red Book 2024 (p. {data['page']})</div>
                """
                st.markdown(dosage_info, unsafe_allow_html=True)
            else:
                st.success(f"✅ سليم: الرئة تظهر طبيعية بنسبة {(1-prediction)*100:.1f}%")
                st.balloons()

st.markdown("---")
st.caption("🩺 تنبيه طبي: هذا النظام أداة مساعدة تقنية للطبيب، والقرار النهائي يعتمد على التقييم السريري.")
