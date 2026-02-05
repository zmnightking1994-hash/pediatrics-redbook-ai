import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# 1. إعداد واجهة احترافية متكاملة
st.set_page_config(page_title="AI Pediatric Radiologist", layout="wide")

# تنسيق CSS مخصص لجعل الواجهة تبدو كبرنامج طبي
st.markdown("""
    <style>
    .stAlert { border-radius: 12px; border: 2px solid #3498db; }
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; text-align: center; font-family: 'Arial'; }
    .dosage-card { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border-right: 5px solid #27ae60; 
        margin-bottom: 10px; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        font-size: 18px;
    }
    .result-text { font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- تحميل موديل الـ H5 (المخ الذكي) ---
@st.cache_resource
def load_ai_model():
    try:
        # قمنا بتغيير الاسم هنا ليتطابق مع ملفك المرفوع 'model.h5'
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        return f"Error: {e}"

# استدعاء الموديل
ai_brain = load_ai_model()

st.title("🩺 مساعد طبيب الأطفال الذكي: Red Book + AI")
st.markdown("---")

# --- قاعدة بيانات البروتوكولات الرسمية (Red Book 2024) ---
RED_BOOK_GUIDELINES = {
    "Streptococcus pneumoniae": {
        "pattern": "Lobar Consolidation (تصلد فصي واضح)",
        "first_line": "Amoxicillin (80–90 mg/kg per day in 2 divided doses)",
        "max_dose": "4 g/day",
        "duration": "5–7 days (for uncomplicated cases)",
        "alternative": "Ceftriaxone (50–100 mg/kg per day IV/IM once daily) or Ampicillin.",
        "page": "646-648"
    }
}

# --- واجهة رفع الملفات ---
uploaded_file = st.file_uploader("ارفع صورة أشعة الصدر (X-ray)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.2])
    
    # تحويل الملف المرفوع إلى مصفوفة صور
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    with col1:
        st.subheader("🔍 التحليل البصري المتقدم")
        # عرض الخريطة الحرارية (Heatmap) لتبسيط الرؤية للطبيب
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)
        st.image(cv2.addWeighted(img, 0.6, heatmap, 0.4, 0), caption="تحديد مناطق الكثافة حرارياً", use_container_width=True)

    with col2:
        st.subheader("📋 نتيجة الفحص والبروتوكول")
        
        if isinstance(ai_brain, str):
            st.error(f"⚠️ خطأ في تحميل ملف الموديل: {ai_brain}")
            st.info("تأكد من وجود ملف باسم model.h5 في المستودع")
        else:
            # --- معالجة الصورة للذكاء الاصطناعي ---
            # المقاس الافتراضي لمعظم الموديلات هو 150 أو 224
            img_input = cv2.resize(img, (150, 150)) 
            img_input = img_input / 255.0  # التطبيع
            img_input = np.expand_dims(img_input, axis=0)
            
            # تنفيذ التوقع
            with st.spinner('جاري تحليل الأنماط الشعاعية...'):
                prediction = ai_brain.predict(img_input)[0][0]
            
            if prediction > 0.5:
                # الحالة إيجابية
                st.markdown(f'<p class="result-text" style="color:#e74c3c;">🚨 النتيجة: إيجابي (التهاب رئوي مريب)</p>', unsafe_allow_html=True)
                st.warning(f"احتمالية الإصابة بناءً على الذكاء الاصطناعي: {prediction*100:.1f}%")
                
                # عرض بروتوكول العلاج من Red Book
                data = RED_BOOK_GUIDELINES["Streptococcus pneumoniae"]
                st.markdown("### 💊 بروتوكول العلاج المعتمد (Red Book 2024):")
                
                st.markdown(f"""
                <div class="dosage-card"><strong>🦠 المسبب المرجح شعاعياً:</strong> Streptococcus pneumoniae</div>
                <div class="dosage-card"><strong>📍 النمط
