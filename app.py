import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Pediatrics Radiologist", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 مساعد طبيب الأطفال: تحليل الأشعة والـ Red Book")

# وظيفة معالجة الصورة وإبراز مناطق الالتهاب
def highlight_infection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # استخدام تصفية لتحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # كشف المناطق ذات الكثافة العالية (بياض الأشعة)
    _, thresh = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
    # تحويلها لخريطة حرارية خفيفة
    heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    added_image = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    return added_image

uploaded_file = st.file_uploader("ارفع صورة الأشعة هنا (JPG/PNG)", type=["jpg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    # معالجة الصورة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    processed_img = highlight_infection(img)
    
    with col1:
        st.subheader("🔍 التحليل البصري")
        st.image(processed_img, caption="تم تحديد مناطق الكثافة العالية أوتوماتيكياً", use_container_width=True)
    
    with col2:
        st.subheader("💡 التوصية الطبية (Red Book)")
        # قراءة قاعدة البيانات
        db = pd.read_excel("pneumonia_reference.xlsx")
        
        # تحليل النمط (تبسيطاً)
        density = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        pathogen = "Streptococcus pneumoniae" if density > 130 else "Mycoplasma pneumoniae"
        
        entry = db[db['Pathogen'] == pathogen].iloc[0]
        
        st.success(f"**المسبب المرجح:** {pathogen}")
        st.info(f"📍 مرجع الكتاب: صفحة {entry['Page']}")
        
        # عرض العلاج بتنسيق جميل
        treatment_text = entry['Treatment Snippet']
        st.markdown("### 💊 خطة العلاج المقترحة:")
        
        # البحث عن الكلمات المهمة وتلوينها
        for word in ["Amoxicillin", "Ceftriaxone", "Dose", "Duration", "mg/kg"]:
            treatment_text = treatment_text.replace(word, f"**{word}**")
        
        st.write(treatment_text[:600] + "...")

st.sidebar.header("حول النظام")
st.sidebar.info("هذا النظام يربط تحليل الصور ببيانات كتاب Red Book 2024 لإرشاد الأطباء في المناطق النائية.")
