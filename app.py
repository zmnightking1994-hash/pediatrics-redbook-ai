import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

# إعداد واجهة احترافية
st.set_page_config(page_title="Pediatrics AI Radiologist", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stAlert { border-radius: 10px; border: 1px solid #d1d8e0; }
    h1 { color: #2c3e50; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 مساعد طبيب الأطفال الذكي (Red Book AI)")

# --- محرك الفلترة الطبية الذكي ---
def clean_medical_text(text):
    # قائمة الأدوية والكلمات التي تهم الطبيب في الـ Red Book
    important_keywords = [
        "Amoxicillin", "Ampicillin", "Ceftriaxone", "Penicillin", "Azithromycin", 
        "dose", "mg/kg", "days", "Duration", "IV", "Oral", "Treatment"
    ]
    
    # تنظيف النص من الروابط وكلام الـ HIV غير ذي الصلة بالحالات العامة
    sentences = text.split('.')
    filtered_sentences = []
    
    for s in sentences:
        # استبعاد الجمل التي تحتوي على HIV إذا لم نكن نبحث عنها
        if "HIV" in s or "clinicalinfo" in s:
            continue
        # الاحتفاظ بالجمل التي تحتوي على أدوية أو جرعات
        if any(key.lower() in s.lower() for key in important_keywords):
            filtered_sentences.append(s.strip())
            
    return filtered_sentences[:5] # إرجاع أهم 5 جمل علاجية فقط

# --- واجهة التطبيق ---
uploaded_file = st.file_uploader("ارفع صورة الأشعة هنا...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.2])
    
    # معالجة الصورة للعرض الحراري
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    with col1:
        st.subheader("🔍 التحليل البصري (Heatmap)")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)
        st.image(cv2.addWeighted(img, 0.6, heatmap, 0.4, 0), use_container_width=True)

    with col2:
        st.subheader("📋 التشخيص والبروتوكول العلاجي")
        
        # تحليل الكثافة (تبسيطاً)
        avg_density = np.mean(gray)
        pathogen = "Streptococcus pneumoniae" if avg_density > 130 else "Mycoplasma pneumoniae"
        
        st.error(f"⚠️ المسبب المرجح: {pathogen}")
        
        # جلب البيانات من المرجع المعدل
        try:
            db = pd.read_excel("pneumonia_reference.xlsx")
            raw_data = db[db['Pathogen'] == pathogen].iloc[0]['Treatment Snippet']
            page_num = db[db['Pathogen'] == pathogen].iloc[0]['Page']
            
            st.info(f"📖 مرجع الكتاب: صفحة {page_num}")
            st.markdown("### 💊 الجرعات والعلاج المقترح:")
            
            # عرض العلاج المفلتر
            clinical_tips = clean_medical_text(raw_data)
            
            if clinical_tips:
                for tip in clinical_tips:
                    st.success(f"**{tip}**")
            else:
                # في حال لم يجد جمل مفلترة، يعرض نصاً افتراضياً دقيقاً طبياً بناءً على المسبب
                if "Streptococcus" in pathogen:
                    st.warning("الجرعة القياسية: Amoxicillin (80–90 mg/kg per day in 2 divided doses)")
                else:
                    st.warning("الجرعة القياسية: Azithromycin (10 mg/kg on day 1, then 5 mg/kg for 4 days)")
                    
        except:
            st.error("تأكد من وجود ملف pneumonia_reference.xlsx")

st.caption("ملاحظة: هذا النظام استرشادي فقط. القرار السريري النهائي للطبيب.")
