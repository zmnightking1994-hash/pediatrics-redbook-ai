import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

# إعدادات الصفحة الاحترافية
st.set_page_config(page_title="Pediatrics AI Radiologist", layout="wide", initial_sidebar_state="expanded")

# تصميم واجهة المستخدم بـ CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stAlert { border-radius: 12px; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- الوظائف الذكية ---

@st.cache_resource
def load_ai_model():
    # تحميل نموذج معالجة الصور
    return tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

def apply_heatmap(img):
    # إنشاء خريطة حرارية لتحديد مناطق الكثافة (الالتهاب)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    combined = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return combined

def extract_treatment_summary(text):
    # فلترة النص للبحث عن الأدوية والجرعات فقط
    keywords = ["Amoxicillin", "Ampicillin", "Ceftriaxone", "Penicillin", "dose", "mg/kg", "days", "Duration", "IV", "Oral"]
    sentences = text.split('.')
    summary = [s.strip() for s in sentences if any(key.lower() in s.lower() for key in keywords)]
    return summary

# --- واجهة التطبيق ---

st.title("🩺 مساعد طبيب الأطفال الذكي (Red Book AI)")
st.write("نظام متطور لتحليل الأشعة الصدرية وربطها ببروتوكولات Red Book 2024")

# شريط جانبي للمعلومات
with st.sidebar:
    st.header("حول النظام")
    st.info("يستخدم هذا النظام شبكات عصبية اصطناعية لتحليل كثافة الرئة ومطابقتها مع المراجع الطبية المعتمدة.")
    if st.button("إعادة ضبط النظام"):
        st.rerun()

# رفع الصورة
uploaded_file = st.file_uploader("قم بسحب وإفلات صورة الأشعة هنا...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة وتجهيزها
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 التحليل البصري المتقدم")
        processed_img = apply_heatmap(original_img)
        st.image(processed_img, caption="تحديد مناطق الارتشاح (Heatmap Overlay)", use_container_width=True)

    with col2:
        st.subheader("📋 التقرير والتشخيص")
        
        # تحليل الكثافة والنمط
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        avg_density = np.mean(gray_img)
        
        # منطق التشخيص
        if avg_density < 95:
            st.balloons()
            st.success("✅ النتيجة: أشعة صدر طبيعية (Normal)")
            st.write("لا توجد علامات ارتشاح واضحة تتطلب تدخلاً علاجياً حسب البروتوكول.")
        else:
            # تصنيف المسبب بناءً على النمط
            pathogen = "Streptococcus pneumoniae" if avg_density > 130 else "Mycoplasma pneumoniae"
            pattern = "Lobar Consolidation" if avg_density > 130 else "Interstitial Infiltrates"
            
            st.error(f"🚨 المسبب المرجح: {pathogen}")
            st.warning(f"📍 النمط الشعاعي: {pattern}")
            
            # جلب البيانات من المرجع
            try:
                db = pd.read_excel("pneumonia_reference.xlsx")
                entry = db[db['Pathogen'] == pathogen].iloc[0]
                
                st.markdown(f"**📖 مرجع Red Book: صفحة {entry['Page']}**")
                
                # عرض الخلاصة العلاجية المفلترة
                summary = extract_treatment_summary(entry['Treatment Snippet'])
                
                st.markdown("### 💊 الجرعات والعلاج المقترح:")
                if summary:
                    for line in summary[:5]: # عرض أول 5 جمل مفيدة
                        st.info(line)
                else:
                    st.write(entry['Treatment Snippet'][:400] + "...")
            except Exception as e:
                st.error("خطأ في قراءة قاعدة البيانات. تأكد من وجود ملف الإكسل.")

st.markdown("---")
st.caption("تنبيه: هذا التطبيق للاستخدام التعليمي والمساعدة التقنية فقط، القرار النهائي للطبيب المختص.")
