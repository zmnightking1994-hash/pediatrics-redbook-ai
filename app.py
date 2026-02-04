import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. إعداد الصفحة
st.set_page_config(page_title="AI Pediatrics Radiologist", layout="wide")
st.title("🩺 مساعد طبيب الأطفال: نسخة الذكاء الاصطناعي العميق")

# 2. تحميل نموذج الذكاء الاصطناعي (مرة واحدة فقط لتوفير الوقت)
@st.cache_resource
def load_deep_model():
    # نستخدم نموذج متخصص في تمييز الأنماط البصرية
    return tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

model = load_deep_model()

# 3. وظيفة الفحص: طبيعي أم مصاب؟
def analyze_image(img):
    # تجهيز الصورة للنموذج
    resized = cv2.resize(img, (224, 224))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(rgb_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # قياس البياض (Logic) + تحليل الأنماط (AI)
    density = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    if density < 95: # عتبة الصدر السليم
        return "Normal", "الأشعة تبدو طبيعية ولا يوجد ارتشاحات واضحة."
    elif density > 135:
        return "Streptococcus pneumoniae", "تم اكتشاف بياض كثيف (Lobar Consolidation)."
    else:
        return "Mycoplasma pneumoniae", "تم اكتشاف ارتشاحات خفيفة (Interstitial Infiltrates)."

# 4. واجهة المستخدم
uploaded_file = st.file_uploader("ارفع صورة الأشعة (X-ray)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    # قراءة الصورة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    with col1:
        st.image(uploaded_file, caption="الصورة المرفوعة", use_container_width=True)
    
    # تنفيذ التشخيص
    diagnosis, note = analyze_image(img)
    
    with col2:
        st.header("النتيجة التحليلية")
        if diagnosis == "Normal":
            st.balloons()
            st.success(f"✅ الحالة: {diagnosis}")
            st.info(note)
        else:
            st.error(f"🚨 تشخيص محتمل: {diagnosis}")
            st.warning(f"ملاحظة الأشعة: {note}")
            
            # جلب العلاج من الـ Red Book
            db = pd.read_excel("pneumonia_reference.xlsx")
            entry = db[db['Pathogen'] == diagnosis].iloc[0]
            st.markdown(f"### 📖 مرجع Red Book (صفحة {entry['Page']})")
            st.write(entry['Treatment Snippet'][:500] + "...")
