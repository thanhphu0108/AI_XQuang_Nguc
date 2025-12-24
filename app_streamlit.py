import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import pydicom
import json
import google.generativeai as genai
from supabase import create_client, Client

# ================= 1. CẤU HÌNH =================
st.set_page_config(page_title="AI Hospital (V29.0 - Universal Test)", page_icon="🌐", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    .stButton>button { width: 100%; font-weight: bold; height: 45px; }
    .gemini-box { background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 5px solid #1976d2; margin: 10px 0; }
    .step-badge { background-color: #002f6c; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; }
    .rating-box { border: 2px solid #ff9800; padding: 10px; border-radius: 10px; background-color: #fff3e0; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# --- KẾT NỐI SUPABASE ---
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except: return None

supabase = init_supabase()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TECHNICAL_OPTS = [
    "✅ Phim chuẩn", "⚠️ Chụp tại giường (AP)", "⚠️ Hít vào nông", 
    "⚠️ Bệnh nhân xoay", "⚠️ Tia cứng/mềm", "⚠️ Dị vật/Áo", 
    "⚠️ Mất góc sườn hoành", "🧠 Ca khó", "🧠 Nhiễu chồng hình"
]

ALLOWED_LABELS = ["Normal", "Cardiomegaly", "Pneumonia", "Effusion", "Pneumothorax", "Nodule_Mass", "Fibrosis_TB", "Fracture", "Pleural_Thickening", "Other"]

DOCTOR_ROSTER = {
    "ANATOMY": "Dr_Anatomy.pt", "PNEUMOTHORAX": "Dr_Pneumothorax.pt", 
    "PNEUMONIA": "Dr_Pneumonia.pt", "TUMOR": "Dr_Tumor.pt",        
    "EFFUSION": "Dr_Effusion.pt", "OPACITY": "Dr_Opacity.pt", "HEART": "Dr_Heart.pt"         
}

# ================= 2. CORE FUNCTIONS =================
@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else 'cpu'
    loaded_models = {}
    for role, filename in DOCTOR_ROSTER.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try: loaded_models[role] = YOLO(path)
            except: pass
    return loaded_models, [], device

MODELS, MODEL_STATUS, DEVICE = load_models()

# --- SUPABASE FUNCTIONS ---
def upload_image(img_cv, filename):
    try:
        # Encode JPG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
        bucket = "xray_images"
        # Upload
        supabase.storage.from_(bucket).upload(filename, buffer.tobytes(), {"content-type": "image/jpeg", "upsert": "true"})
        # Get URL
        return supabase.storage.from_(bucket).get_public_url(filename)
    except Exception as e:
        # Fallback lấy URL nếu lỗi upload (do file tồn tại)
        try: return supabase.storage.from_("xray_images").get_public_url(filename)
        except: return None

def save_log(data):
    try:
        supabase.table("logs").upsert(data).execute()
        return True
    except Exception as e:
        st.error(f"Lỗi DB: {e}")
        return False

def get_logs():
    try:
        response = supabase.table("logs").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(response.data)
    except: return pd.DataFrame()

# --- GEMINI ---
def ask_gemini(api_key, image, context, note, guide, tags):
    try:
        genai.configure(api_key=api_key)
        labels_str = ", ".join(ALLOWED_LABELS)
        tech_note = ", ".join(tags) if tags else "Chuẩn."
        
        prompt = f"""
        Role: Senior Radiologist.
        INPUTS:
        - Clinical Context: "{context}"
        - Expert Note: "{note}"
        - Guidance: "{guide}"
        - Technical QA: "{tech_note}"
        
        TASK: Analyze Chest X-ray. Select labels from [{labels_str}] or 'Normal'.
        OUTPUT JSON: {{ "labels": ["..."], "reasoning": "..." }}
        """
        
        models = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro"]
        for m in models:
            try:
                model = genai.GenerativeModel(m)
                response = model.generate_content([prompt, image], generation_config={"response_mime_type": "application/json", "temperature": 0.0})
                res = json.loads(response.text)
                res["used_model"] = m
                res["sent_prompt"] = prompt
                return res
            except: continue
        return {"labels": [], "reasoning": "Lỗi Gemini.", "sent_prompt": prompt}
    except Exception as e: return {"labels": [], "reasoning": str(e), "sent_prompt": ""}

def read_dicom_image(file_buffer):
    try:
        ds = pydicom.dcmread(file_buffer)
        p_name = str(ds.get("PatientName", "Anonymous")).replace('^', ' ').strip()
        p_id = str(ds.get("PatientID", "Unknown"))
        img = ds.pixel_array.astype(float)
        img = (np.maximum(img, 0) / img.max()) * 255.0
        img = np.uint8(img)
        if ds.get("PhotometricInterpretation") == "MONOCHROME1": img = 255 - img
        if len(img.shape) == 2: img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else: img_rgb = img
        return img_rgb, f"{p_name} ({p_id})"
    except: return None, "Lỗi DICOM"

# --- PROCESS & SAVE (UNIVERSAL MODE) ---
def process_and_save(image_file):
    start_t = time.time()
    filename = image_file.name.lower()
    img_rgb, patient_info = None, "Ẩn danh"
    
    # 1. Reset file
    image_file.seek(0)
    
    # 2. Đọc ảnh (Hỗ trợ mọi loại)
    if filename.endswith(('.dcm', '.dicom')):
        img_rgb, p_info = read_dicom_image(image_file)
        if img_rgb is None: return None, {"Error": f"Lỗi DICOM: {p_info}"}, False, None
        patient_info = p_info
    else:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
        if img_cv is None: return None, {"Error": "Không đọc được ảnh"}, False, None
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Resize chuẩn
    h, w = img_rgb.shape[:2]
    scale = 1280 / max(h, w)
    img_resized = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
    display_img = img_resized.copy()
    
    findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    has_danger = False
    
    # 3. Chạy YOLO (Nếu có Model)
    if "ANATOMY" in MODELS:
        try:
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            anatomy_res = MODELS["ANATOMY"](img_bgr, conf=0.35, verbose=False)[0]
            for box in anatomy_res.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                region = anatomy_res.names[cls_id]
                x1, y1, x2, y2 = coords
                
                roi = img_bgr[max(0, y1-40):min(h, y2+40), max(0, x1-40):min(w, x2+40)]
                if roi.size == 0: continue
                
                target_models = []
                if "Lung" in region: target_models = ["PNEUMOTHORAX", "EFFUSION", "PNEUMONIA", "TUMOR"]
                elif "Heart" in region: target_models = ["HEART"]
                
                for spec in target_models:
                    if spec in MODELS:
                        res = MODELS[spec](roi, verbose=False)[0]
                        if res.probs.top1conf.item() > 0.6 and res.names[res.probs.top1] == "Disease":
                            pct = res.probs.top1conf.item() * 100
                            if pct > 75: has_danger = True
                            text = f"{region}: {spec} ({pct:.0f}%)"
                            
                            if "HEART" in spec: findings_db["Heart"].append(text)
                            elif "PLEURA" in spec or "EFFUSION" in spec: findings_db["Pleura"].append(text)
                            else: findings_db["Lung"].append(text)
                            
                            color = (255, 0, 0) if pct > 75 else (0, 165, 255)
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_img, spec[:4], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except: pass # Bỏ qua lỗi AI để vẫn upload được
    else:
        # Chế độ Mockup (Không có AI)
        findings_db["Lung"].append("Chế độ Test (Không có Model AI)")

    # 4. UPLOAD & SAVE (Luôn thực hiện)
    img_id = datetime.now().strftime("%d%m%Y%H%M%S")
    file_name = f"XRAY_{img_id}.jpg"
    img_url = upload_image(display_img, file_name)
    
    if img_url:
        data = {
            "id": img_id,
            "created_at": datetime.now().isoformat(),
            "image_url": img_url,
            "result": "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG",
            "details": str(findings_db),
            "patient_info": patient_info,
            "feedback_1": "Chưa đánh giá",
            "feedback_2": "Chưa đánh giá"
        }
        save_log(data)
    else:
        return display_img, {"Error": "Lỗi Upload Supabase (Check Bucket/Key)"}, has_danger, img_id
        
    return display_img, findings_db, has_danger, img_id

# ================= 3. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("ĐIỀU KHIỂN")
    api_key = st.text_input("🔑 Gemini API Key:", value=st.secrets.get("GEMINI_API_KEY", ""), type="password")
    mode = st.radio("Menu:", ["🔍 Upload & Phân Tích", "📂 Hội Chẩn & Labeling", "🛠️ Xuất Dataset"])

if mode == "🔍 Upload & Phân Tích":
    st.title("🏥 AI CHẨN ĐOÁN (SUPABASE CLOUD)")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG, DICOM):", type=["jpg", "png", "jpeg", "dcm"])
        if uploaded_file and st.button("🚀 PHÂN TÍCH & UPLOAD", type="primary"):
            with col2:
                with st.spinner("Đang xử lý & Upload..."):
                    img_out, findings, danger, img_id = process_and_save(uploaded_file)
                    
                    if img_out is not None:
                        st.image(img_out, caption=f"ID: {img_id}", use_container_width=True)
                        if isinstance(findings, dict) and "Error" in findings:
                            st.error(f"⚠️ {findings['Error']}")
                        else:
                            if danger: st.error("🔴 PHÁT HIỆN BẤT THƯỜNG")
                            else: st.success("✅ ĐÃ UPLOAD THÀNH CÔNG")
                            st.json(findings)
                            st.toast("Đã lưu vào Cloud!", icon="☁️")
                    else:
                        err = findings.get("Error", "Lỗi") if isinstance(findings, dict) else "Lỗi"
                        st.error(f"❌ {err}")

elif mode == "📂 Hội Chẩn & Labeling":
    st.title("📂 DATA LABELING (SUPABASE)")
    if "sent_prompt" not in st.session_state: st.session_state["sent_prompt"] = ""
    
    df = get_logs()
    if not df.empty:
        df = df.fillna("")
        selected_id = st.selectbox("👉 Chọn Ca bệnh:", df['id'].unique())
        if selected_id:
            record = df[df["id"] == selected_id].iloc[0]
            c1, c2 = st.columns([1, 1])
            with c1:
                if record.get('image_url'):
                    st.image(record['image_url'], use_container_width=True)
                    try:
                        import requests
                        from io import BytesIO
                        response = requests.get(record['image_url'])
                        pil_img = Image.open(BytesIO(response.content))
                    except: pil_img = None
            
            with c2:
                st.info(f"BN: {record.get('patient_info')}")
                st.warning(f"AI: {record.get('result')}")
                
                ctx = st.text_area("🏥 Bệnh cảnh:", value=record.get("clinical_context") or "", height=70)
                note = st.text_area("👨‍⚕️ Ý kiến chuyên gia:", value=record.get("expert_note") or "", height=70)
                guide = st.text_area("🤖 Dẫn dắt Prompt:", value=record.get("prompt_guidance") or "", height=70)
                
                tags_str = record.get("technical_tags") or ""
                def_tags = [t.strip() for t in tags_str.split(";")] if tags_str else []
                tags = st.multiselect("⚙️ QA/QC:", TECHNICAL_OPTS, default=def_tags)
                
                gemini_lbls, gemini_txt, used_model = [], "", ""
                if api_key and pil_img and st.button("🧠 Hỏi Gemini"):
                    with st.spinner("Gemini đang nghĩ..."):
                        res = ask_gemini(api_key, pil_img, ctx, note, guide, tags)
                        gemini_lbls = res.get("labels", [])
                        gemini_txt = res.get("reasoning", "")
                        used_model = res.get("used_model", "Unknown")
                        st.session_state["sent_prompt"] = res.get("sent_prompt", "")
                        
                        if gemini_lbls: st.success(f"Gợi ý: {', '.join(gemini_lbls)}")
                        else: st.error(gemini_txt)

                cur_prompt = st.session_state["sent_prompt"] or record.get("full_prompt", "")
                if cur_prompt:
                    with st.expander("Debug Prompt"): st.code(cur_prompt)
                    saved_rating = record.get("prompt_rating", "Khá")
                    rating_opts = ["Tệ", "TB", "Khá", "Tốt", "Xuất sắc"]
                    val = saved_rating if saved_rating in rating_opts else "Khá"
                    rating = st.select_slider("🌟 Đánh giá Prompt:", options=rating_opts, value=val)
                else: rating = "Khá"

                st.markdown("---")
                fb1 = str(record.get("feedback_1") or "")
                
                if fb1 == "Chưa đánh giá" or not fb1:
                    st.markdown('<div class="step-badge">🔹 LẦN 1</div>', unsafe_allow_html=True)
                    new_fb = st.radio("Đánh giá:", ["Chưa", "Đúng", "Sai"], index=0)
                    new_lbls = st.multiselect("Chốt bệnh:", ALLOWED_LABELS)
                    if st.button("💾 LƯU LẦN 1"):
                        data = {
                            "id": selected_id, "clinical_context": ctx, "expert_note": note, "prompt_guidance": guide,
                            "technical_tags": "; ".join(tags), "full_prompt": cur_prompt,
                            "prompt_rating": rating, "feedback_1": new_fb, "label_1": "; ".join(new_lbls),
                            "ai_reasoning": gemini_txt, "used_model": used_model
                        }
                        if save_log(data): st.success("Đã lưu!"); time.sleep(0.5); st.rerun()
                else:
                    st.success("Đã đánh giá Lần 1.")
                    st.markdown('<div class="step-badge" style="background:#c62828">🔸 LẦN 2 (CHỐT)</div>', unsafe_allow_html=True)
                    new_fb2 = st.radio("Đánh giá:", ["Chưa", "Đúng", "Sai"], index=0, key="fb2")
                    new_lbls2 = st.multiselect("Chốt bệnh:", ALLOWED_LABELS, key="lbl2")
                    if st.button("💾 LƯU CHỐT"):
                        data = {"id": selected_id, "feedback_2": new_fb2, "label_2": "; ".join(new_lbls2)}
                        if save_log(data): st.success("Đã chốt!"); time.sleep(0.5); st.rerun()

elif mode == "🛠️ Xuất Dataset":
    st.title("🛠️ XUẤT DATASET (SUPABASE)")
    if st.button("📥 Tải CSV"):
        df = get_logs()
        if not df.empty:
            st.download_button("Download", df.to_csv(index=False).encode('utf-8'), "supabase_data.csv", "text/csv")
        else: st.error("Trống.")