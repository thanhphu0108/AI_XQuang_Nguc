import streamlit as st
import subprocess
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
from datetime import datetime
from PIL import Image
import pandas as pd
import pydicom
import json
import ast
import hashlib
import zipfile
from supabase import create_client, Client
import requests
from io import BytesIO

# --- 🛠️ AUTO-FIX LIB ---
try:
    import google.generativeai as genai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
    st.rerun()

# ================= 1. CẤU HÌNH & CSS "KHOA HỌC" =================
st.set_page_config(page_title="AI Hospital (V33.4 - Pro)", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    
    /* CARD STYLE (Giao diện khoa học) */
    .sci-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-top: 3px solid #002f6c;
    }
    
    .sci-header {
        font-size: 16px;
        font-weight: bold;
        color: #002f6c;
        margin-bottom: 15px;
        text-transform: uppercase;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }

    /* CHAT HISTORY STYLE (Đẹp) */
    .chat-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 10px;
        background: white;
        overflow: hidden;
    }
    .chat-meta {
        background-color: #f8f9fa;
        padding: 8px 12px;
        font-size: 11px;
        color: #666;
        border-bottom: 1px solid #eee;
        display: flex; justify-content: space-between;
    }
    .chat-prompt {
        padding: 10px;
        font-family: 'Consolas', monospace;
        font-size: 12px;
        color: #444;
        background-color: #fffde7;
        border-bottom: 1px dashed #eee;
    }
    .chat-response {
        padding: 12px;
        font-size: 14px;
        color: #000;
        line-height: 1.5;
        background-color: #fff;
    }
    
    /* BADGE */
    .badge-ai { background:#e3f2fd; color:#1565c0; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:10px; }
    .stButton>button { width: 100%; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- TỪ ĐIỂN NHÃN BỆNH (Map cho YOLO Export) ---
# Format: "Tên hiển thị": ID_YOLO
LABEL_MAPPING = {
    "Phổi / Bình thường (Normal)": 0,
    "Tim / Bóng tim to (Cardiomegaly)": 1,
    "Phổi / Viêm phổi (Pneumonia)": 2,
    "Màng phổi / Tràn dịch (Effusion)": 3,
    "Màng phổi / Tràn khí (Pneumothorax)": 4,
    "Phổi / Nốt - Khối mờ (Nodule/Mass)": 5,
    "Phổi / Xơ hóa - Lao (Fibrosis/TB)": 6,
    "Xương / Gãy xương (Fracture)": 7,
    "Màng phổi / Dày dính (Pleural Thickening)": 8,
    "Khác / Bệnh lý khác (Other)": 9
}
STRUCTURED_LABELS = list(LABEL_MAPPING.keys())

TECHNICAL_OPTS = ["✅ Phim đạt chuẩn", "⚠️ Chụp tại giường", "⚠️ Hít vào nông", "⚠️ Bệnh nhân xoay", "⚠️ Tia cứng/mềm", "⚠️ Dị vật/Áo"]
FEEDBACK_OPTS = ["Chưa đánh giá", "✅ Đồng thuận", "⚠️ Dương tính giả", "⚠️ Âm tính giả", "❌ Sai hoàn toàn"]
RATING_OPTS = ["Tệ", "TB", "Khá", "Tốt", "Xuất sắc"]

# --- KẾT NỐI SUPABASE ---
@st.cache_resource
def init_supabase():
    if "supabase" not in st.secrets: return None
    try: return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])
    except: return None

supabase = init_supabase()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DOCTOR_ROSTER = {
    "ANATOMY": "Dr_Anatomy.pt", "PNEUMOTHORAX": "Dr_Pneumothorax.pt", 
    "PNEUMONIA": "Dr_Pneumonia.pt", "TUMOR": "Dr_Tumor.pt",        
    "EFFUSION": "Dr_Effusion.pt", "OPACITY": "Dr_Opacity.pt", "HEART": "Dr_Heart.pt"         
}

@st.cache_resource
def load_models():
    loaded_models = {}
    for role, filename in DOCTOR_ROSTER.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try: loaded_models[role] = YOLO(path)
            except: pass
    return loaded_models

MODELS = load_models()

# --- HELPER FUNCTIONS ---
def check_password(password):
    # Hash của "Admin@123ptp"
    correct_hash = "25e4d273760a373b976d9102372d627c" # MD5
    return hashlib.md5(password.encode()).hexdigest() == correct_hash

def upload_image(img_cv, filename):
    if not supabase: return None
    try:
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
        bucket = "xray_images"
        supabase.storage.from_(bucket).upload(filename, buffer.tobytes(), {"content-type": "image/jpeg", "upsert": "true"})
        return supabase.storage.from_(bucket).get_public_url(filename)
    except:
        try: return supabase.storage.from_("xray_images").get_public_url(filename)
        except: return None

def save_log(data):
    if not supabase: return False
    try:
        supabase.table("logs").upsert(data).execute()
        return True
    except: return False

def get_logs():
    if not supabase: return pd.DataFrame()
    try:
        response = supabase.table("logs").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(response.data)
    except: return pd.DataFrame()

# --- GEMINI (V33.4 - Auto Retry & Prompt) ---
def ask_gemini(api_key, image, context="", note="", guide="", tags=[]):
    if not api_key: return {"labels": [], "reasoning": "Thiếu API Key", "prompt": ""}
    
    try:
        genai.configure(api_key=api_key)
        model_priority = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        
        labels_str = ", ".join(STRUCTURED_LABELS) 
        tech_note = ", ".join(tags) if tags else "Chuẩn."
        
        prompt = f"""
        Role: Senior Radiologist.
        INPUTS: 
        - Clinical Context: "{context}"
        - Expert Note: "{note}"
        - Technical QA: "{tech_note}"
        - Guidance: "{guide}"
        
        TASK: Analyze Chest X-ray. 
        Select closest labels from this list: {labels_str}.
        Provide reasoning in Vietnamese.
        
        OUTPUT JSON: {{ "labels": ["..."], "reasoning": "..." }}
        """
        
        last_error = ""
        for model_name in model_priority:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, image], generation_config={"response_mime_type": "application/json"})
                result = json.loads(response.text)
                result["used_model"] = model_name
                result["sent_prompt"] = prompt
                return result
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg:
                    time.sleep(1)
                    continue
                else:
                    last_error = err_msg
                    continue

        return {"labels": [], "reasoning": f"Lỗi: {last_error}", "sent_prompt": prompt}

    except Exception as e:
        return {"labels": [], "reasoning": f"System Error: {str(e)}", "sent_prompt": ""}

# --- PROCESS IMAGE ---
def process_and_save(image_file):
    filename = image_file.name.lower()
    img_rgb, patient_info = None, "Nguyễn Văn A (Demo)"
    image_file.seek(0)
    
    if filename.endswith(('.dcm', '.dicom')):
        try:
            ds = pydicom.dcmread(image_file)
            patient_info = str(ds.get("PatientName", "Anonymous")).replace('^', ' ').strip()
            img = ds.pixel_array.astype(float)
            img = (np.maximum(img, 0) / img.max()) * 255.0
            img = np.uint8(img)
            if ds.get("PhotometricInterpretation") == "MONOCHROME1": img = 255 - img
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
        except: return None, {"Error": "Lỗi DICOM"}, False, None, None
    else:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
        if img_cv is not None: img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    if img_rgb is None: return None, {"Error": "Lỗi File"}, False, None, None

    h, w = img_rgb.shape[:2]
    scale = 1024 / max(h, w)
    img_resized = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
    display_img = img_resized.copy()
    
    findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    has_danger = False

    if "ANATOMY" in MODELS:
        try:
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            anatomy_res = MODELS["ANATOMY"](img_bgr, conf=0.35, verbose=False)[0]
            for box in anatomy_res.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                region = anatomy_res.names[int(box.cls[0])]
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
                            has_danger = True if pct > 75 else has_danger
                            text = f"{region}: {spec} ({pct:.0f}%)"
                            if "HEART" in spec: findings_db["Heart"].append(text)
                            elif "PLEURA" in spec or "EFFUSION" in spec: findings_db["Pleura"].append(text)
                            else: findings_db["Lung"].append(text)
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255,0,0), 2)
                            cv2.putText(display_img, spec[:4], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        except: pass
    else: findings_db["Lung"].append("Chế độ Test (No Model)")

    img_id = datetime.now().strftime("%d%m%Y%H%M%S")
    img_url = upload_image(display_img, f"XRAY_{img_id}.jpg")
    if img_url:
        save_log({"id": img_id, "created_at": datetime.now().isoformat(), "image_url": img_url, "result": "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG", "details": str(findings_db), "patient_info": patient_info})
    return display_img, findings_db, has_danger, img_id, Image.fromarray(img_resized)

# ================= UI CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("ĐIỀU KHIỂN")
    api_key = st.text_input("🔑 Gemini API Key:", value=st.secrets.get("GEMINI_API_KEY", ""), type="password")
    mode = st.radio("Menu:", ["🔍 Phân Tích & In Phiếu", "📂 Hội Chẩn (Cloud)", "🛠️ Xuất Dataset (Admin)"])

# ... (Tab 1: Phân Tích giữ nguyên như cũ cho gọn code) ...
if mode == "🔍 Phân Tích & In Phiếu":
    st.title("🏥 TRỢ LÝ CHẨN ĐOÁN (A4)")
    uploaded_file = st.file_uploader("Chọn ảnh X-quang:", type=["jpg", "png", "jpeg", "dcm"])
    if uploaded_file and st.button("🚀 PHÂN TÍCH", type="primary"):
        with st.spinner("Đang xử lý..."):
            img_out, findings, danger, img_id, pil_img = process_and_save(uploaded_file)
            if img_out:
                st.image(img_out, caption=f"ID: {img_id}", use_container_width=True)
                st.success("Đã phân tích xong và lưu vào Cloud.")
            else: st.error("Lỗi.")

# ... (TAB 2: HỘI CHẨN - GIAO DIỆN KHOA HỌC) ...
elif mode == "📂 Hội Chẩn (Cloud)":
    st.title("📂 HỘI CHẨN CHUYÊN GIA")
    
    if not supabase: st.error("⛔ Chưa kết nối Cloud.")
    else:
        df = get_logs()
        if not df.empty:
            df = df.fillna("")
            id_list = df['id'].tolist()
            
            # Layout chọn ID
            selected_id = st.selectbox("👉 Chọn Mã Hồ Sơ Bệnh Án:", id_list)
            
            if selected_id:
                record = df[df["id"] == selected_id].iloc[0]
                
                # Load ảnh
                pil_img = None
                if record.get('image_url'):
                    try: pil_img = Image.open(BytesIO(requests.get(record['image_url'], timeout=5).content))
                    except: pass
                
                # --- CHIA 2 CỘT: TRÁI (ẢNH + KẾT QUẢ) - PHẢI (BÀN LÀM VIỆC) ---
                col_left, col_right = st.columns([1.2, 1])
                
                with col_left:
                    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
                    st.markdown('<div class="sci-header">🖼️ HÌNH ẢNH & KẾT QUẢ MÁY</div>', unsafe_allow_html=True)
                    if record.get('image_url'):
                        st.image(record['image_url'], use_container_width=True)
                    
                    st.info(f"**Bệnh nhân:** {record.get('patient_info')}")
                    
                    res_yolo = record.get('result')
                    color = "red" if res_yolo == "BẤT THƯỜNG" else "green"
                    st.markdown(f"**Sàng lọc YOLO:** <span style='color:{color}; font-weight:bold'>{res_yolo}</span>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_right:
                    # --- CARD 1: LÂM SÀNG ---
                    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
                    st.markdown('<div class="sci-header">📝 DỮ LIỆU LÂM SÀNG</div>', unsafe_allow_html=True)
                    ctx = st.text_area("Bệnh cảnh:", value=record.get("clinical_context") or "", height=100)
                    note = st.text_area("Ý kiến chuyên gia:", value=record.get("expert_note") or "", height=70)
                    guide = st.text_area("Prompt Guide:", value=record.get("prompt_guidance") or "", height=70)
                    tags = st.multiselect("Kỹ thuật:", TECHNICAL_OPTS, default=[t.strip() for t in (record.get("technical_tags") or "").split(";") if t])
                    
                    # --- NÚT GEMINI ---
                    if st.button("🧠 HỎI GEMINI (Lưu Nhật Ký)"):
                        if not api_key: st.error("Thiếu API Key")
                        else:
                            with st.spinner("Gemini đang suy nghĩ..."):
                                res = ask_gemini(api_key, pil_img, ctx, note, guide, tags)
                                txt = res.get("reasoning", "")
                                if txt:
                                    # Update JSON Log
                                    hist = record.get('ai_reasoning', [])
                                    if isinstance(hist, str): 
                                        try: hist = json.loads(hist)
                                        except: hist = []
                                    
                                    new_entry = {
                                        "time": datetime.now().strftime("%H:%M %d/%m"),
                                        "prompt": res.get("sent_prompt", ""),
                                        "response": txt,
                                        "model": res.get("used_model", "AI")
                                    }
                                    hist.insert(0, new_entry)
                                    save_log({"id": selected_id, "ai_reasoning": json.dumps(hist)})
                                    st.success("Đã cập nhật!")
                                    time.sleep(0.5); st.rerun()
                                else: st.error(f"Lỗi: {res}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # --- CARD 2: NHẬT KÝ CHAT (ĐẸP) ---
                    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
                    st.markdown('<div class="sci-header">💬 NHẬT KÝ HỘI CHẨN</div>', unsafe_allow_html=True)
                    
                    hist_data = record.get('ai_reasoning', [])
                    if isinstance(hist_data, str):
                        try: hist_data = json.loads(hist_data)
                        except: hist_data = [] # Handle old format
                    
                    if not hist_data:
                        st.caption("Chưa có dữ liệu.")
                    else:
                        for item in hist_data:
                            with st.container():
                                st.markdown(f"""
                                <div class="chat-box">
                                    <div class="chat-meta">
                                        <span>⏰ {item.get('time','')}</span>
                                        <span class="badge-ai">{item.get('model','Gemini')}</span>
                                    </div>
                                    <div class="chat-prompt">❓ {item.get('prompt','')}</div>
                                    <div class="chat-response">🤖 {item.get('response','')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # --- CARD 3: GÁN NHÃN (QUAN TRỌNG) ---
                    st.markdown('<div class="sci-card" style="border-top-color: #ff9800;">', unsafe_allow_html=True)
                    st.markdown('<div class="sci-header">🏷️ KẾT LUẬN & GÁN NHÃN</div>', unsafe_allow_html=True)
                    
                    # LOGIC AUTO-SELECT THÔNG MINH
                    saved_lbls = [l.strip() for l in (record.get("label_1") or "").split(";") if l]
                    
                    # Nếu chưa có nhãn lưu, thử tìm trong kết quả Gemini mới nhất để gợi ý
                    if not saved_lbls and hist_data:
                        last_resp = hist_data[0].get("response", "")
                        for sl in STRUCTURED_LABELS:
                            # Tìm tên bệnh trong text (đơn giản)
                            clean_name = sl.split("(")[0].split("/")[-1].strip() # Lấy "Viêm phổi"
                            if clean_name.lower() in last_resp.lower():
                                saved_lbls.append(sl)
                    
                    # Lọc chỉ lấy những nhãn hợp lệ trong list mới
                    valid_defaults = [l for l in saved_lbls if l in STRUCTURED_LABELS]

                    new_fb = st.radio("Đánh giá AI:", FEEDBACK_OPTS, index=0, horizontal=True)
                    new_lbls = st.multiselect("Chốt bệnh (Auto-Fill):", STRUCTURED_LABELS, default=valid_defaults)
                    
                    safe_rating = record.get("prompt_rating") if record.get("prompt_rating") in RATING_OPTS else "Khá"
                    rating = st.select_slider("Chất lượng Prompt:", options=RATING_OPTS, value=safe_rating)
                    
                    if st.button("💾 LƯU KẾT QUẢ"):
                        save_log({
                            "id": selected_id, 
                            "clinical_context": ctx, "expert_note": note, "prompt_guidance": guide, 
                            "technical_tags": "; ".join(tags), 
                            "feedback_1": new_fb, "label_1": "; ".join(new_lbls), 
                            "prompt_rating": rating
                        })
                        st.success("Đã lưu hồ sơ!")
                    st.markdown('</div>', unsafe_allow_html=True)

        else: st.warning("📭 Database trống.")

# ... (TAB 3: XUẤT DATASET YOLO - BẢO MẬT & XỊN XÒ) ...
elif mode == "🛠️ Xuất Dataset (Admin)":
    st.title("🛠️ XUẤT DATASET YOLO (Admin Only)")
    
    col_auth, col_empty = st.columns([1, 2])
    with col_auth:
        pwd = st.text_input("Nhập mật khẩu quản trị:", type="password")
        
    if pwd:
        if check_password(pwd):
            st.success("✅ Đăng nhập thành công! (Hash Verified)")
            
            df = get_logs()
            if not df.empty:
                st.markdown("### 📊 Xem trước dữ liệu")
                st.dataframe(df.head(5))
                
                # MÔ PHỎNG ẢNH
                st.markdown("### 👁️ Mô phỏng ảnh xuất ra")
                sample = df.iloc[0]
                c1, c2 = st.columns([1, 2])
                with c1:
                    if sample.get('image_url'):
                        st.image(sample['image_url'], caption="Ảnh gốc")
                with c2:
                    st.markdown(f"**Filename:** `image_{sample['id']}.jpg`")
                    st.markdown(f"**YOLO Label (.txt):** `image_{sample['id']}.txt`")
                    
                    # Logic tạo nội dung file txt giả lập (Vì chưa có tọa độ thật)
                    lbls = str(sample.get('label_1') or "")
                    txt_preview = ""
                    for l in lbls.split(";"):
                        l = l.strip()
                        if l in LABEL_MAPPING:
                            class_id = LABEL_MAPPING[l]
                            # Giả lập Box full ảnh (0.5 0.5 1.0 1.0) để người dùng tự sửa sau
                            txt_preview += f"{class_id} 0.5000 0.5000 1.0000 1.0000\n"
                    
                    st.text_area("Nội dung file .txt (Mô phỏng):", value=txt_preview, height=100)
                    st.warning("⚠️ Lưu ý: Tọa độ Box đang để mặc định (Full ảnh) vì Database chưa lưu tọa độ chi tiết.")

                if st.button("📦 TẢI VỀ DATASET (ZIP)"):
                    with st.spinner("Đang tải ảnh và tạo Dataset YOLO..."):
                        # Tạo file Zip trong bộ nhớ
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zf:
                            # Tạo file classes.txt
                            class_content = "\n".join([f"{k}" for k in LABEL_MAPPING.keys()])
                            zf.writestr("classes.txt", class_content)
                            
                            progress_bar = st.progress(0)
                            total = len(df)
                            
                            for idx, row in df.iterrows():
                                img_url = row.get('image_url')
                                img_id = row['id']
                                if img_url:
                                    try:
                                        # Tải ảnh
                                        img_data = requests.get(img_url, timeout=5).content
                                        zf.writestr(f"images/image_{img_id}.jpg", img_data)
                                        
                                        # Tạo label txt
                                        labels_str = str(row.get('label_1') or "")
                                        label_content = ""
                                        for l in labels_str.split(";"):
                                            l = l.strip()
                                            if l in LABEL_MAPPING:
                                                cid = LABEL_MAPPING[l]
                                                label_content += f"{cid} 0.5 0.5 1.0 1.0\n"
                                        
                                        zf.writestr(f"labels/image_{img_id}.txt", label_content)
                                    except: pass
                                
                                progress_bar.progress((idx + 1) / total)
                        
                        st.download_button(
                            label="📥 CLICK ĐỂ TẢI DATASET.ZIP",
                            data=zip_buffer.getvalue(),
                            file_name="yolo_dataset_xray.zip",
                            mime="application/zip",
                            type="primary"
                        )
            else:
                st.info("Chưa có dữ liệu.")
        else:
            st.error("⛔ Mật khẩu sai!")