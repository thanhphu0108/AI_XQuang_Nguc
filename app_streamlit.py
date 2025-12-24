import streamlit as st
import subprocess
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
from datetime import datetime, timedelta
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

# ================= 1. CẤU HÌNH & CSS (COMPACT & CLEAN) =================
st.set_page_config(page_title="AI Hospital (V34.3 - VN Time)", page_icon="🇻🇳", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* 1. KHUNG LABELING (GỌN GÀNG - KHÔNG KHOẢNG TRẮNG) */
    .labeling-box {
        background-color: #fff8e1;
        border: 2px solid #ffb74d;
        border-radius: 8px;
        padding: 10px 15px; /* Giảm padding dọc */
        margin-top: 5px;
        margin-bottom: 10px;
    }
    .labeling-header {
        font-weight: bold; color: #e65100; border-bottom: 1px dashed #ffb74d; 
        margin-bottom: 8px; font-size: 14px; text-transform: uppercase;
    }

    /* 2. KẾT QUẢ GEMINI (HIỆN FULL) */
    .gemini-full-box {
        background-color: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        font-family: 'Segoe UI', sans-serif;
        color: #1b5e20;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* 3. INPUT COMPACT */
    .stTextArea textarea { font-size: 13px; min-height: 80px; }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 6px; }
    
    /* 4. CARD ẢNH */
    .img-card { background: white; padding: 10px; border-radius: 8px; border: 1px solid #ddd; text-align: center; margin-bottom: 10px; }
    
    /* 5. HISTORY ITEM */
    .history-item {
        border-left: 3px solid #ccc; padding-left: 10px; margin-bottom: 8px; font-size: 13px; color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- TỪ ĐIỂN ---
LABEL_MAPPING = {
    "Phổi / Bình thường (Normal)": 0, "Tim / Bóng tim to (Cardiomegaly)": 1, "Phổi / Viêm phổi (Pneumonia)": 2,
    "Màng phổi / Tràn dịch (Effusion)": 3, "Màng phổi / Tràn khí (Pneumothorax)": 4, "Phổi / Nốt - Khối mờ (Nodule/Mass)": 5,
    "Phổi / Xơ hóa - Lao (Fibrosis/TB)": 6, "Xương / Gãy xương (Fracture)": 7, "Màng phổi / Dày dính (Pleural Thickening)": 8,
    "Khác / Bệnh lý khác (Other)": 9
}
STRUCTURED_LABELS = list(LABEL_MAPPING.keys())
TECHNICAL_OPTS = ["✅ Phim đạt chuẩn kỹ thuật", "⚠️ Chụp tại giường (AP)", "⚠️ Hít vào không đủ sâu", "⚠️ Bệnh nhân xoay lệch", "⚠️ Tia cứng/mềm", "⚠️ Dị vật/Áo"]
FEEDBACK_OPTS = ["Chưa đánh giá", "✅ Đồng thuận", "⚠️ Dương tính giả", "⚠️ Âm tính giả", "❌ Sai hoàn toàn"]
RATING_OPTS = ["Tệ", "TB", "Khá", "Tốt", "Xuất sắc"]

# --- HÀM THỜI GIAN VIỆT NAM (UTC+7) ---
def get_vn_time():
    return (datetime.utcnow() + timedelta(hours=7)).strftime("%H:%M %d/%m")

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
    "ANATOMY": "Dr_Anatomy.pt", "PNEUMOTHORAX": "Dr_Pneumothorax.pt", "PNEUMONIA": "Dr_Pneumonia.pt", 
    "TUMOR": "Dr_Tumor.pt", "EFFUSION": "Dr_Effusion.pt", "OPACITY": "Dr_Opacity.pt", "HEART": "Dr_Heart.pt"         
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

# --- UTILS ---
def check_password(password):
    return hashlib.md5(password.encode()).hexdigest() == "25e4d273760a373b976d9102372d627c"

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

# --- GEMINI ---
def ask_gemini(api_key, image, context="", note="", guide="", tags=[]):
    if not api_key: return {"labels": [], "reasoning": "Thiếu API Key", "prompt": ""}
    
    try:
        genai.configure(api_key=api_key)
        model_priority = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        
        labels_str = ", ".join(STRUCTURED_LABELS) 
        tech_note = ", ".join(tags) if tags else "Phim đạt chuẩn kỹ thuật."
        
        prompt = f"""
Vai trò: Bác sĩ chẩn đoán hình ảnh chuyên sâu (Senior Radiologist).

==== 1. DỮ LIỆU ĐẦU VÀO ====
- BỆNH CẢNH (Context): "{context}"
- GHI CHÚ CHUYÊN GIA (Expert Note): "{note}"
- HƯỚNG DẪN CỤ THỂ (Guidance): "{guide}"

==== 2. ĐIỀU KIỆN KỸ THUẬT (QA/QC) QUAN TRỌNG ====
- Trạng thái phim: {tech_note}
(Lưu ý: Hãy cân nhắc các yếu tố kỹ thuật trên để tránh Dương tính giả/Âm tính giả).

==== 3. NHIỆM VỤ ====
- Phân tích hình ảnh X-quang đính kèm.
- Chọn nhãn bệnh lý chính xác từ danh sách: [{labels_str}].
- Nếu bình thường, chọn 'Bình thường (Normal)'.

OUTPUT JSON FORMAT:
{{
  "labels": ["Label1", "Label2"],
  "reasoning": "VIẾT THEO CẤU TRÚC SAU (BẮT BUỘC):\\nKỹ thuật: ...\\nMô tả:\\n- Bóng tim: ...\\n- Nhu mô phổi: ...\\n- Màng phổi: ...\\n- Xương và phần mềm: ...\\nBiện luận: ... (Kết hợp hình ảnh và lâm sàng)\\nKết luận: (Gạch đầu dòng các bệnh lý)"
}}
        """
        
        for model_name in model_priority:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, image], generation_config={"response_mime_type": "application/json"})
                result = json.loads(response.text)
                result["used_model"] = model_name
                result["sent_prompt"] = prompt
                return result
            except Exception as e:
                if "429" in str(e): time.sleep(1); continue
                elif "API_KEY_INVALID" in str(e) or "expired" in str(e):
                    return {"labels": [], "reasoning": "🔑 KEY HẾT HẠN! Vui lòng đổi Key mới.", "prompt": ""}
                else: continue

        return {"labels": [], "reasoning": "Hệ thống bận, vui lòng thử lại.", "sent_prompt": prompt}

    except Exception as e:
        return {"labels": [], "reasoning": f"Lỗi: {str(e)}", "sent_prompt": ""}

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
        # Lưu thời gian VN khi tạo mới
        vn_time = get_vn_time()
        save_log({"id": img_id, "created_at": datetime.now().isoformat(), "image_url": img_url, "result": "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG", "details": str(findings_db), "patient_info": patient_info})
    return display_img, findings_db, has_danger, img_id, Image.fromarray(img_resized)

# ================= UI CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("ĐIỀU KHIỂN")
    api_key = st.text_input("🔑 Gemini API Key:", value=st.secrets.get("GEMINI_API_KEY", ""), type="password")
    mode = st.radio("Menu:", ["🔍 Phân Tích & In Phiếu", "📂 Hội Chẩn (Cloud)", "🛠️ Xuất Dataset (Admin)"])

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

elif mode == "📂 Hội Chẩn (Cloud)":
    # st.title("📂 HỘI CHẨN CHUYÊN GIA")
    
    if not supabase: st.error("⛔ Chưa kết nối Cloud.")
    else:
        df = get_logs()
        if not df.empty:
            df = df.fillna("")
            id_list = df['id'].tolist()
            
            c_sel, c_blank = st.columns([1, 2])
            with c_sel:
                selected_id = st.selectbox("Mã Hồ Sơ:", id_list, label_visibility="collapsed")
            
            if selected_id:
                record = df[df["id"] == selected_id].iloc[0]
                
                pil_img = None
                if record.get('image_url'):
                    try: pil_img = Image.open(BytesIO(requests.get(record['image_url'], timeout=5).content))
                    except: pass
                
                hist_data = record.get('ai_reasoning', [])
                if isinstance(hist_data, str):
                    try: hist_data = json.loads(hist_data)
                    except: hist_data = []
                
                col_left, col_right = st.columns([1, 1.1])
                
                # === CỘT TRÁI: ẢNH + LABELING (ĐÃ XÓA DƯ) ===
                with col_left:
                    # 1. ẢNH
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    if record.get('image_url'): st.image(record['image_url'], use_container_width=True)
                    res_yolo = record.get('result')
                    color = "red" if res_yolo == "BẤT THƯỜNG" else "green"
                    st.caption(f"YOLO: {res_yolo} | BN: {record.get('patient_info')}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 2. KHUNG GÁN NHÃN (CLEAN UI)
                    st.markdown('<div class="labeling-box">', unsafe_allow_html=True)
                    st.markdown('<div class="labeling-header">🏷️ CHỐT KẾT QUẢ</div>', unsafe_allow_html=True)
                    
                    # Logic Auto-Select
                    saved_lbls = [l.strip() for l in (record.get("label_1") or "").split(";") if l]
                    if not saved_lbls and hist_data:
                        last_resp = hist_data[0].get("response", "")
                        for sl in STRUCTURED_LABELS:
                            clean_name = sl.split("(")[0].split("/")[-1].strip()
                            if clean_name.lower() in last_resp.lower(): saved_lbls.append(sl)
                    
                    # Layout 2 cột cho Radio và Slider để tiết kiệm chỗ
                    c1, c2 = st.columns([1.5, 1])
                    with c1: 
                        new_fb = st.radio("Đánh giá AI:", FEEDBACK_OPTS, index=0)
                    with c2: 
                        rating = st.select_slider("Chất lượng:", options=RATING_OPTS, value="Khá")
                    
                    st.caption("Bệnh lý (Đa chọn):")
                    new_lbls = st.multiselect("", STRUCTURED_LABELS, default=[l for l in saved_lbls if l in STRUCTURED_LABELS], label_visibility="collapsed")
                    
                    st.markdown("---")
                    # NÚT LƯU
                    if st.button("💾 LƯU KẾT QUẢ (SAVE)", type="primary", use_container_width=True):
                        # Lấy ctx, note hiện tại
                        save_log({
                            "id": selected_id, "feedback_1": new_fb, "label_1": "; ".join(new_lbls), "prompt_rating": rating
                        })
                        st.success("✅ Đã lưu kết quả!")
                    st.markdown('</div>', unsafe_allow_html=True)

                # === CỘT PHẢI: INPUT -> NÚT HỎI -> KẾT QUẢ FULL ===
                with col_right:
                    # 1. INPUTS
                    c_in1, c_in2 = st.columns(2)
                    with c_in1:
                        ctx = st.text_area("Bệnh cảnh:", value=record.get("clinical_context") or "", height=70)
                        guide = st.text_area("Prompt:", value=record.get("prompt_guidance") or "", height=70)
                    with c_in2:
                        note = st.text_area("Ghi chú:", value=record.get("expert_note") or "", height=70)
                        tags = st.multiselect("Kỹ thuật:", TECHNICAL_OPTS, default=[t.strip() for t in (record.get("technical_tags") or "").split(";") if t])
                    
                    if st.button("Lưu thông tin lâm sàng", key="save_info"):
                        save_log({"id": selected_id, "clinical_context": ctx, "expert_note": note, "prompt_guidance": guide, "technical_tags": "; ".join(tags)})
                        st.toast("Đã lưu thông tin!")

                    # 2. NÚT HỎI GEMINI
                    st.markdown("---")
                    if st.button("🧠 HỎI GEMINI (PHÂN TÍCH)", type="secondary", use_container_width=True):
                        if not api_key: st.error("Thiếu API Key")
                        else:
                            with st.spinner("Gemini đang phân tích..."):
                                res = ask_gemini(api_key, pil_img, ctx, note, guide, tags)
                                txt = res.get("reasoning", "")
                                if txt:
                                    if "KEY HẾT HẠN" in txt: st.error(txt)
                                    else:
                                        # Lấy giờ VN
                                        vn_time = get_vn_time()
                                        new_entry = {
                                            "time": vn_time,
                                            "prompt": res.get("sent_prompt", ""),
                                            "response": txt,
                                            "model": res.get("used_model", "AI")
                                        }
                                        hist_data.insert(0, new_entry)
                                        save_log({"id": selected_id, "ai_reasoning": json.dumps(hist_data)})
                                        st.success("Đã cập nhật!")
                                        time.sleep(0.5); st.rerun()
                                else: st.error(f"Lỗi: {res}")

                    # 3. KẾT QUẢ MỚI NHẤT (FULL)
                    if hist_data:
                        last_item = hist_data[0]
                        model = last_item.get('model', 'Gemini')
                        resp = last_item.get('response', '').replace("\n", "<br>")
                        
                        st.markdown(f"""
                        <div class="gemini-full-box">
                            <strong>🤖 KẾT QUẢ MỚI NHẤT ({model})</strong><br>
                            <hr style="margin:5px 0; border-color:#c8e6c9">
                            {resp}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 4. NHẬT KÝ (LỊCH SỬ CŨ) - HIỆN CHI TIẾT KẾT LUẬN
                        if len(hist_data) > 1:
                            st.markdown("---")
                            st.caption("📜 Nhật ký chẩn đoán (Cũ hơn):")
                            for item in hist_data[1:]:
                                with st.expander(f"🕒 {item.get('time')} - {item.get('model')}"):
                                    # Hiện nội dung kết luận chi tiết
                                    st.markdown(item.get('response', '').replace("\n", "<br>"), unsafe_allow_html=True)

        else: st.warning("📭 Database trống.")

elif mode == "🛠️ Xuất Dataset (Admin)":
    st.title("🛠️ XUẤT DATASET YOLO (Admin Only)")
    col_auth, col_empty = st.columns([1, 2])
    with col_auth: pwd = st.text_input("Nhập mật khẩu quản trị:", type="password")
    if pwd:
        if check_password(pwd):
            st.success("✅ Verified")
            df = get_logs()
            if not df.empty:
                st.dataframe(df.head(5))
                if st.button("📦 TẢI DATASET (ZIP)"):
                    with st.spinner("Đang tạo..."):
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zf:
                            zf.writestr("classes.txt", "\n".join(LABEL_MAPPING.keys()))
                            progress_bar = st.progress(0); total = len(df)
                            for idx, row in df.iterrows():
                                img_url = row.get('image_url'); img_id = row['id']
                                if img_url:
                                    try:
                                        zf.writestr(f"images/image_{img_id}.jpg", requests.get(img_url, timeout=5).content)
                                        txt_cont = ""
                                        for l in str(row.get('label_1') or "").split(";"):
                                            if l.strip() in LABEL_MAPPING: txt_cont += f"{LABEL_MAPPING[l.strip()]} 0.5 0.5 1.0 1.0\n"
                                        zf.writestr(f"labels/image_{img_id}.txt", txt_cont)
                                    except: pass
                                progress_bar.progress((idx + 1) / total)
                        st.download_button("📥 TẢI XUỐNG", zip_buffer.getvalue(), "dataset.zip", "application/zip")
        else: st.error("Sai mật khẩu!")