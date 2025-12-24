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

# ================= 1. CẤU HÌNH & CSS (CHUẨN & THOÁNG) =================
st.set_page_config(page_title="AI Hospital (V34.6 - Final UI)", page_icon="🇻🇳", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
    
    /* 1. KHUNG LABELING (GỌN NHƯNG KHÔNG ÉP) */
    .labeling-box {
        background-color: #fff8e1; border: 2px solid #ffb74d; border-radius: 6px;
        padding: 10px 15px; margin-top: 10px; margin-bottom: 10px;
    }
    .labeling-header {
        font-weight: bold; color: #e65100; border-bottom: 1px dashed #ffb74d; 
        margin-bottom: 10px; font-size: 14px; text-transform: uppercase;
    }
    
    /* 2. KHUNG KẾT QUẢ GEMINI (CỘT PHẢI) */
    .gemini-full-box {
        background-color: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        font-family: 'Segoe UI', sans-serif;
        color: #1b5e20;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* 3. TITLE INPUT */
    .input-title {
        font-size: 16px; font-weight: bold; color: #333; margin-top: 0px; margin-bottom: 10px; text-transform: uppercase;
    }
    
    /* 4. CARD ẢNH */
    .img-card { background: white; padding: 5px; border-radius: 8px; border: 1px solid #ddd; text-align: center; margin-bottom: 10px; }
    
    /* 5. HISTORY ITEM */
    .history-item {
        border-left: 4px solid #ccc; padding-left: 10px; margin-bottom: 8px; font-size: 13px; color: #555; background: white; padding: 8px; border-radius: 4px;
    }
    
    /* 6. BUTTONS */
    .stButton>button { width: 100%; font-weight: bold; border-radius: 6px; height: 45px; }
    
    /* 7. POPUP */
    div[role="dialog"][aria-modal="true"] { width: 90vw !important; max-width: 90vw !important; }
    .popup-result-box { background: #f1f8e9; padding: 20px; border-radius: 8px; color: #1b5e20; line-height: 1.6; font-size: 15px; }
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

# --- HÀM THỜI GIAN VN ---
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
    except: return None

def save_log(data):
    if not supabase: return False
    try: supabase.table("logs").upsert(data).execute(); return True
    except: return False

def get_logs():
    if not supabase: return pd.DataFrame()
    try: return pd.DataFrame(supabase.table("logs").select("*").order("created_at", desc=True).execute().data)
    except: return pd.DataFrame()

# --- POPUP DIALOG ---
@st.dialog("📋 CHI TIẾT HỘI CHẨN (FULL SCREEN)", width="large")
def view_log_popup(item):
    st.markdown(f"**Thời gian:** {item.get('time')} | **Model:** {item.get('model')}")
    st.markdown("### 🤖 KẾT LUẬN CHI TIẾT")
    st.markdown(f"""<div class="popup-result-box">{item.get('response', '').replace("\n", "<br>")}</div>""", unsafe_allow_html=True)
    with st.expander("🔌 Debug: Xem nội dung Prompt đã gửi đi"): st.code(item.get('prompt', ''), language="text")

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
                elif "API_KEY" in str(e): return {"labels": [], "reasoning": "🔑 KEY HẾT HẠN! Vui lòng đổi Key mới.", "prompt": ""}
                else: continue
        return {"labels": [], "reasoning": "Hệ thống bận, vui lòng thử lại.", "sent_prompt": prompt}
    except Exception as e: return {"labels": [], "reasoning": str(e), "sent_prompt": ""}

# --- PROCESS IMAGE ---
def process_and_save(image_file):
    filename = image_file.name.lower()
    img_rgb, patient_info = None, "Nguyễn Văn A (Demo)"
    image_file.seek(0)
    try:
        if filename.endswith(('.dcm', '.dicom')):
            ds = pydicom.dcmread(image_file)
            img = ds.pixel_array.astype(float)
            img = (np.maximum(img, 0) / img.max()) * 255.0
            img_rgb = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2RGB)
        else:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    except: return None, {}, False, None, None

    h, w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (int(w*(1024/max(h,w))), int(h*(1024/max(h,w)))))
    display_img = img_resized.copy()
    findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    has_danger = False

    if "ANATOMY" in MODELS:
        try:
            anatomy_res = MODELS["ANATOMY"](display_img, conf=0.35, verbose=False)[0]
            for box in anatomy_res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                region = anatomy_res.names[int(box.cls[0])]
                roi = display_img[max(0, y1-20):min(img_resized.shape[0], y2+20), max(0, x1-20):min(img_resized.shape[1], x2+20)]
                target_models = ["HEART"] if "Heart" in region else ["PNEUMOTHORAX", "EFFUSION", "PNEUMONIA", "TUMOR"]
                for spec in target_models:
                    if spec in MODELS:
                        res = MODELS[spec](roi, verbose=False)[0]
                        if res.probs.top1conf.item() > 0.6 and res.names[res.probs.top1] == "Disease":
                            has_danger = True
                            text = f"{region}: {spec} ({res.probs.top1conf.item()*100:.0f}%)"
                            if "HEART" in spec: findings_db["Heart"].append(text)
                            elif "PLEURA" in spec or "EFFUSION" in spec: findings_db["Pleura"].append(text)
                            else: findings_db["Lung"].append(text)
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255,0,0), 2)
        except: pass

    img_id = datetime.now().strftime("%d%m%Y%H%M%S")
    img_url = upload_image(display_img, f"XRAY_{img_id}.jpg")
    if img_url: save_log({"id": img_id, "created_at": datetime.now().isoformat(), "image_url": img_url, "result": "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG", "details": str(findings_db), "patient_info": patient_info})
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
    if uploaded_file and st.button("🚀 PHÂN TÍCH"):
        with st.spinner("Đang chạy AI Nội bộ..."):
            img_out, findings, danger, img_id, pil_img = process_and_save(uploaded_file)
            if img_out is not None:
                c1, c2 = st.columns(2)
                with c1: st.image(img_out, caption=f"ID: {img_id}", use_container_width=True)
                with c2: st.success("Đã phân tích xong! (A4 Mode)") # Giản lược cho gọn code
            else: st.error("Lỗi file.")

elif mode == "📂 Hội Chẩn (Cloud)":
    if not supabase: st.error("⛔ Chưa kết nối Cloud.")
    else:
        df = get_logs()
        if not df.empty:
            df = df.fillna("")
            id_list = df['id'].tolist()
            
            # --- THANH CHỌN HỒ SƠ (HIỆN LẠI ĐỂ LẤP KHOẢNG TRỐNG) ---
            selected_id = st.selectbox("👉 Chọn Mã Hồ Sơ Bệnh Án:", id_list)
            
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
                
                # --- CHIA 2 CỘT: 45/55 ---
                col_left, col_right = st.columns([1, 1.2])
                
                # === CỘT TRÁI: ẢNH + LABELING (THOÁNG HƠN) ===
                with col_left:
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    if record.get('image_url'): st.image(record['image_url'], use_container_width=True)
                    res_yolo = record.get('result')
                    color = "red" if res_yolo == "BẤT THƯỜNG" else "green"
                    st.caption(f"YOLO: {res_yolo} | BN: {record.get('patient_info')}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="labeling-box">', unsafe_allow_html=True)
                    st.markdown('<div class="labeling-header">🏷️ KẾT LUẬN & GÁN NHÃN</div>', unsafe_allow_html=True)
                    
                    # Auto-fill
                    saved_lbls = [l.strip() for l in (record.get("label_1") or "").split(";") if l]
                    if not saved_lbls and hist_data:
                        last_resp = hist_data[0].get("response", "")
                        for sl in STRUCTURED_LABELS:
                            if sl.split("(")[0].split("/")[-1].strip().lower() in last_resp.lower(): saved_lbls.append(sl)
                    
                    # Layout thoáng hơn cho Radio/Slider
                    st.caption("Đánh giá AI & Chất lượng Prompt:")
                    c1, c2 = st.columns([1.5, 1])
                    with c1: new_fb = st.radio("Feedback", FEEDBACK_OPTS, index=0, label_visibility="collapsed")
                    with c2: rating = st.select_slider("Rating", options=RATING_OPTS, value="Khá", label_visibility="collapsed")
                    
                    st.caption("Chốt bệnh lý:")
                    new_lbls = st.multiselect("Disease", STRUCTURED_LABELS, default=[l for l in saved_lbls if l in STRUCTURED_LABELS], label_visibility="collapsed")
                    
                    st.markdown("---")
                    if st.button("💾 LƯU KẾT QUẢ", type="primary", use_container_width=True):
                        # Khi lưu kết quả, lưu luôn context hiện tại (để tránh mất)
                        # Lấy giá trị từ session state hoặc giả định người dùng đã nhập
                        # Lưu ý: Trong Streamlit, giá trị widget bên phải sẽ được gửi về khi bấm nút bên trái nếu form chưa clear.
                        save_log({
                            "id": selected_id, "feedback_1": new_fb, "label_1": "; ".join(new_lbls), "prompt_rating": rating
                        })
                        st.success("✅ Đã lưu!")
                    st.markdown('</div>', unsafe_allow_html=True)

                # === CỘT PHẢI: INPUT THEO ẢNH MẪU ===
                with col_right:
                    st.markdown('<div class="input-title">1. DỮ LIỆU ĐẦU VÀO</div>', unsafe_allow_html=True)
                    
                    # 1. KỸ THUẬT (ĐƯA LÊN ĐẦU)
                    tags = st.multiselect("⚙️ Điều kiện kỹ thuật (QA/QC - Gửi kèm cho AI):", TECHNICAL_OPTS, default=[t.strip() for t in (record.get("technical_tags") or "").split(";") if t])
                    
                    # 2. INPUTS (ĐÚNG TỪ NGỮ)
                    ctx = st.text_area("🤒 Bệnh cảnh (Context):", value=record.get("clinical_context") or "", height=80)
                    note = st.text_area("👨‍⚕️ Ý kiến chuyên gia (Ghi chú ban đầu):", value=record.get("expert_note") or "", height=60)
                    guide = st.text_area("📝 Dẫn dắt AI (Prompt/Yêu cầu):", value=record.get("prompt_guidance") or "", height=60)
                    
                    # 3. NÚT HỎI (TỰ LƯU CONTEXT)
                    st.markdown("---")
                    if st.button("🧠 Xin ý kiến Gemini (Auto-Label)", type="secondary", use_container_width=True):
                        if not api_key: st.error("Thiếu Key")
                        else:
                            # TỰ LƯU THÔNG TIN LÂM SÀNG TRƯỚC KHI HỎI
                            save_log({"id": selected_id, "clinical_context": ctx, "expert_note": note, "prompt_guidance": guide, "technical_tags": "; ".join(tags)})
                            
                            with st.spinner("Gemini đang phân tích..."):
                                res = ask_gemini(api_key, pil_img, ctx, note, guide, tags)
                                txt = res.get("reasoning", "")
                                if txt:
                                    if "KEY" in txt: st.error(txt)
                                    else:
                                        vn_time = get_vn_time()
                                        hist_data.insert(0, {"time": vn_time, "prompt": res.get("sent_prompt"), "response": txt, "model": res.get("used_model")})
                                        save_log({"id": selected_id, "ai_reasoning": json.dumps(hist_data)})
                                        st.rerun()
                                else: st.error(f"Lỗi: {res}")

                    # 4. KẾT QUẢ & DEBUG & NHẬT KÝ
                    if hist_data:
                        last_item = hist_data[0]
                        st.markdown(f"""
                        <div class="gemini-full-box">
                            <strong>🤖 KẾT QUẢ MỚI NHẤT ({last_item.get('model')}) - {last_item.get('time')}</strong><br>
                            <hr style="margin:5px 0; border-color:#c8e6c9">
                            {last_item.get('response', '').replace("\n", "<br>")}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("🔌 Debug: Xem nội dung Prompt đã gửi đi"):
                            st.code(last_item.get('prompt', ''), language="text")
                        
                        if len(hist_data) > 0:
                            st.caption("📜 Nhật ký Hội chẩn (Cũ hơn):")
                            for i, item in enumerate(hist_data):
                                c_txt, c_btn = st.columns([5, 1])
                                with c_txt:
                                    st.markdown(f"""<div class="history-item">🕒 <b>{item.get('time')}</b>: {item.get('response')[:60]}...</div>""", unsafe_allow_html=True)
                                with c_btn:
                                    if st.button("🔍", key=f"v_{i}"): view_log_popup(item)

        else: st.warning("Trống.")

elif mode == "🛠️ Xuất Dataset (Admin)":
    st.title("🛠️ DATASET YOLO")
    pwd = st.text_input("Password:", type="password")
    if pwd and check_password(pwd):
        df = get_logs()
        if not df.empty:
            if st.button("📦 TẢI DATASET (ZIP)"):
                with st.spinner("Zipping..."):
                    buf = BytesIO()
                    with zipfile.ZipFile(buf, "w") as zf:
                        zf.writestr("classes.txt", "\n".join(LABEL_MAPPING.keys()))
                        for i, r in df.iterrows():
                            if r.get('image_url'):
                                try:
                                    zf.writestr(f"images/{r['id']}.jpg", requests.get(r['image_url'], timeout=3).content)
                                    txt = "".join([f"{LABEL_MAPPING[l.strip()]} 0.5 0.5 1.0 1.0\n" for l in str(r.get('label_1') or "").split(";") if l.strip() in LABEL_MAPPING])
                                    zf.writestr(f"labels/{r['id']}.txt", txt)
                                except: pass
                    st.download_button("📥 TẢI", buf.getvalue(), "data.zip", "application/zip")