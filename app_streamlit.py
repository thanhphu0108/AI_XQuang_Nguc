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

# ================= 1. CẤU HÌNH & CSS (FIX ID & ZIP) =================
st.set_page_config(page_title="Hệ thống AI hỗ trợ phân tích X-quang ngực", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
    
    /* 1. KHUNG LABELING */
    .labeling-box {
        background-color: #fff3e0; border: 2px solid #ff9800; border-radius: 8px;
        padding: 15px; margin-top: 20px; margin-bottom: 10px;
    }
    .labeling-header {
        font-weight: bold; color: #e65100; border-bottom: 1px dashed #ff9800; 
        margin-bottom: 10px; font-size: 14px; text-transform: uppercase;
    }
    
    /* 2. GEMINI BOX */
    .gemini-full-box {
        background-color: #e8f5e9; border: 1px solid #4caf50; border-radius: 8px;
        padding: 15px; margin-top: 15px; font-family: 'Segoe UI'; color: #1b5e20; font-size: 14px; line-height: 1.5;
    }
    
    /* 3. HISTORY */
    .history-item {
        border-left: 4px solid #9e9e9e; padding-left: 10px; margin-bottom: 8px; 
        font-size: 12px; color: #444; background: white; padding: 8px; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* 4. A4 PAPER */
    .a4-paper {
        background-color: white !important; padding: 40px; border: 1px solid #ccc;
        box-shadow: 0 0 15px rgba(0,0,0,0.1); font-family: 'Times New Roman', serif; color: #000; margin-top: 10px; min-height: 600px;
    }
    .rp-header { text-align: center; border-bottom: 2px solid #002f6c; padding-bottom: 15px; margin-bottom: 20px; }
    .rp-title { font-size: 22px; font-weight: bold; color: #002f6c; text-transform: uppercase; margin: 0; }
    .rp-section { 
        background-color: #f0f2f5; border-left: 5px solid #002f6c; padding: 8px; 
        font-weight: bold; font-size: 14px; text-transform: uppercase; margin-top: 20px; margin-bottom: 10px; 
    }
    
    /* 5. COMMON */
    div[data-testid="stRadio"] { margin-top: -5px !important; }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 6px; height: 45px; }
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

# --- UTILS (ĐÃ FIX UTC+7 CHO ID) ---
def get_vn_now(): 
    return datetime.utcnow() + timedelta(hours=7)

def get_vn_time_str(): 
    return get_vn_now().strftime("%H:%M %d/%m/%Y")

def get_id_vn():
    # ID theo giờ Việt Nam: DDMMYYYYHHMMSS
    return get_vn_now().strftime("%d%m%Y%H%M%S")

def check_password(password): return password == "Admin@123p"

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
DOCTOR_ROSTER = { "ANATOMY": "Dr_Anatomy.pt", "PNEUMOTHORAX": "Dr_Pneumothorax.pt", "PNEUMONIA": "Dr_Pneumonia.pt", "TUMOR": "Dr_Tumor.pt", "EFFUSION": "Dr_Effusion.pt", "OPACITY": "Dr_Opacity.pt", "HEART": "Dr_Heart.pt" }

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

# --- HELPERS ---
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

# Cache dữ liệu 5 giây để tránh reload làm mất check của Admin
@st.cache_data(ttl=5) 
def get_logs():
    if not supabase: return pd.DataFrame()
    try: return pd.DataFrame(supabase.table("logs").select("*").order("created_at", desc=True).execute().data)
    except: return pd.DataFrame()

@st.dialog("📋 CHI TIẾT HỘI CHẨN (FULL SCREEN)", width="large")
def view_log_popup(item):
    st.markdown(f"**Thời gian:** {item.get('time')} | **Model:** {item.get('model')}")
    st.markdown("### 🤖 KẾT LUẬN CHI TIẾT")
    st.markdown(f"""<div class="popup-result-box">{item.get('response', '').replace("\n", "<br>")}</div>""", unsafe_allow_html=True)
    with st.expander("🔌 Xem Prompt"): st.code(item.get('prompt', ''), language="text")

# --- GEMINI (AUTO DETECT) ---
def ask_gemini(api_key, image, context="", note="", guide="", tags=[]):
    if not api_key: return {"labels": [], "reasoning": "Thiếu API Key", "prompt": ""}
    try:
        genai.configure(api_key=api_key)
        
        # --- AUTO DETECT MODEL ---
        try:
            available = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                    available.append(m.name)
            # Ưu tiên: Flash -> Pro -> 1.5 -> Khác
            available.sort(key=lambda x: 0 if '1.5-flash' in x else 1 if '1.5-pro' in x else 2 if 'flash' in x else 3 if 'pro' in x else 4)
            model_priority = available if available else ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
        except:
            model_priority = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]

        labels_str = ", ".join(STRUCTURED_LABELS) 
        tech_note = ", ".join(tags) if tags else "Phim đạt chuẩn kỹ thuật."
        prompt = f"""
Role: Senior Radiologist.
Inputs: Context="{context}", Note="{note}", Guide="{guide}", Technical="{tech_note}".
Task: Analyze Chest X-ray. Select from: [{labels_str}].
Output JSON: {{ "labels": ["..."], "reasoning": "Structure: Technique, Description (Lungs, Heart, Pleura, Bones), Discussion, Conclusion." }}
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
                if "429" in err_msg: time.sleep(1); continue
                elif "API_KEY" in err_msg: return {"labels": [], "reasoning": "🔑 KEY HẾT HẠN HOẶC SAI!", "prompt": ""}
                else: last_error = err_msg; continue
        
        return {"labels": [], "reasoning": f"⚠️ Lỗi kết nối (RAW): {last_error}", "sent_prompt": prompt}
    except Exception as e: return {"labels": [], "reasoning": f"CRASH: {str(e)}", "sent_prompt": ""}

# --- HTML REPORT ---
def generate_html_report(findings_db, has_danger, patient_info, img_id):
    current_time = get_vn_time_str()
    def mk_list(items, default):
        if not items: return f"<li>{default}</li>"
        return "".join([f"<li style='color:#c62828'><b>PHÁT HIỆN:</b> {i}</li>" for i in items])

    lung = mk_list(findings_db.get("Lung", []), "Hai trường phổi sáng đều.")
    heart = mk_list(findings_db.get("Heart", []), "Bóng tim không to.")
    pleura = mk_list(findings_db.get("Pleura", []), "Góc sườn hoành nhọn.")
    concl = "<div style='color:#c62828; font-weight:bold; border:2px solid #c62828; padding:10px; border-radius:5px;'>⚠️ CÓ HÌNH ẢNH BẤT THƯỜNG</div>" if has_danger else "<div style='color:#2e7d32; font-weight:bold; border:2px solid #2e7d32; padding:10px; border-radius:5px;'>✅ HÌNH ẢNH BÌNH THƯỜNG</div>"

    return f"""
    <div class="a4-paper">
        <div class="rp-header">
            <h2 class="rp-title">PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH</h2>
            <div class="rp-sub">(Hệ thống AI hỗ trợ phân tích X-quang ngực)</div>
        </div>
        <table style="width:100%; border-bottom:1px solid #ccc; margin-bottom:15px;">
            <tr><td style="padding:5px;"><b>Họ tên:</b> {patient_info}</td><td style="text-align:right;"><b>Ngày:</b> {current_time}</td></tr>
            <tr><td style="padding:5px;"><b>Mã HS:</b> {img_id}</td><td style="text-align:right;"><b>Chỉ định:</b> X-quang Ngực</td></tr>
        </table>
        <div class="rp-section">I. MÔ TẢ HÌNH ẢNH (AI SCAN)</div>
        <ul style="line-height:1.6;">
            <li><b>🫁 Phổi:</b> <ul>{lung}</ul></li>
            <li><b>❤️ Tim:</b> <ul>{heart}</ul></li>
            <li><b>🛡️ Màng phổi:</b> <ul>{pleura}</ul></li>
        </ul>
        <div class="rp-section">II. KẾT LUẬN</div>
        <div style="text-align:center; margin-top:15px;">{concl}</div>
        
    </div>
    """

# --- PROCESS IMAGE ---
def process_and_save(image_file):
    filename = image_file.name.lower()
    img_rgb, patient_info = None, "Dem"
    image_file.seek(0)
    try:
        if filename.endswith(('.dcm', '.dicom')):
            ds = pydicom.dcmread(image_file)
            patient_info = str(ds.get("PatientName", "Dem")).replace('^', ' ').strip()
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

    # --- FIX: DÙNG HÀM ID VN ---
    img_id = get_id_vn()
    
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
    st.title("🏥 Hệ thống AI hỗ trợ phân tích X-quang ngực")
    uploaded_file = st.file_uploader("Chọn ảnh X-quang:", type=["jpg", "png", "jpeg", "dcm"])
    if uploaded_file and st.button("🚀 PHÂN TÍCH"):
        with st.spinner("Đang chạy AI Nội bộ..."):
            img_out, findings, danger, img_id, pil_img = process_and_save(uploaded_file)
            if img_out is not None:
                c1, c2 = st.columns(2)
                with c1: st.image(img_out, caption=f"ID: {img_id}", use_container_width=True)
                with c2: st.markdown(generate_html_report(findings, danger, "Nguyễn Văn A", img_id), unsafe_allow_html=True)
                st.success("✅ Đã lưu kết quả!")
            else: st.error("Lỗi file.")

elif mode == "📂 AI Gemini + Dán nhãn":
    if not supabase: st.error("⛔ Chưa kết nối Cloud.")
    else:
        df = get_logs()
        if not df.empty:
            df = df.fillna("")
            id_list = df['id'].tolist()
            c_sel, _ = st.columns([1, 2])
            with c_sel: selected_id = st.selectbox("👉 Chọn Mã Hồ Sơ:", id_list)
            
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
                
                col_left, col_right = st.columns([1, 1.2])
                with col_left:
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    if record.get('image_url'): st.image(record['image_url'], use_container_width=True)
                    st.caption(f"YOLO: {record.get('result')} | BN: {record.get('patient_info')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    if len(hist_data) > 0:
                        st.markdown('<div class="labeling-header">📜 NHẬT KÝ HỘI CHẨN</div>', unsafe_allow_html=True)
                        st.markdown('<div class="history-container">', unsafe_allow_html=True)
                        for i, item in enumerate(hist_data):
                            c_txt, c_btn = st.columns([5, 1])
                            with c_txt: st.markdown(f"""<div class="history-item">🕒 <b>{item.get('time')}</b>: {item.get('response')[:60]}...</div>""", unsafe_allow_html=True)
                            with c_btn: 
                                if st.button("🔍", key=f"v_{i}"): view_log_popup(item)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else: st.info("Chưa có lịch sử.")

                with col_right:
                    st.markdown('<div class="labeling-header">1. DỮ LIỆU ĐẦU VÀO</div>', unsafe_allow_html=True)
                    tags = st.multiselect("⚙️ Điều kiện kỹ thuật (QA/QC):", TECHNICAL_OPTS, default=[t.strip() for t in (record.get("technical_tags") or "").split(";") if t])
                    ctx = st.text_area("🤒 Bệnh cảnh (Context):", value=record.get("clinical_context") or "", height=80)
                    note = st.text_area("👨‍⚕️ Ý kiến chuyên gia:", value=record.get("expert_note") or "", height=60)
                    guide = st.text_area("📝 Dẫn dắt/Yêu cầu(Prompt):", value=record.get("prompt_guidance") or "", height=60)
                    
                    st.markdown("---")
                    if st.button("🧠 AI Gemini gợi ý", type="secondary", use_container_width=True):
                        if not api_key: st.error("Thiếu Key")
                        else:
                            save_log({"id": selected_id, "clinical_context": ctx, "expert_note": note, "prompt_guidance": guide, "technical_tags": "; ".join(tags)})
                            with st.spinner("Gemini đang phân tích..."):
                                res = ask_gemini(api_key, pil_img, ctx, note, guide, tags)
                                txt = res.get("reasoning", "")
                                if txt:
                                    if "KEY" in txt: st.error(txt)
                                    else:
                                        hist_data.insert(0, {"time": get_vn_time_str(), "prompt": res.get("sent_prompt"), "response": txt, "model": res.get("used_model")})
                                        save_log({"id": selected_id, "ai_reasoning": json.dumps(hist_data)})
                                        st.rerun()
                                else: st.error(f"Lỗi: {res}")

                    if hist_data:
                        last_item = hist_data[0]
                        model_name = last_item.get('model', 'N/A')
                        st.markdown(f"""<div class="gemini-full-box"><strong>🤖 KẾT QUẢ MỚI NHẤT ({model_name})</strong><br><hr style="margin:5px 0">{last_item.get('response', '').replace("\n", "<br>")}</div>""", unsafe_allow_html=True)
                        with st.expander("🔌 Debug: Xem Prompt"): st.code(last_item.get('prompt', ''), language="text")

                    st.markdown('<div class="labeling-box">', unsafe_allow_html=True)
                    st.markdown('<div class="labeling-header">🏷️ KẾT LUẬN & GÁN NHÃN</div>', unsafe_allow_html=True)
                    saved_lbls = [l.strip() for l in (record.get("label_1") or "").split(";") if l]
                    if not saved_lbls and hist_data:
                        last_resp = hist_data[0].get("response", "")
                        for sl in STRUCTURED_LABELS:
                            if sl.split("(")[0].split("/")[-1].strip().lower() in last_resp.lower(): saved_lbls.append(sl)
                    c1, c2 = st.columns([1.5, 1])
                    with c1: new_fb = st.radio("Đánh giá AI:", FEEDBACK_OPTS, index=0, label_visibility="collapsed")
                    with c2: rating = st.select_slider("Rating:", options=RATING_OPTS, value="Khá", label_visibility="collapsed")
                    new_lbls = st.multiselect("Chốt bệnh:", STRUCTURED_LABELS, default=[l for l in saved_lbls if l in STRUCTURED_LABELS], label_visibility="collapsed")
                    st.markdown("---")
                    if st.button("💾 LƯU KẾT QUẢ (SAVE)", type="primary", use_container_width=True):
                        save_log({"id": selected_id, "feedback_1": new_fb, "label_1": "; ".join(new_lbls), "prompt_rating": rating, "clinical_context": ctx, "expert_note": note, "prompt_guidance": guide, "technical_tags": "; ".join(tags)})
                        st.success("✅ Đã lưu!")
                    st.markdown('</div>', unsafe_allow_html=True)
        else: st.warning("Trống.")

elif mode == "🛠️ Xuất Dataset (Admin)":
    st.title("🛠️ XUẤT DATASET YOLO (Chọn lọc)")
    pwd = st.text_input("Password:", type="password")
    if pwd and check_password(pwd):
        df = get_logs() # Data is cached now
        if not df.empty:
            st.markdown("### 📋 Chọn hồ sơ muốn xuất:")
            if "Select" not in df.columns: df.insert(0, "Select", False)
            
            # EDITOR (Giữ key để không mất state)
            edited_df = st.data_editor(df, column_config={"Select": st.column_config.CheckboxColumn("Chọn", default=False), "image_url": st.column_config.ImageColumn("Ảnh")}, disabled=df.columns.drop("Select"), hide_index=True, use_container_width=True, key="admin_editor")
            
            # --- LOGIC QUAN TRỌNG: LẤY ĐÚNG DÒNG ĐÃ CHECK ---
            selected_rows = edited_df[edited_df["Select"] == True]
            st.info(f"👉 Đang chọn: {len(selected_rows)} hồ sơ.")
            
            if 'zip_btn' not in st.session_state: st.session_state.zip_btn = None
            
            if st.button(f"🚀 ĐÓNG GÓI {len(selected_rows)} HỒ SƠ"):
                if len(selected_rows) == 0: st.warning("Chọn ít nhất 1 dòng!")
                else:
                    with st.spinner("Đang xử lý..."):
                        buf = BytesIO()
                        with zipfile.ZipFile(buf, "w") as zf:
                            zf.writestr("classes.txt", "\n".join(LABEL_MAPPING.keys()))
                            for i, r in selected_rows.iterrows():
                                if r.get('image_url'):
                                    try:
                                        zf.writestr(f"images/{r['id']}.jpg", requests.get(r['image_url'], timeout=3).content)
                                        txt = "".join([f"{LABEL_MAPPING[l.strip()]} 0.5 0.5 1.0 1.0\n" for l in str(r.get('label_1') or "").split(";") if l.strip() in LABEL_MAPPING])
                                        zf.writestr(f"labels/{r['id']}.txt", txt)
                                    except: pass
                        st.session_state.zip_btn = buf.getvalue()
                        st.success("Xong! Bấm nút dưới để tải.")
                        # Không rerun ở đây để tránh mất tích xanh
            
            if st.session_state.zip_btn:
                st.download_button("📥 TẢI DATA.ZIP", st.session_state.zip_btn, "data.zip", "application/zip", type="primary")
        else: st.info("Trống.")
    elif pwd: st.error("Sai mật khẩu!")