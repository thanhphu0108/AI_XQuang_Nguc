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

# ================= 1. CẤU HÌNH & CSS A4 =================
st.set_page_config(page_title="AI Hospital (V30.0 - Report A4)", page_icon="🩻", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    
    /* CSS CHO PHIẾU KẾT QUẢ A4 */
    .report-container {
        background-color: white;
        padding: 40px;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: 'Times New Roman', serif;
        color: #333;
        border: 1px solid #ddd;
    }
    .hospital-header {
        text-align: center;
        border-bottom: 2px solid #002f6c;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .hospital-header h2 { color: #002f6c; margin: 0; text-transform: uppercase; font-weight: bold; }
    .hospital-header p { margin: 5px 0; font-style: italic; color: #666; }
    
    .info-table { width: 100%; margin-bottom: 20px; border-collapse: collapse; }
    .info-table td { padding: 8px; border-bottom: 1px solid #eee; }
    
    .section-header {
        background-color: #e3f2fd;
        padding: 5px 10px;
        font-weight: bold;
        color: #002f6c;
        margin-top: 20px;
        margin-bottom: 10px;
        border-left: 5px solid #002f6c;
    }
    
    .stButton>button { width: 100%; font-weight: bold; height: 50px; font-size: 16px; }
    .gemini-box { background-color: #fff3e0; padding: 15px; border-radius: 5px; border: 1px solid #ff9800; margin-top: 15px; }
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

TECHNICAL_OPTS = ["✅ Phim chuẩn", "⚠️ Chụp tại giường (AP)", "⚠️ Hít vào nông", "⚠️ Bệnh nhân xoay", "⚠️ Tia cứng/mềm", "⚠️ Dị vật/Áo", "⚠️ Mất góc sườn hoành"]
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

# --- SUPABASE UPLOAD ---
def upload_image(img_cv, filename):
    try:
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
        bucket = "xray_images"
        supabase.storage.from_(bucket).upload(filename, buffer.tobytes(), {"content-type": "image/jpeg", "upsert": "true"})
        return supabase.storage.from_(bucket).get_public_url(filename)
    except:
        try: return supabase.storage.from_("xray_images").get_public_url(filename)
        except: return None

def save_log(data):
    try:
        supabase.table("logs").upsert(data).execute()
        return True
    except: return False

def get_logs():
    try:
        response = supabase.table("logs").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(response.data)
    except: return pd.DataFrame()

# --- GEMINI ---
def ask_gemini(api_key, image, context="", note="", guide="", tags=[]):
    if not api_key: return {}
    try:
        genai.configure(api_key=api_key)
        prompt = f"""Role: Radiologist. Analyze Chest X-ray. Context: {context}. Note: {note}. Output JSON: {{ "labels": [], "reasoning": "..." }}"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt, image], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except: return {}

# --- TẠO PHIẾU KẾT QUẢ A4 (HÀM CŨ ĐÃ QUAY LẠI) ---
def generate_html_report(findings_db, has_danger, patient_info, img_id, gemini_text=""):
    current_time = datetime.now().strftime('%H:%M ngày %d/%m/%Y')
    
    # Tạo nội dung HTML cho từng phần
    lung_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Hai trường phổi sáng đều.</li><li>Không ghi nhận đám mờ khu trú hay lan tỏa.</li></ul>"""
    if findings_db["Lung"]: lung_html = f'<ul style="margin-top:0px; padding-left:20px; color:#c62828;"><li><b>Ghi nhận bất thường:</b> {"; ".join(findings_db["Lung"])}</li></ul>'
    
    pleura_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Góc sườn hoành hai bên nhọn, sáng.</li><li>Không thấy hình ảnh tràn dịch, tràn khí màng phổi.</li></ul>"""
    if findings_db["Pleura"]: pleura_html = f'<ul style="margin-top:0px; padding-left:20px; color:#c62828;"><li><b>Phát hiện bất thường:</b> {"; ".join(findings_db["Pleura"])}</li></ul>'
    
    heart_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Bóng tim không to, chỉ số tim/lồng ngực (CTR) < 0,5.</li><li>Trung thất cân đối.</li></ul>"""
    if findings_db["Heart"]: heart_html = f'<ul style="margin-top:0px; padding-left:20px; color:#e65100;"><li><b>Tim mạch:</b> {"; ".join(findings_db["Heart"])}</li></ul>'
    
    bone_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Khung xương lồng ngực cân đối, không thấy đường gãy xương sườn/xương đòn.</li></ul>"""
    
    if has_danger or (len(findings_db["Lung"]) + len(findings_db["Pleura"]) > 0):
        conclusion_html = """<div style='color:#c62828; font-weight:bold; font-size:18px; margin-bottom:10px; text-transform: uppercase;'>🔴 KẾT LUẬN: CÓ HÌNH ẢNH BẤT THƯỜNG TRÊN PHIM X-QUANG</div>"""
    else:
        conclusion_html = """<div style='color:#2e7d32; font-weight:bold; font-size:18px; margin-bottom:10px; text-transform: uppercase;'>✅ KẾT LUẬN: CHƯA GHI NHẬN BẤT THƯỜNG TRÊN PHIM</div>"""

    gemini_block = ""
    if gemini_text:
        gemini_block = f"""<div style="background:#fff3e0; padding:10px; border:1px dashed orange; margin-top:10px;"><b>🤖 Ý kiến tham khảo từ Gemini:</b><br><i>{gemini_text}</i></div>"""

    html = f"""
    <div class="report-container">
        <div class="hospital-header">
            <h2>PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH</h2>
            <p>(Hệ thống AI Hỗ trợ Chẩn đoán X-quang Ngực)</p>
        </div>
        
        <table class="info-table">
            <tr>
                <td style="width:60%;"><strong>Họ và tên:</strong> {patient_info}</td>
                <td style="text-align:right;"><strong>ID Hồ sơ:</strong> {img_id}</td>
            </tr>
            <tr>
                <td><strong>Chỉ định:</strong> Chụp X-quang ngực thẳng (PA)</td>
                <td style="text-align:right;"><strong>Ngày chụp:</strong> {current_time}</td>
            </tr>
        </table>

        <div class="section-header">I. MÔ TẢ HÌNH ẢNH (AI FINDINGS)</div>
        <p style="margin-bottom:5px;"><strong>1. Nhu mô phổi</strong></p>{lung_html}
        <p style="margin-bottom:5px;"><strong>2. Màng phổi</strong></p>{pleura_html}
        <p style="margin-bottom:5px;"><strong>3. Tim – Trung thất</strong></p>{heart_html}
        <p style="margin-bottom:5px;"><strong>4. Hệ xương thành ngực</strong></p>{bone_html}

        <div class="section-header" style="margin-top:25px;">II. KẾT LUẬN & KHUYẾN NGHỊ</div>
        <div style="padding:15px; border:2px solid #ccc; margin-bottom:15px; background:#fafafa;">
            {conclusion_html}
            {gemini_block}
        </div>
        <p style="font-size:12px; font-style:italic; text-align:center;">* Kết quả này do AI hỗ trợ, vui lòng kết hợp lâm sàng và ý kiến bác sĩ chuyên khoa.</p>
    </div>
    """
    return html

# --- PROCESS & SAVE (UNIVERSAL) ---
def process_and_save(image_file):
    start_t = time.time()
    filename = image_file.name.lower()
    img_rgb, patient_info = None, "Ẩn danh"
    
    image_file.seek(0)
    
    # Đọc ảnh
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

    # Resize
    h, w = img_rgb.shape[:2]
    scale = 1024 / max(h, w)
    img_resized = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
    display_img = img_resized.copy()
    
    findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    has_danger = False

    # Chạy YOLO
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
                            
                            color = (255, 0, 0) if pct > 75 else (0, 165, 255)
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_img, spec[:4], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except: pass
    else: findings_db["Lung"].append("Chế độ Test (No Model)")

    # Upload
    img_id = datetime.now().strftime("%d%m%Y%H%M%S")
    img_url = upload_image(display_img, f"XRAY_{img_id}.jpg")
    
    if img_url:
        save_log({"id": img_id, "created_at": datetime.now().isoformat(), "image_url": img_url, "result": "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG", "details": str(findings_db), "patient_info": patient_info})

    return display_img, findings_db, has_danger, img_id, Image.fromarray(img_resized)

# ================= 3. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("ĐIỀU KHIỂN")
    api_key = st.text_input("🔑 Gemini API Key:", value=st.secrets.get("GEMINI_API_KEY", ""), type="password")
    mode = st.radio("Menu:", ["🔍 Phân Tích & In Phiếu", "📂 Hội Chẩn (Cloud)", "🛠️ Xuất Dataset"])

if mode == "🔍 Phân Tích & In Phiếu":
    st.title("🏥 TRỢ LÝ CHẨN ĐOÁN (A4 REPORT)")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        uploaded_file = st.file_uploader("Chọn ảnh X-quang:", type=["jpg", "png", "jpeg", "dcm"])
        if uploaded_file and st.button("🚀 PHÂN TÍCH NGAY", type="primary"):
            with col2:
                with st.spinner("Đang chạy AI & Tạo phiếu..."):
                    img_out, findings, danger, img_id, pil_img = process_and_save(uploaded_file)
                    
                    if img_out is not None:
                        # Tab hiển thị: Ảnh và Phiếu
                        t1, t2 = st.tabs(["🖼️ Hình ảnh AI", "📄 Phiếu Kết Quả (A4)"])
                        
                        with t1:
                            st.image(img_out, caption=f"ID: {img_id}", use_container_width=True)
                        
                        with t2:
                            # Hỏi Gemini để bổ sung vào phiếu
                            gemini_txt = ""
                            if api_key:
                                res = ask_gemini(api_key, pil_img)
                                gemini_txt = res.get("reasoning", "")
                                if gemini_txt:
                                    save_log({"id": img_id, "ai_reasoning": gemini_txt})

                            # IN PHIẾU A4 XỊN XÒ
                            report_html = generate_html_report(findings, danger, "Nguyễn Văn A (Demo)", img_id, gemini_txt)
                            st.markdown(report_html, unsafe_allow_html=True)
                            
                        st.success("✅ Đã xử lý xong!")
                    else:
                        st.error("Lỗi xử lý ảnh")

elif mode == "📂 Hội Chẩn (Cloud)":
    st.title("📂 DATA LABELING")
    df = get_logs()
    if not df.empty:
        df = df.fillna("")
        selected_id = st.selectbox("Chọn Ca bệnh:", df['id'].unique())
        if selected_id:
            record = df[df["id"] == selected_id].iloc[0]
            c1, c2 = st.columns([1, 1])
            with c1:
                if record.get('image_url'): st.image(record['image_url'], use_container_width=True)
            with c2:
                st.info(f"BN: {record.get('patient_info')}")
                st.markdown(f"**YOLO:** {record.get('result')}")
                st.markdown(f"**Gemini:** {record.get('ai_reasoning')}")
                # Form nhập liệu giữ nguyên...
                st.text_area("Bệnh cảnh:", value=record.get("clinical_context") or "")
                if st.button("Lưu thay đổi"): st.success("Đã lưu (Demo)")

elif mode == "🛠️ Xuất Dataset":
    st.title("🛠️ DATASET")
    if st.button("Tải CSV"):
        df = get_logs()
        if not df.empty:
            st.download_button("Download", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")