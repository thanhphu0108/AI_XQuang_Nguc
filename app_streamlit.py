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
import ast # Dùng để convert chuỗi lưu trong DB thành Dict
import google.generativeai as genai
from supabase import create_client, Client

# ================= 1. CẤU HÌNH & CSS XỊN =================
st.set_page_config(page_title="AI Hospital (V30.2 - Cloud Pro)", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main { background-color: #e9ecef; }
    
    /* KHUNG GIẤY A4 */
    .a4-paper {
        background-color: white;
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
        padding: 40px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        font-family: 'Times New Roman', serif;
        color: #333;
    }
    
    .hospital-header {
        text-align: center;
        border-bottom: 3px double #002f6c;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }
    .hospital-header h1 { color: #002f6c; margin: 0; font-size: 24px; text-transform: uppercase; font-weight: 900; }
    
    .info-table { width: 100%; margin-bottom: 20px; border-collapse: collapse; font-size: 15px; }
    .info-table td { padding: 8px 5px; border-bottom: 1px solid #eee; vertical-align: top; }
    
    .section-header {
        background-color: #e3f2fd;
        padding: 8px 15px;
        font-weight: bold;
        font-size: 16px;
        color: #002f6c;
        margin-top: 25px;
        margin-bottom: 15px;
        border-left: 5px solid #002f6c;
        text-transform: uppercase;
    }
    
    .conclusion-box {
        padding: 20px;
        border: 2px solid #ccc;
        margin-bottom: 20px;
        background-color: #fafafa;
        text-align: center;
    }
    
    .gemini-block {
        background-color: #fff8e1;
        border: 1px dashed #ff8f00;
        padding: 15px;
        margin-top: 15px;
        border-radius: 5px;
        text-align: left;
        font-size: 14px;
    }
    
    .stButton>button { width: 100%; font-weight: bold; height: 45px; }
    
    /* Box cho phần Hội chẩn */
    .workstation-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
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

# --- SUPABASE UTILS ---
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

# --- HTML REPORT GENERATOR (A4 STYLE) ---
def generate_html_report(findings_db, has_danger, patient_info, img_id, gemini_text=""):
    current_time = datetime.now().strftime('%H:%M ngày %d/%m/%Y')
    
    # Xử lý input findings_db (vì từ DB nó là chuỗi string)
    if isinstance(findings_db, str):
        try: findings_db = ast.literal_eval(findings_db)
        except: findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    
    lung_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Hai trường phổi sáng đều.</li><li>Không ghi nhận đám mờ khu trú hay lan tỏa.</li></ul>"""
    if findings_db.get("Lung"): lung_html = f'<ul style="margin-top:0px; padding-left:20px; color:#c62828;"><li><b>Ghi nhận bất thường:</b> {"; ".join(findings_db["Lung"])}</li></ul>'
    
    pleura_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Góc sườn hoành hai bên nhọn, sáng.</li><li>Không thấy hình ảnh tràn dịch, tràn khí màng phổi.</li></ul>"""
    if findings_db.get("Pleura"): pleura_html = f'<ul style="margin-top:0px; padding-left:20px; color:#c62828;"><li><b>Phát hiện bất thường:</b> {"; ".join(findings_db["Pleura"])}</li></ul>'
    
    heart_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Bóng tim không to, chỉ số tim/lồng ngực (CTR) < 0,5.</li><li>Trung thất cân đối.</li></ul>"""
    if findings_db.get("Heart"): heart_html = f'<ul style="margin-top:0px; padding-left:20px; color:#e65100;"><li><b>Tim mạch:</b> {"; ".join(findings_db["Heart"])}</li></ul>'
    
    bone_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Khung xương lồng ngực cân đối, không thấy đường gãy xương sườn/xương đòn.</li></ul>"""
    
    # Logic kết luận
    is_abnormal = has_danger or (len(findings_db.get("Lung", [])) + len(findings_db.get("Pleura", [])) > 0) or ("BẤT THƯỜNG" in str(has_danger)) # Fix logic
    
    if is_abnormal:
        conclusion_html = """<div style='color:#c62828; font-weight:900; font-size:20px; margin-bottom:5px; text-transform: uppercase;'>🔴 KẾT LUẬN: CÓ HÌNH ẢNH BẤT THƯỜNG</div>"""
    else:
        conclusion_html = """<div style='color:#2e7d32; font-weight:900; font-size:20px; margin-bottom:5px; text-transform: uppercase;'>✅ KẾT LUẬN: CHƯA GHI NHẬN BẤT THƯỜNG</div>"""

    gemini_block = ""
    if gemini_text:
        gemini_block = f"""<div class="gemini-block"><b>🤖 Ý kiến tham khảo từ Gemini:</b><br><i>"{gemini_text}"</i></div>"""

    html = f"""
    <div class="a4-paper">
        <div class="hospital-header">
            <h1>PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH</h1>
            <p>Hệ thống AI Hỗ trợ Chẩn đoán X-quang Ngực</p>
        </div>
        
        <table class="info-table">
            <tr>
                <td style="width:60%;"><strong>Họ và tên:</strong> {patient_info}</td>
                <td style="text-align:right;"><strong>Mã hồ sơ:</strong> {img_id}</td>
            </tr>
            <tr>
                <td><strong>Chỉ định:</strong> Chụp X-quang ngực thẳng (PA)</td>
                <td style="text-align:right;"><strong>Thời gian:</strong> {current_time}</td>
            </tr>
        </table>

        <div class="section-header">I. MÔ TẢ HÌNH ẢNH (AI FINDINGS)</div>
        <p><strong>1. Nhu mô phổi</strong></p>{lung_html}
        <p><strong>2. Màng phổi</strong></p>{pleura_html}
        <p><strong>3. Tim – Trung thất</strong></p>{heart_html}
        <p><strong>4. Hệ xương thành ngực</strong></p>{bone_html}

        <div class="section-header">II. KẾT LUẬN & KHUYẾN NGHỊ</div>
        <div class="conclusion-box">
            {conclusion_html}
            {gemini_block}
        </div>
        
        <div class="footer">
            <div style="text-align:center; font-style:italic; font-size:12px; margin-top:30px; border-top:1px solid #ddd; padding-top:10px;">
                * Kết quả này do AI hỗ trợ phân tích, vui lòng kết hợp lâm sàng và ý kiến bác sĩ chuyên khoa.
            </div>
        </div>
    </div>
    """
    return html

# --- PROCESS & SAVE ---
def process_and_save(image_file):
    start_t = time.time()
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

    # Resize
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
                            color = (255, 0, 0) if pct > 75 else (0, 165, 255)
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_img, spec[:4], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except: pass
    else: findings_db["Lung"].append("Chế độ Test (No Model)")

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
                        t1, t2 = st.tabs(["🖼️ Hình ảnh AI", "📄 Phiếu Kết Quả (A4)"])
                        
                        with t1:
                            st.image(img_out, caption=f"ID: {img_id}", use_container_width=True)
                        
                        with t2:
                            gemini_txt = ""
                            if api_key:
                                res = ask_gemini(api_key, pil_img)
                                gemini_txt = res.get("reasoning", "")
                                if gemini_txt:
                                    save_log({"id": img_id, "ai_reasoning": gemini_txt})

                            # RENDER A4
                            report_html = generate_html_report(findings, danger, "Nguyễn Văn A (Demo)", img_id, gemini_txt)
                            st.markdown(report_html, unsafe_allow_html=True)
                            
                        st.success("✅ Đã xử lý xong!")
                    else:
                        st.error("Lỗi xử lý ảnh")

elif mode == "📂 Hội Chẩn (Cloud)":
    st.title("📂 HỘI CHẨN & HỒ SƠ BỆNH ÁN")
    df = get_logs()
    
    if not df.empty:
        df = df.fillna("")
        id_list = df['id'].unique()
        selected_id = st.selectbox("👉 Chọn Mã Hồ Sơ:", id_list)
        
        if selected_id:
            record = df[df["id"] == selected_id].iloc[0]
            
            # --- CHIA 2 TAB: 1 LÀM VIỆC, 1 XEM HỒ SƠ A4 ---
            tab_work, tab_report = st.tabs(["👨‍⚕️ Bàn Làm Việc", "📄 Hồ Sơ Bệnh Án (A4)"])
            
            # === TAB 1: NHẬP LIỆU & AI ===
            with tab_work:
                c1, c2 = st.columns([1, 1])
                with c1:
                    if record.get('image_url'):
                        st.image(record['image_url'], caption="Ảnh X-quang đã lưu", use_container_width=True)
                        # Load ảnh để hỏi lại Gemini nếu cần
                        try:
                            import requests
                            from io import BytesIO
                            response = requests.get(record['image_url'])
                            pil_img = Image.open(BytesIO(response.content))
                        except: pil_img = None
                
                with c2:
                    st.markdown('<div class="workstation-box">', unsafe_allow_html=True)
                    st.info(f"**Bệnh nhân:** {record.get('patient_info')}")
                    st.success(f"**Kết quả YOLO:** {record.get('result')}")
                    
                    if record.get('ai_reasoning'):
                        with st.expander("🤖 Đọc kết quả Gemini cũ"):
                            st.write(record.get('ai_reasoning'))
                    
                    st.markdown("---")
                    st.markdown("#### 📝 Nhập liệu lâm sàng")
                    ctx = st.text_area("Bệnh cảnh:", value=record.get("clinical_context") or "")
                    note = st.text_area("Ý kiến chuyên gia:", value=record.get("expert_note") or "")
                    guide = st.text_area("Hướng dẫn AI (Prompt):", value=record.get("prompt_guidance") or "")
                    
                    tags_str = record.get("technical_tags") or ""
                    def_tags = [t.strip() for t in tags_str.split(";")] if tags_str else []
                    tags = st.multiselect("Đánh giá Kỹ thuật:", TECHNICAL_OPTS, default=def_tags)
                    
                    c_btn1, c_btn2 = st.columns(2)
                    with c_btn1:
                        if st.button("💾 Cập nhật Dữ liệu"):
                            save_log({
                                "id": selected_id, "clinical_context": ctx, 
                                "expert_note": note, "prompt_guidance": guide, 
                                "technical_tags": "; ".join(tags)
                            })
                            st.toast("Đã lưu!")
                            
                    with c_btn2:
                        if api_key and pil_img and st.button("🧠 Hỏi lại Gemini"):
                            with st.spinner("Gemini đang đọc lại..."):
                                res = ask_gemini(api_key, pil_img, ctx, note, guide, tags)
                                txt = res.get("reasoning", "")
                                if txt:
                                    save_log({"id": selected_id, "ai_reasoning": txt})
                                    st.success("Đã cập nhật ý kiến Gemini mới!")
                                    st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            # === TAB 2: XEM LẠI PHIẾU A4 ===
            with tab_report:
                # Lấy dữ liệu từ record để tái tạo phiếu
                raw_findings = record.get("details", "")
                is_danger = record.get("result") == "BẤT THƯỜNG"
                p_info = record.get("patient_info", "N/A")
                gemini_old = record.get("ai_reasoning", "")
                
                # Gọi hàm tạo HTML
                html_repro = generate_html_report(raw_findings, is_danger, p_info, selected_id, gemini_old)
                st.markdown(html_repro, unsafe_allow_html=True)

    else:
        st.info("Chưa có dữ liệu nào trên Cloud.")

elif mode == "🛠️ Xuất Dataset":
    st.title("🛠️ DATASET")
    if st.button("Tải CSV"):
        df = get_logs()
        if not df.empty:
            st.download_button("Download", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")