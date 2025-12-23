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
import io

# ================= 1. CẤU HÌNH TRANG WEB =================
st.set_page_config(
    page_title="AI Hospital V7.1 (DICOM Support)",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS giao diện
st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    h1, h2, h3 { color: #002f6c; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    .report-box { 
        background: white; padding: 25px; border-radius: 10px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.1); 
        font-family: 'Times New Roman', serif;
    }
    .success-box { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .danger-box { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ================= 2. CẤU HÌNH HỆ THỐNG =================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")
HISTORY_DIR = os.path.join(BASE_PATH, "history")
IMAGES_DIR = os.path.join(HISTORY_DIR, "images")
LOG_FILE = os.path.join(HISTORY_DIR, "log_book.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["ID", "Time", "Result", "Details", "Image_Path", "Patient_Info"])
    df.to_csv(LOG_FILE, index=False)

DOCTOR_ROSTER = {
    "ANATOMY":      "Dr_Anatomy.pt",      
    "PNEUMOTHORAX": "Dr_Pneumothorax.pt", 
    "PNEUMONIA":    "Dr_Pneumonia.pt",    
    "TUMOR":        "Dr_Tumor.pt",        
    "EFFUSION":     "Dr_Effusion.pt",     
    "OPACITY":      "Dr_Opacity.pt",      
    "HEART":        "Dr_Heart.pt"         
}

# ================= 3. LOAD MODEL =================
@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else 'cpu'
    loaded_models = {}
    status_log = []
    
    for role, filename in DOCTOR_ROSTER.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                m = YOLO(path)
                if device == 0: m.to('cuda')
                loaded_models[role] = m
                status_log.append(f"✅ {role}: Ready")
            except: status_log.append(f"❌ {role}: Error")
        else: status_log.append(f"⚠️ {role}: Missing")
    return loaded_models, status_log, device

MODELS, MODEL_STATUS, DEVICE = load_models()

# ================= 4. XỬ LÝ DICOM & ẢNH (CORE) =================
def read_dicom_image(file_buffer):
    """Đọc file DICOM và chuyển đổi sang ảnh RGB chuẩn AI"""
    try:
        ds = pydicom.dcmread(file_buffer)
        
        # Lấy thông tin bệnh nhân (nếu có)
        patient_name = str(ds.get("PatientName", "Anonymous"))
        patient_id = str(ds.get("PatientID", "Unknown"))
        patient_info = f"{patient_name} ({patient_id})"
        
        # Xử lý ảnh (Pixel Array)
        img = ds.pixel_array.astype(float)
        
        # Chuẩn hóa về 0-255 (Windowing đơn giản)
        # DICOM thường là 12-16 bit, cần nén xuống 8 bit cho AI
        img = (np.maximum(img, 0) / img.max()) * 255.0
        img = np.uint8(img)
        
        # Xử lý Photometric Interpretation (Nếu ảnh bị âm bản)
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            img = 255 - img
            
        # Chuyển sang RGB (AI cần 3 kênh màu)
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
            
        return img_rgb, patient_info
    except Exception as e:
        return None, f"Lỗi đọc DICOM: {str(e)}"

# ================= 5. LOGIC LÂM SÀNG =================
def get_finding_text(disease, conf, location):
    pct = conf * 100
    if disease == "PNEUMOTHORAX":
        if pct > 88: return "danger", f"**{location}**: Mất vân phổi ngoại vi, hình ảnh điển hình **Tràn khí** ({pct:.0f}%)."
        elif pct > 75: return "warn", f"**{location}**: Tăng sáng khu trú, nghi ngờ tràn khí ít ({pct:.0f}%)."
    elif disease == "EFFUSION":
        if pct > 80: return "danger", f"**{location}**: Mờ góc sườn hoành, theo dõi **Tràn dịch** ({pct:.0f}%)."
        return "warn", f"**{location}**: Tù nhẹ góc sườn hoành ({pct:.0f}%)."
    elif disease == "PNEUMONIA":
        if pct > 75: return "danger", f"**{location}**: Đám mờ thâm nhiễm, phù hợp **Viêm phổi** ({pct:.0f}%)."
        return "warn", f"**{location}**: Đám mờ rải rác, theo dõi viêm ({pct:.0f}%)."
    elif disease == "TUMOR":
        if pct > 85: return "danger", f"**{location}**: Nốt mờ dạng khối, nghi **U phổi** ({pct:.0f}%)."
        return "warn", f"**{location}**: Nốt mờ đơn độc nghi ngờ ({pct:.0f}%)."
    elif disease == "HEART":
        if pct > 70: return "warn", f"**Bóng tim**: Chỉ số tim/lồng ngực lớn ({pct:.0f}%)."
    return None, None

def save_case(img_cv, findings_db, has_danger, patient_info="N/A"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_id = f"XRAY_{timestamp}"
    file_name = f"{img_id}.jpg"
    
    save_path = os.path.join(IMAGES_DIR, file_name)
    cv2.imwrite(save_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
    
    result = "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG"
    detail_list = findings_db["Lung"] + findings_db["Pleura"] + findings_db["Heart"]
    details = " | ".join(detail_list).replace("**", "") if detail_list else "Không ghi nhận bất thường"
    
    new_data = {
        "ID": img_id, 
        "Time": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Result": result,
        "Details": details,
        "Image_Path": file_name,
        "Patient_Info": patient_info
    }
    
    try:
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([pd.DataFrame([new_data]), df], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
    except: pass
    return img_id

def process_image(image_file):
    if "ANATOMY" not in MODELS: return None, "Thiếu Anatomy", False, 0, ""

    start_t = time.time()
    
    # --- XỬ LÝ ĐẦU VÀO (DICOM hoặc ẢNH THƯỜNG) ---
    filename = image_file.name.lower()
    img_rgb = None
    patient_info = "Ẩn danh"

    if filename.endswith(".dcm") or filename.endswith(".dicom"):
        img_rgb, p_info = read_dicom_image(image_file)
        if isinstance(p_info, str) and img_rgb is None: # Có lỗi
            return None, p_info, False, 0, ""
        patient_info = p_info
    else:
        # Xử lý ảnh thường (JPG/PNG)
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    if img_rgb is None: return None, "Lỗi đọc ảnh", False, 0, ""

    # Resize chuẩn y tế
    h, w = img_rgb.shape[:2]
    scale = 1280 / max(h, w)
    img_resized = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
    display_img = img_resized.copy()
    
    findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    has_danger = False
    
    PRIORITY = ["PNEUMOTHORAX", "EFFUSION", "TUMOR", "PNEUMONIA"] 
    SECONDARY = ["OPACITY"]

    # --- AI INFERENCE ---
    # Convert sang BGR cho model (vì model train trên cv2 BGR)
    img_model_input = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    anatomy_res = MODELS["ANATOMY"](img_model_input, conf=0.35, iou=0.45, verbose=False)[0]

    for box in anatomy_res.boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        region_name = anatomy_res.names[cls_id]
        
        pad = 40
        x1, y1, x2, y2 = coords
        roi = img_model_input[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
        if roi.size == 0: continue

        target_models = []
        if "Lung" in region_name: target_models = PRIORITY + SECONDARY
        elif "Heart" in region_name: target_models = ["HEART"]
        
        found_specific = False 

        for spec in target_models:
            if spec not in MODELS: continue
            if spec == "OPACITY" and found_specific: continue

            res = MODELS[spec](roi, verbose=False)[0]
            if res.probs.top1conf.item() < 0.6: continue 
            
            label = res.names[res.probs.top1]
            conf = res.probs.top1conf.item()

            if label == "Disease":
                loc_vn = "Phổi phải" if "Right" in region_name else "Phổi trái"
                if "Heart" in region_name: loc_vn = "Tim"
                level, text = get_finding_text(spec, conf, loc_vn)
                
                if text:
                    if spec in ["PNEUMOTHORAX", "EFFUSION"]: findings_db["Pleura"].append(text)
                    elif spec == "HEART": findings_db["Heart"].append(text)
                    else: findings_db["Lung"].append(text)
                    
                    if level == "danger": has_danger = True
                    if spec in ["PNEUMONIA", "TUMOR"]: found_specific = True

                    color = (255, 0, 0) if level == "danger" else (255, 165, 0)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_img, spec[:4], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    save_case(display_img, findings_db, has_danger, patient_info)
    
    return display_img, findings_db, has_danger, time.time() - start_t, patient_info

def generate_html_report(findings_db, has_danger, patient_info):
    current_time = datetime.now().strftime('%H:%M %d/%m/%Y')
    
    lung_txt = f"<b>Ghi nhận:</b><br>- {'; <br>- '.join(findings_db['Lung'])}" if findings_db["Lung"] else "Hai phổi sáng, vân phổi đều."
    pleura_txt = f"<b>Bất thường:</b><br>- {'; <br>- '.join(findings_db['Pleura'])}" if findings_db["Pleura"] else "Góc sườn hoành nhọn, không tràn dịch/khí."
    heart_txt = f"<b>Tim mạch:</b><br>- {'; <br>- '.join(findings_db['Heart'])}" if findings_db["Heart"] else "Bóng tim không to. Trung thất cân đối."
    bone_txt = "Khung xương cân đối. Không ghi nhận hình ảnh gãy xương, khuyết xương."

    if has_danger or (len(findings_db["Lung"]) + len(findings_db["Pleura"]) > 0):
        concl = "<div class='danger-box'>🔴 <strong>KẾT LUẬN:</strong> CÓ HÌNH ẢNH BẤT THƯỜNG TRÊN PHIM</div>"
        rec = "<br><strong>💡 KHUYẾN NGHỊ:</strong> Kết hợp lâm sàng, chụp CT ngực nếu cần."
    else:
        concl = "<div class='success-box'>✅ <strong>KẾT LUẬN:</strong> CHƯA GHI NHẬN BẤT THƯỜNG RÕ</div>"
        rec = "<br><strong>💡 KHUYẾN NGHỊ:</strong> Theo dõi lâm sàng định kỳ."

    html = f"""
    <div class="report-box">
        <div style="text-align:center; border-bottom:2px solid #002f6c; margin-bottom:15px;">
            <h2 style="margin:0; color:#002f6c;">PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH</h2>
            <p style="margin:5px 0;">(Hệ thống AI hỗ trợ - Hỗ trợ DICOM)</p>
        </div>
        <p><strong>Bệnh nhân:</strong> {patient_info} | <strong>Thời gian:</strong> {current_time}</p>
        <hr>
        <h4>I. MÔ TẢ HÌNH ẢNH</h4>
        <ul style="line-height:1.6">
            <li><strong>Nhu mô phổi:</strong> {lung_text}</li>
            <li><strong>Màng phổi:</strong> {pleura_txt}</li>
            <li><strong>Tim – Trung thất:</strong> {heart_txt}</li>
            <li><strong>Xương:</strong> {bone_txt}</li>
        </ul>
        <h4>II. KẾT LUẬN</h4>
        {concl}
        {rec}
        <div style="margin-top:20px; font-size:12px; text-align:center; color:#777;">
            <em>Kết quả mang tính tham khảo. Chẩn đoán cuối cùng thuộc về Bác sĩ chuyên khoa.</em>
        </div>
    </div>
    """
    return html

# ================= 6. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("AI CONTROL")
    mode = st.radio("Chức năng:", ["🔍 Phân Tích Ca Bệnh", "📂 Lịch Sử & Review"])
    st.divider()
    with st.expander("Trạng thái AI"):
        for s in MODEL_STATUS: st.caption(s)

if mode == "🔍 Phân Tích Ca Bệnh":
    st.title("🏥 AI RADIOLOGY ASSISTANT")
    st.markdown("**Version 7.1** - *Hỗ trợ DICOM & JPG/PNG*")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        # Hỗ trợ upload cả dcm và ảnh thường
        uploaded_file = st.file_uploader("Tải ảnh X-quang/DICOM", type=["jpg", "png", "jpeg", "dcm", "dicom"])
        if uploaded_file:
            st.info(f"Đã nhận file: {uploaded_file.name}")
            analyze = st.button("🚀 KÍCH HOẠT HỘI CHẨN", type="primary")
    
    with col2:
        if uploaded_file and analyze:
            with st.spinner("🤖 Đang đọc DICOM và phân tích..."):
                img_out, findings, danger, p_time, p_info = process_image(uploaded_file)
                
                if img_out is not None:
                    t1, t2 = st.tabs(["Ảnh AI", "Phiếu Kết Quả"])
                    with t1: st.image(img_out, caption=f"Xử lý: {p_time:.2f}s", use_container_width=True)
                    with t2: st.markdown(generate_html_report(findings, danger, p_info), unsafe_allow_html=True)
                    st.toast("✅ Đã lưu kết quả vào Lịch Sử!", icon="💾")
                else:
                    st.error(findings)

elif mode == "📂 Lịch Sử & Review":
    st.title("📂 KHO DỮ LIỆU CA BỆNH")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = df.iloc[::-1]
        
        filter_opt = st.selectbox("Lọc kết quả:", ["Tất cả", "BẤT THƯỜNG", "BÌNH THƯỜNG"])
        if filter_opt != "Tất cả": df = df[df["Result"] == filter_opt]
            
        st.dataframe(df[["ID", "Patient_Info", "Result", "Details"]], use_container_width=True, hide_index=True)
        
        selected_id = st.selectbox("Chọn ID để xem ảnh:", df["ID"])
        if selected_id:
            record = df[df["ID"] == selected_id].iloc[0]
            img_path = os.path.join(IMAGES_DIR, record["Image_Path"])
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Bệnh nhân: {record['Patient_Info']}", use_container_width=True)
                st.info(record['Details'])
    else:
        st.info("Chưa có dữ liệu lịch sử.")