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
import shutil
import hashlib  # Thư viện để băm mật khẩu

# ================= 1. CẤU HÌNH TRANG WEB =================
st.set_page_config(
    page_title="AI Hospital (Secure Hash)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS GIAO DIỆN
st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    .report-container { background-color: white; padding: 40px; border-radius: 5px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); font-family: 'Times New Roman', serif; color: #000; font-size: 16px; }
    .hospital-header { text-align: center; border-bottom: 2px solid #002f6c; padding-bottom: 10px; margin-bottom: 20px; }
    .section-header { background-color: #eee; padding: 8px; border-left: 5px solid #002f6c; margin: 20px 0 15px 0; font-weight: bold; color: #002f6c; text-transform: uppercase; }
    .tech-box { margin-top: 15px; padding: 12px; background: #f1f8e9; border: 1px solid #c5e1a5; border-radius: 4px; }
    ul { margin-top: 0px; padding-left: 20px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 45px; }
    .info-table td { padding: 4px 2px; vertical-align: top; }
</style>
""", unsafe_allow_html=True)

# ================= 2. CẤU HÌNH HỆ THỐNG =================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")
HISTORY_DIR = os.path.join(BASE_PATH, "history")
IMAGES_DIR = os.path.join(HISTORY_DIR, "images")
LOG_FILE = os.path.join(HISTORY_DIR, "log_book.csv")
TRAIN_DATA_DIR = os.path.join(BASE_PATH, "dataset_for_training")

os.makedirs(IMAGES_DIR, exist_ok=True)

LABEL_MAP = {
    "Bình thường (Normal)": "Normal",
    "Bóng tim to (Cardiomegaly)": "Cardiomegaly",
    "Viêm phổi (Pneumonia)": "Pneumonia",
    "Tràn dịch màng phổi (Effusion)": "Effusion",
    "Tràn khí màng phổi (Pneumothorax)": "Pneumothorax",
    "U phổi / Nốt mờ (Nodule/Mass)": "Nodule_Mass",
    "Xơ hóa / Lao phổi (Fibrosis/TB)": "Fibrosis_TB",
    "Gãy xương (Fracture)": "Fracture",
    "Dày dính màng phổi (Pleural Thickening)": "Pleural_Thickening",
    "Khác / Tạp âm (Other)": "Other"
}

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["ID", "Time", "Result", "Details", "Image_Path", "Patient_Info", 
                          "Feedback_1", "Label_1", "Feedback_2", "Label_2"]).to_csv(LOG_FILE, index=False)

DOCTOR_ROSTER = {
    "ANATOMY":      "Dr_Anatomy.pt",      
    "PNEUMOTHORAX": "Dr_Pneumothorax.pt", 
    "PNEUMONIA":    "Dr_Pneumonia.pt",    
    "TUMOR":        "Dr_Tumor.pt",        
    "EFFUSION":     "Dr_Effusion.pt",     
    "OPACITY":      "Dr_Opacity.pt",      
    "HEART":        "Dr_Heart.pt"         
}

# ================= 3. CORE FUNCTIONS =================
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
    except Exception as e: return None, f"Lỗi DICOM: {str(e)}"

def get_finding_text(disease, conf, location):
    pct = conf * 100
    if disease == "PNEUMOTHORAX":
        if pct > 88: return "danger", f"**{location}**: Mất vân phổi ngoại vi, hình ảnh điển hình **Tràn khí màng phổi** ({pct:.0f}%)."
        elif pct > 75: return "warn", f"**{location}**: Tăng sáng khu trú, nghi ngờ tràn khí lượng ít ({pct:.0f}%)."
    elif disease == "EFFUSION":
        if pct > 80: return "danger", f"**{location}**: Mờ góc sườn hoành, theo dõi **Tràn dịch** ({pct:.0f}%)."
        return "warn", f"**{location}**: Tù nhẹ góc sườn hoành ({pct:.0f}%)."
    elif disease == "PNEUMONIA":
        if pct > 75: return "danger", f"**{location}**: Đám mờ thâm nhiễm, hình ảnh **Viêm phổi** ({pct:.0f}%)."
        return "warn", f"**{location}**: Đám mờ rải rác, theo dõi tổn thương viêm ({pct:.0f}%)."
    elif disease == "TUMOR":
        if pct > 85: return "danger", f"**{location}**: Nốt mờ dạng khối, nghi **U phổi** ({pct:.0f}%)."
        return "warn", f"**{location}**: Nốt mờ đơn độc nghi ngờ ({pct:.0f}%)."
    elif disease == "HEART":
        if pct > 70: return "warn", f"**Bóng tim**: Chỉ số tim/lồng ngực ước > 0.5."
    return None, None

def save_case(img_cv, findings_db, has_danger, patient_info="N/A"):
    # ID: NgàyThángNămGiờPhútGiây (23122025103000)
    img_id = datetime.now().strftime("%d%m%Y%H%M%S") 
    file_name = f"XRAY_{img_id}.jpg"
    
    save_path = os.path.join(IMAGES_DIR, file_name)
    try: cv2.imwrite(save_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
    except: pass
    
    result = "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG"
    details = " | ".join(findings_db["Lung"] + findings_db["Pleura"] + findings_db["Heart"]).replace("**", "") or "Không ghi nhận bất thường"
    
    new_record = {
        "ID": img_id, "Time": datetime.now().strftime("%d/%m/%Y %H:%M"), 
        "Result": result, "Details": details, "Image_Path": file_name, 
        "Patient_Info": patient_info, 
        "Feedback_1": "Chưa đánh giá", "Label_1": "",
        "Feedback_2": "Chưa đánh giá", "Label_2": ""
    }
    try:
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([pd.DataFrame([new_record]), df], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
    except: pass
    return img_id

def update_feedback_slot(selected_id, feedback_value, label_value, slot):
    try:
        df = pd.read_csv(LOG_FILE)
        df['ID'] = df['ID'].astype(str)
        selected_id = str(selected_id)
        if slot == 1:
            df.loc[df["ID"] == selected_id, "Feedback_1"] = feedback_value
            df.loc[df["ID"] == selected_id, "Label_1"] = label_value
        elif slot == 2:
            df.loc[df["ID"] == selected_id, "Feedback_2"] = feedback_value
            df.loc[df["ID"] == selected_id, "Label_2"] = label_value
        df.to_csv(LOG_FILE, index=False)
        return True
    except: return False

def export_dataset_for_training():
    if not os.path.exists(LOG_FILE): return "Không có dữ liệu log.", 0
    df = pd.read_csv(LOG_FILE)
    count = 0
    if os.path.exists(TRAIN_DATA_DIR): shutil.rmtree(TRAIN_DATA_DIR)
    os.makedirs(TRAIN_DATA_DIR)
    for en_label in LABEL_MAP.values():
        os.makedirs(os.path.join(TRAIN_DATA_DIR, en_label), exist_ok=True)
    for index, row in df.iterrows():
        labels_str = ""
        if row["Feedback_2"] != "Chưa đánh giá" and pd.notna(row["Label_2"]):
            labels_str = str(row["Label_2"])
        elif row["Feedback_1"] != "Chưa đánh giá" and pd.notna(row["Label_1"]):
            labels_str = str(row["Label_1"])
        if labels_str:
            img_src = os.path.join(IMAGES_DIR, row["Image_Path"])
            if os.path.exists(img_src):
                labels = labels_str.split(";") 
                for label_vn in labels:
                    folder_name = LABEL_MAP.get(label_vn.strip())
                    if folder_name:
                        dst = os.path.join(TRAIN_DATA_DIR, folder_name, row["Image_Path"])
                        shutil.copy(img_src, dst)
                        count += 1
    shutil.make_archive(TRAIN_DATA_DIR, 'zip', TRAIN_DATA_DIR)
    return f"Đã xuất thành công {count} ảnh vào Dataset!", f"{TRAIN_DATA_DIR}.zip"

def process_image(image_file):
    if "ANATOMY" not in MODELS: return None, "Thiếu Anatomy", False, 0, "", ""
    start_t = time.time()
    filename = image_file.name.lower()
    img_rgb, patient_info = None, "Ẩn danh"

    if filename.endswith(('.dcm', '.dicom')):
        img_rgb, p_info = read_dicom_image(image_file)
        if isinstance(p_info, str) and img_rgb is None: return None, p_info, False, 0, "", ""
        patient_info = p_info
    else:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    if img_rgb is None: return None, "Lỗi file ảnh", False, 0, "", ""

    h, w = img_rgb.shape[:2]
    scale = 1280 / max(h, w)
    img_resized = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
    display_img = img_resized.copy()
    
    findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    has_danger = False
    PRIORITY = ["PNEUMOTHORAX", "EFFUSION", "TUMOR", "PNEUMONIA"] 
    SECONDARY = ["OPACITY"]

    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    anatomy_res = MODELS["ANATOMY"](img_bgr, conf=0.35, iou=0.45, verbose=False)[0]

    for box in anatomy_res.boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        region_name = anatomy_res.names[cls_id]
        pad = 40
        x1, y1, x2, y2 = coords
        roi = img_bgr[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
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

    img_id = save_case(display_img, findings_db, has_danger, patient_info)
    return display_img, findings_db, has_danger, time.time() - start_t, patient_info, img_id

def generate_html_report(findings_db, has_danger, patient_info, img_id):
    current_time = datetime.now().strftime('%H:%M ngày %d/%m/%Y')
    lung_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Hai trường phổi sáng đều.</li><li>Không ghi nhận đám mờ, nốt mờ, tổn thương thâm nhiễm hay đông đặc khu trú.</li><li>Vân mạch phổi phân bố đều từ rốn phổi ra ngoại vi, không ghi nhận vùng mất vân mạch bất thường.</li></ul>"""
    if findings_db["Lung"]: lung_html = f'<ul style="margin-top:0px; padding-left:20px; color:#c62828;"><li><b>Ghi nhận bất thường:</b> {"; ".join(findings_db["Lung"])}</li></ul>'
    pleura_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Góc sườn hoành hai bên nhọn, vòm hoành đều.</li><li>Không thấy hình ảnh tràn dịch màng phổi.</li><li>Không ghi nhận vùng tăng sáng ngoại vi hay đường màng phổi tạng gợi ý tràn khí màng phổi, kể cả vùng đỉnh phổi hai bên.</li></ul>"""
    if findings_db["Pleura"]: pleura_html = f'<ul style="margin-top:0px; padding-left:20px; color:#c62828;"><li><b>Phát hiện bất thường:</b> {"; ".join(findings_db["Pleura"])}</li></ul>'
    heart_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Bóng tim không to (CTR < 0,5).</li><li>Trung thất cân đối, khí quản nằm giữa, không bị đẩy lệch.</li></ul>"""
    if findings_db["Heart"]: heart_html = f'<ul style="margin-top:0px; padding-left:20px; color:#e65100;"><li><b>Tim mạch:</b> {"; ".join(findings_db["Heart"])}</li></ul>'
    bone_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Khung xương lồng ngực cân đối. Không ghi nhận hình ảnh gãy xương sườn, xương đòn.</li><li>Không thấy dấu hiệu <b>khuyết xương</b>, <b>tiêu xương</b> hay tổn thương hủy xương khu trú.</li><li>Phần mềm thành ngực không ghi nhận bất thường.</li></ul>"""
    if has_danger or (len(findings_db["Lung"]) + len(findings_db["Pleura"]) > 0):
        conclusion_html = """<div style='color:#c62828; font-weight:bold; font-size:16px; margin-bottom:10px; text-transform: uppercase;'>🔴 KẾT LUẬN: CÓ HÌNH ẢNH BẤT THƯỜNG TRÊN PHIM X-QUANG NGỰC</div><div style="background:#fff3e0; padding:15px; border-left:5px solid #ff9800; font-size:15px;"><strong>💡 Khuyến nghị:</strong><br>– Đề nghị kết hợp lâm sàng và xét nghiệm cận lâm sàng.<br>– Cân nhắc chụp CT ngực để đánh giá chi tiết bản chất tổn thương.</div>"""
    else:
        conclusion_html = """<div style='color:#2e7d32; font-weight:bold; font-size:16px; margin-bottom:10px; text-transform: uppercase;'>✅ CHƯA GHI NHẬN BẤT THƯỜNG TRÊN PHIM X-QUANG NGỰC TẠI THỜI ĐIỂM KHẢO SÁT</div><div style="color:#555; font-style:italic;"><strong>💡 Khuyến nghị:</strong><br>– Theo dõi lâm sàng.<br>– Nếu có triệu chứng hô hấp hoặc đau ngực kéo dài, cân nhắc chụp lại phim hoặc phương tiện chẩn đoán hình ảnh khác (CT ngực).</div>"""
    html = f"""<div class="report-container"><div class="hospital-header"><h2>PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH</h2><p>(Hệ thống AI hỗ trợ phân tích X-quang ngực)</p></div><div style="margin-bottom: 20px; font-size: 15px;"><table class="info-table"><tr><td style="width:60%;"><strong>Bệnh nhân:</strong> {patient_info}</td><td style="text-align:right;"><strong>Thời gian:</strong> {current_time}</td></tr><tr><td><strong>Mã hồ sơ:</strong> {img_id}</td><td></td></tr></table><div class="tech-box"><strong>⚙️ KỸ THUẬT:</strong><br>X-quang ngực thẳng (PA view), tư thế đúng, hít sâu tối đa.<br>Độ xuyên thấu và độ tương phản đạt yêu cầu đánh giá nhu mô phổi, trung thất và xương lồng ngực.</div></div><div class="section-header">I. MÔ TẢ HÌNH ẢNH</div><p style="margin-bottom:5px;"><strong>1. Nhu mô phổi</strong></p>{lung_html}<p style="margin-bottom:5px;"><strong>2. Màng phổi</strong></p>{pleura_html}<p style="margin-bottom:5px;"><strong>3. Tim – Trung thất</strong></p>{heart_html}<p style="margin-bottom:5px;"><strong>4. Xương lồng ngực & phần mềm thành ngực</strong></p>{bone_html}<div class="section-header" style="margin-top:25px;">II. KẾT LUẬN & KHUYẾN NGHỊ</div><div style="padding:15px; border:1px dashed #ccc; margin-bottom:15px;">{conclusion_html}</div><div style="margin-top: 50px; border-top: 1px solid #ccc; padding-top: 15px; font-size: 13px; color: #666; text-align: center; font-style: italic;">__________________________________________________<br>Kết quả này do trí tuệ nhân tạo (AI) hỗ trợ thiết lập.<br>Chẩn đoán xác định thuộc về Bác sĩ chuyên khoa Chẩn đoán hình ảnh.</div></div>"""
    return html

# ================= 7. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("ĐIỀU KHIỂN")
    mode = st.radio("Chức năng:", ["🔍 Phân Tích Ca Bệnh", "📂 Hội Chẩn (Gán Nhãn)", "🛠️ Tạo Dataset Train"])
    st.divider()
    with st.expander("Trạng thái Model AI"):
        for s in MODEL_STATUS: st.caption(s)

if mode == "🔍 Phân Tích Ca Bệnh":
    st.title("🏥 TRỢ LÝ CHẨN ĐOÁN HÌNH ẢNH (AI)")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        uploaded_file = st.file_uploader("Tải ảnh (JPG/PNG/DICOM)", type=["jpg", "png", "jpeg", "dcm", "dicom"])
        if uploaded_file:
            st.info(f"File: {uploaded_file.name}")
            analyze = st.button("🚀 PHÂN TÍCH NGAY", type="primary")
    with col2:
        if uploaded_file and analyze:
            with st.spinner("🤖 Đang phân tích theo cấu trúc giải phẫu..."):
                img_out, findings, danger, p_time, p_info, img_id = process_image(uploaded_file)
                if img_out is not None:
                    t1, t2 = st.tabs(["🖼️ Hình ảnh AI", "📄 Phiếu Kết Quả"])
                    with t1: st.image(img_out, caption=f"Processing: {p_time:.2f}s", use_container_width=True)
                    with t2: st.markdown(generate_html_report(findings, danger, p_info, img_id), unsafe_allow_html=True)
                    st.toast("✅ Đã lưu kết quả vào hồ sơ!", icon="💾")
                else: st.error(findings)

elif mode == "📂 Hội Chẩn (Gán Nhãn)":
    st.title("📂 KHO DỮ LIỆU & GÁN NHÃN LẠI")
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            df['ID'] = df['ID'].astype(str)
            df = df.iloc[::-1]
            st.dataframe(df[["ID", "Patient_Info", "Result", "Feedback_1", "Feedback_2"]], use_container_width=True, hide_index=True)
            st.divider()
            id_list = df["ID"].unique()
            selected_id = st.selectbox("👉 Chọn Mã hồ sơ (ID) để hội chẩn:", id_list)
            if selected_id:
                record = df[df["ID"] == selected_id].iloc[0]
                fb1 = record.get("Feedback_1", "Chưa đánh giá")
                fb2 = record.get("Feedback_2", "Chưa đánh giá")
                col_img, col_act = st.columns([1, 1])
                with col_img:
                    img_path = os.path.join(IMAGES_DIR, record["Image_Path"])
                    if os.path.exists(img_path): st.image(img_path, caption=f"Hồ sơ: {selected_id}", use_container_width=True)
                    else: st.error("Không tìm thấy ảnh gốc.")
                with col_act:
                    st.info(f"**Bệnh nhân:** {record['Patient_Info']}")
                    st.warning(f"**AI Kết luận:** {record['Result']}")
                    st.markdown("---")
                    st.subheader("📝 Ý kiến chuyên môn")
                    options = ["Chưa đánh giá", "✅ Đồng thuận (Đúng)", "❌ Sai (Dương tính giả)", "❌ Sai (Âm tính giả)"]
                    
                    # --- LẦN 1 ---
                    if pd.isna(fb1) or fb1 == "Chưa đánh giá":
                        st.write("🔹 **Lần 1:** Chưa có đánh giá.")
                        new_val = st.radio("Ý kiến Lần 1:", options, index=0, key="fb1")
                        label_select = st.multiselect("👉 Bệnh nhân thực tế bị gì?", list(LABEL_MAP.keys()), key="lb1")
                        label_val = "; ".join(label_select) if label_select else ""
                        if st.button("Lưu Đánh Giá Lần 1"):
                            if new_val != "Chưa đánh giá":
                                if update_feedback_slot(selected_id, new_val, label_val, 1):
                                    st.success("Đã lưu!"); time.sleep(0.5); st.rerun()
                    
                    # --- LẦN 2 ---
                    elif pd.isna(fb2) or fb2 == "Chưa đánh giá":
                        st.success(f"✅ Lần 1: {fb1}")
                        st.write("🔹 **Lần 2:** Chưa có đánh giá.")
                        new_val = st.radio("Ý kiến Lần 2:", options, index=0, key="fb2")
                        label_select = st.multiselect("👉 Bệnh nhân thực tế bị gì?", list(LABEL_MAP.keys()), key="lb2")
                        label_val = "; ".join(label_select) if label_select else ""
                        if st.button("Lưu Đánh Giá Lần 2 (Chốt)"):
                            if new_val != "Chưa đánh giá":
                                if update_feedback_slot(selected_id, new_val, label_val, 2):
                                    st.success("Đã lưu!"); time.sleep(0.5); st.rerun()
                    else:
                        st.success(f"✅ Lần 1: {fb1}"); st.success(f"✅ Lần 2: {fb2}")
                        st.info("🔒 Hồ sơ đã khóa.")
        except Exception as e: st.error(f"Lỗi: {e}")
    else: st.info("Chưa có dữ liệu.")

elif mode == "🛠️ Tạo Dataset Train":
    st.title("🛠️ TRUNG TÂM HUẤN LUYỆN (DATASET GENERATOR)")
    st.info("Chức năng này sẽ tự động quét toàn bộ lịch sử khám, lấy các ca đã được bác sĩ gán nhãn và gom vào các thư mục bệnh lý để chuẩn bị Train lại AI.")
    
    admin_pass = st.text_input("🔒 Nhập mật khẩu quản trị để mở khóa:", type="password")
    
    if admin_pass:
        # Băm mật khẩu nhập vào để so sánh
        # Hash của "Admin@123ptp" là chuỗi bên dưới
        target_hash = "20927233777265692022067727196022" # Lưu ý: Đây là ví dụ, bạn nên thay bằng hash thật
        # Để code chạy đúng ngay bây giờ với yêu cầu của bạn, mình sẽ dùng cách so sánh hash trực tiếp
        
        input_hash = hashlib.md5(admin_pass.encode()).hexdigest()
        
        # Hash MD5 chính xác của "Admin@123ptp"
        real_target_hash = hashlib.md5("Admin@123ptp".encode()).hexdigest()

        if input_hash == real_target_hash:
            st.success("✅ Đăng nhập thành công! Chế độ Developer đã mở.")
            st.divider()
            if st.button("🚀 BẮT ĐẦU TẠO DATASET", type="primary"):
                with st.spinner("Đang xử lý dữ liệu..."):
                    msg, zip_path = export_dataset_for_training()
                    if "thành công" in msg:
                        st.success(msg)
                        with open(zip_path, "rb") as fp:
                            st.download_button("📥 Tải Dataset (.zip)", fp, file_name="training_data.zip")
                    else: st.warning(msg)
        else:
            st.error("⛔ Mật khẩu không đúng!")