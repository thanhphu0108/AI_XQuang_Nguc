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
import hashlib
import random

# ================= 1. CẤU HÌNH TRANG WEB =================
st.set_page_config(
    page_title="AI Hospital (Dataset Intelligence)",
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
TRAIN_DATA_DIR = os.path.join(BASE_PATH, "dataset_yolo_ready")

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
    pd.DataFrame(columns=["ID", "Time", "Result", "Image_Path", "Patient_Info", 
                          "Feedback_1", "Label_1", "Feedback_2", "Label_2"]).to_csv(LOG_FILE, index=False)

DOCTOR_ROSTER = {
    "ANATOMY": "Dr_Anatomy.pt",      
    "PNEUMOTHORAX": "Dr_Pneumothorax.pt", 
    "PNEUMONIA": "Dr_Pneumonia.pt",    
    "TUMOR": "Dr_Tumor.pt",        
    "EFFUSION": "Dr_Effusion.pt",     
    "OPACITY": "Dr_Opacity.pt",      
    "HEART": "Dr_Heart.pt"         
}

# ================= 3. CORE FUNCTIONS =================
@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else 'cpu'
    loaded_models = {}
    for role, filename in DOCTOR_ROSTER.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                m = YOLO(path)
                if device == 0: m.to('cuda')
                loaded_models[role] = m
            except: pass
    return loaded_models, [], device

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
    except: return None, "Lỗi DICOM"

def get_finding_text(disease, conf, location):
    pct = conf * 100
    if disease == "PNEUMOTHORAX":
        if pct > 88: return "danger", f"**{location}**: Điển hình Tràn khí ({pct:.0f}%)."
        elif pct > 75: return "warn", f"**{location}**: Nghi ngờ Tràn khí ({pct:.0f}%)."
    elif disease == "EFFUSION":
        if pct > 80: return "danger", f"**{location}**: Theo dõi Tràn dịch ({pct:.0f}%)."
        return "warn", f"**{location}**: Tù góc sườn hoành ({pct:.0f}%)."
    elif disease == "PNEUMONIA":
        if pct > 75: return "danger", f"**{location}**: Thâm nhiễm Viêm ({pct:.0f}%)."
        return "warn", f"**{location}**: Tổn thương mờ ({pct:.0f}%)."
    elif disease == "TUMOR":
        if pct > 85: return "danger", f"**{location}**: Khối u/Nốt mờ ({pct:.0f}%)."
        return "warn", f"**{location}**: Nốt mờ nghi ngờ ({pct:.0f}%)."
    elif disease == "HEART":
        if pct > 70: return "warn", f"**Bóng tim**: To > 0.5 ({pct:.0f}%)."
    return None, None

def save_case(img_cv, findings_db, has_danger, patient_info="N/A"):
    img_id = datetime.now().strftime("%d%m%Y%H%M%S") 
    file_name = f"XRAY_{img_id}.jpg"
    save_path = os.path.join(IMAGES_DIR, file_name)
    try: cv2.imwrite(save_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
    except: pass
    
    result = "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG"
    details = " | ".join(findings_db["Lung"] + findings_db["Pleura"] + findings_db["Heart"]).replace("**", "") or "Bình thường"
    
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

# --- HÀM TÍNH TOÁN NHÃN CHỐT (PRIORITY LOGIC) ---
def get_final_label(row):
    # Ưu tiên Label 2 nếu có
    if pd.notna(row["Label_2"]) and row["Label_2"] != "" and row["Feedback_2"] != "Chưa đánh giá":
        return row["Label_2"]
    # Nếu không, lấy Label 1
    elif pd.notna(row["Label_1"]) and row["Label_1"] != "" and row["Feedback_1"] != "Chưa đánh giá":
        return row["Label_1"]
    return ""

def preview_auto_label(df_selected):
    if df_selected.empty: return None, "Chưa chọn dòng nào!"
    random_row = df_selected.sample(1).iloc[0]
    img_path = os.path.join(IMAGES_DIR, random_row["Image_Path"])
    if not os.path.exists(img_path): return None, "Không tìm thấy file ảnh gốc!"
    
    img = cv2.imread(img_path)
    anatomy_model = MODELS.get("ANATOMY")
    
    detected_classes = [] 
    if anatomy_model:
        results = anatomy_model(img, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            label_name = anatomy_model.names[cls_id]
            detected_classes.append(label_name)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Lấy nhãn chốt để hiển thị
    final_label = get_final_label(random_row)
    
    msg = f"""
    🖼️ **File:** {random_row['Image_Path']}
    🏆 **Nhãn chốt (Final Label):** {final_label if final_label else '⚠️ Chưa gán nhãn'}
    🤖 **AI Giải phẫu:** {', '.join(set(detected_classes))}
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), msg

# --- HÀM XUẤT DATASET (OPTIMIZED) ---
def export_selected_data(df_selected, use_anatomy_auto_label=True):
    count = 0
    if os.path.exists(TRAIN_DATA_DIR): shutil.rmtree(TRAIN_DATA_DIR)
    
    os.makedirs(os.path.join(TRAIN_DATA_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DATA_DIR, "labels"), exist_ok=True)
    
    # Tạo cấu trúc thư mục phân loại (nếu muốn dùng yolo classify)
    for en_label in LABEL_MAP.values():
        os.makedirs(os.path.join(TRAIN_DATA_DIR, "classified", en_label), exist_ok=True)
    
    anatomy_model = MODELS.get("ANATOMY")
    
    # Tạo file classes.txt
    if anatomy_model:
        with open(os.path.join(TRAIN_DATA_DIR, "classes.txt"), "w") as f:
            for idx, name in anatomy_model.names.items(): f.write(f"{name}\n")
    
    progress_bar = st.progress(0)
    total = len(df_selected)
    
    for idx, (index, row) in enumerate(df_selected.iterrows()):
        # ÁP DỤNG LOGIC CHỐT: Lấy Lần 2 -> Lần 1
        labels_str = get_final_label(row)
        
        img_src = os.path.join(IMAGES_DIR, row["Image_Path"])
        
        if os.path.exists(img_src) and labels_str:
            # Xử lý Đa nhãn (Multi-label split by ;)
            label_list = labels_str.split(";")
            
            # 1. Copy vào thư mục phân loại (Classified Folders)
            for lbl_vn in label_list:
                folder_name = LABEL_MAP.get(lbl_vn.strip())
                if folder_name:
                    dst_class = os.path.join(TRAIN_DATA_DIR, "classified", folder_name, row["Image_Path"])
                    shutil.copy(img_src, dst_class)

            # 2. Chuẩn bị cho Detection (YOLO Format)
            # Lấy tên bệnh chính (cái đầu tiên) làm prefix
            primary_disease = label_list[0].strip()
            folder_prefix = LABEL_MAP.get(primary_disease, "Unknown")
            new_filename = f"{folder_prefix}_{row['Image_Path']}"
            
            dst_img = os.path.join(TRAIN_DATA_DIR, "images", new_filename)
            shutil.copy(img_src, dst_img)
            
            # Auto-Label Anatomy
            if use_anatomy_auto_label and anatomy_model:
                try:
                    results = anatomy_model(img_src, verbose=False)[0]
                    txt_content = ""
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        x, y, w, h = box.xywhn[0].tolist()
                        txt_content += f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                    
                    dst_txt = os.path.join(TRAIN_DATA_DIR, "labels", new_filename.replace(".jpg", ".txt").replace(".png", ".txt"))
                    with open(dst_txt, "w") as f: f.write(txt_content)
                except: pass
            count += 1
        progress_bar.progress((idx + 1) / total)
            
    shutil.make_archive(TRAIN_DATA_DIR, 'zip', TRAIN_DATA_DIR)
    return f"Đã xuất {count} ảnh (Ưu tiên nhãn Lần 2)!", f"{TRAIN_DATA_DIR}.zip"

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
        for spec in target_models:
            if spec not in MODELS: continue
            if spec == "OPACITY": continue
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
                    color = (255, 0, 0) if level == "danger" else (255, 165, 0)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_img, spec[:4], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    img_id = save_case(display_img, findings_db, has_danger, patient_info)
    return display_img, findings_db, has_danger, time.time() - start_t, patient_info, img_id

def generate_html_report(findings_db, has_danger, patient_info, img_id):
    current_time = datetime.now().strftime('%H:%M ngày %d/%m/%Y')
    lung_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Hai trường phổi sáng đều.</li><li>Không ghi nhận đám mờ...</li></ul>"""
    if findings_db["Lung"]: lung_html = f'<ul style="margin-top:0px; padding-left:20px; color:#c62828;"><li><b>Ghi nhận bất thường:</b> {"; ".join(findings_db["Lung"])}</li></ul>'
    pleura_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Góc sườn hoành hai bên nhọn...</li></ul>"""
    if findings_db["Pleura"]: pleura_html = f'<ul style="margin-top:0px; padding-left:20px; color:#c62828;"><li><b>Phát hiện bất thường:</b> {"; ".join(findings_db["Pleura"])}</li></ul>'
    heart_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Bóng tim không to (CTR < 0,5).</li></ul>"""
    if findings_db["Heart"]: heart_html = f'<ul style="margin-top:0px; padding-left:20px; color:#e65100;"><li><b>Tim mạch:</b> {"; ".join(findings_db["Heart"])}</li></ul>'
    bone_html = """<ul style="margin-top:0px; padding-left:20px;"><li>Khung xương lồng ngực cân đối...</li></ul>"""
    if has_danger or (len(findings_db["Lung"]) + len(findings_db["Pleura"]) > 0):
        conclusion_html = """<div style='color:#c62828; font-weight:bold; font-size:16px; margin-bottom:10px; text-transform: uppercase;'>🔴 KẾT LUẬN: CÓ HÌNH ẢNH BẤT THƯỜNG...</div>"""
    else:
        conclusion_html = """<div style='color:#2e7d32; font-weight:bold; font-size:16px; margin-bottom:10px; text-transform: uppercase;'>✅ CHƯA GHI NHẬN BẤT THƯỜNG...</div>"""
    html = f"""<div class="report-container"><div class="hospital-header"><h2>PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH</h2><p>(Hệ thống AI hỗ trợ phân tích X-quang ngực)</p></div><div style="margin-bottom: 20px; font-size: 15px;"><table class="info-table"><tr><td style="width:60%;"><strong>Bệnh nhân:</strong> {patient_info}</td><td style="text-align:right;"><strong>Thời gian:</strong> {current_time}</td></tr><tr><td><strong>Mã hồ sơ:</strong> {img_id}</td><td></td></tr></table><div class="tech-box"><strong>⚙️ KỸ THUẬT:</strong><br>X-quang ngực thẳng (PA view), tư thế đúng, hít sâu tối đa.</div></div><div class="section-header">I. MÔ TẢ HÌNH ẢNH</div><p style="margin-bottom:5px;"><strong>1. Nhu mô phổi</strong></p>{lung_html}<p style="margin-bottom:5px;"><strong>2. Màng phổi</strong></p>{pleura_html}<p style="margin-bottom:5px;"><strong>3. Tim – Trung thất</strong></p>{heart_html}<p style="margin-bottom:5px;"><strong>4. Xương</strong></p>{bone_html}<div class="section-header" style="margin-top:25px;">II. KẾT LUẬN</div><div style="padding:15px; border:1px dashed #ccc; margin-bottom:15px;">{conclusion_html}</div></div>"""
    return html

# ================= 7. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("ĐIỀU KHIỂN")
    mode = st.radio("Chức năng:", ["🔍 Phân Tích Ca Bệnh", "📂 Hội Chẩn (Gán Nhãn)", "🛠️ Tạo Dataset Train"])
    st.divider()

if mode == "🔍 Phân Tích Ca Bệnh":
    st.title("🏥 TRỢ LÝ CHẨN ĐOÁN HÌNH ẢNH (AI)")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        uploaded_file = st.file_uploader("Tải ảnh", type=["jpg", "png", "jpeg", "dcm", "dicom"])
        if uploaded_file:
            st.info(f"File: {uploaded_file.name}")
            analyze = st.button("🚀 PHÂN TÍCH NGAY", type="primary")
    with col2:
        if uploaded_file and analyze:
            with st.spinner("🤖 Đang phân tích..."):
                img_out, findings, danger, p_time, p_info, img_id = process_image(uploaded_file)
                if img_out is not None:
                    t1, t2 = st.tabs(["🖼️ Hình ảnh AI", "📄 Phiếu Kết Quả"])
                    with t1: st.image(img_out, caption=f"Time: {p_time:.2f}s", use_container_width=True)
                    with t2: st.markdown(generate_html_report(findings, danger, p_info, img_id), unsafe_allow_html=True)
                    st.toast("✅ Đã lưu kết quả!", icon="💾")
                else: st.error(findings)

elif mode == "📂 Hội Chẩn (Gán Nhãn)":
    st.title("📂 KHO DỮ LIỆU & GÁN NHÃN LẠI")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df['ID'] = df['ID'].astype(str)
        df = df.iloc[::-1]
        st.dataframe(df[["ID", "Patient_Info", "Result", "Feedback_1", "Feedback_2"]], use_container_width=True, hide_index=True)
        st.divider()
        id_list = df["ID"].unique()
        selected_id = st.selectbox("👉 Chọn Mã hồ sơ (ID) để hội chẩn:", id_list)
        if selected_id:
            record = df[df["ID"] == selected_id].iloc[0]
            col_img, col_act = st.columns([1, 1])
            with col_img:
                img_path = os.path.join(IMAGES_DIR, record["Image_Path"])
                if os.path.exists(img_path): st.image(img_path, caption=f"Hồ sơ: {selected_id}", use_container_width=True)
            with col_act:
                st.info(f"**BN:** {record['Patient_Info']} | **AI:** {record['Result']}")
                st.markdown("---")
                options = ["Chưa đánh giá", "✅ Đồng thuận (Đúng)", "❌ Sai (Dương tính giả)", "❌ Sai (Âm tính giả)"]
                
                fb1 = record.get("Feedback_1", "Chưa đánh giá")
                fb2 = record.get("Feedback_2", "Chưa đánh giá")
                
                if pd.isna(fb1) or fb1 == "Chưa đánh giá":
                    st.write("🔹 **Lần 1:** Chưa có đánh giá.")
                    new_val = st.radio("Ý kiến Lần 1:", options, index=0, key="fb1")
                    label_select = st.multiselect("👉 Bệnh nhân thực tế bị gì?", list(LABEL_MAP.keys()), key="lb1")
                    if st.button("Lưu Đánh Giá Lần 1"):
                        if update_feedback_slot(selected_id, new_val, "; ".join(label_select), 1):
                            st.success("Đã lưu!"); time.sleep(0.5); st.rerun()
                elif pd.isna(fb2) or fb2 == "Chưa đánh giá":
                    st.success(f"✅ Lần 1: {fb1}")
                    st.write("🔹 **Lần 2:** Chưa có đánh giá.")
                    new_val = st.radio("Ý kiến Lần 2:", options, index=0, key="fb2")
                    label_select = st.multiselect("👉 Bệnh nhân thực tế bị gì?", list(LABEL_MAP.keys()), key="lb2")
                    if st.button("Lưu Đánh Giá Lần 2 (Chốt)"):
                        if update_feedback_slot(selected_id, new_val, "; ".join(label_select), 2):
                            st.success("Đã lưu!"); time.sleep(0.5); st.rerun()
                else:
                    st.success(f"✅ Lần 1: {fb1}"); st.success(f"✅ Lần 2: {fb2}"); st.info("🔒 Hồ sơ đã khóa.")
    else: st.info("Chưa có dữ liệu.")

elif mode == "🛠️ Tạo Dataset Train":
    st.title("🛠️ DATASET MASTER (AUTO-LABEL)")
    admin_pass = st.text_input("🔒 Nhập mật khẩu quản trị:", type="password")
    if admin_pass:
        if hashlib.md5(admin_pass.encode()).hexdigest() == hashlib.md5("Admin@123456p".encode()).hexdigest():
            st.success("✅ Đã mở khóa Developer Mode!")
            if os.path.exists(LOG_FILE):
                df = pd.read_csv(LOG_FILE)
                # TÍNH TOÁN CỘT "NHÃN CUỐI CÙNG" ĐỂ HIỂN THỊ
                df["Final_Label"] = df.apply(get_final_label, axis=1)
                
                df["Select"] = False
                st.write("### 📋 Chọn ca để xuất dữ liệu:")
                
                # Highlight dòng mâu thuẫn (Lần 1 != Lần 2)
                st.info("💡 Lưu ý: Cột 'Final_Label' sẽ ưu tiên lấy đánh giá Lần 2 (nếu có).")
                
                df_editor = st.data_editor(
                    df[["Select", "ID", "Patient_Info", "Label_1", "Label_2", "Final_Label"]],
                    column_config={"Select": st.column_config.CheckboxColumn("Chọn", default=False)},
                    hide_index=True, use_container_width=True
                )
                selected_rows = df_editor[df_editor["Select"] == True]
                df_final = df.iloc[selected_rows.index]
                st.write(f"Đang chọn: **{len(df_final)}** ca.")
                
                c1, c2, c3 = st.columns(3)
                auto_label = c1.checkbox("🤖 Auto-Label Anatomy", value=True)
                
                if c2.button("👁️ Xem thử"):
                    prev_img, prev_msg = preview_auto_label(df_final)
                    if prev_img is not None: st.image(prev_img, caption=prev_msg, width=500)
                    else: st.warning(prev_msg)
                
                if c3.button("🚀 XUẤT DATASET"):
                    if not df_final.empty:
                        with st.spinner("Đang xử lý..."):
                            msg, zip_path = export_selected_data(df_final, use_anatomy_auto_label=auto_label)
                            st.success(msg)
                            with open(zip_path, "rb") as fp: st.download_button("📥 Tải Dataset (.zip)", fp, file_name="yolo_dataset_master.zip")
                    else: st.warning("Vui lòng chọn ít nhất 1 ca!")
            else: st.info("Chưa có dữ liệu.")
        else: st.error("⛔ Mật khẩu sai!")