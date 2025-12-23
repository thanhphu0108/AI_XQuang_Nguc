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
import base64
import json
from openai import OpenAI

# ================= 1. CẤU HÌNH TRANG WEB =================
st.set_page_config(
    page_title="AI Hospital (AI Teacher Mode)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS GIAO DIỆN
st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    .report-container { background-color: white; padding: 40px; border-radius: 5px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 45px; }
    .gpt-suggestion { background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50; margin-bottom: 10px; }
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

# DANH SÁCH BỆNH CHUẨN (Để ép ChatGPT trả lời đúng form này)
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
# Tạo list tên bệnh để đưa vào Prompt
ALLOWED_LABELS = list(LABEL_MAP.keys())

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["ID", "Time", "Result", "Image_Path", "Patient_Info", 
                          "Feedback_1", "Label_1", "Feedback_2", "Label_2", "GPT_Reasoning"]).to_csv(LOG_FILE, index=False)

DOCTOR_ROSTER = {
    "ANATOMY": "Dr_Anatomy.pt",      
    "PNEUMOTHORAX": "Dr_Pneumothorax.pt", "PNEUMONIA": "Dr_Pneumonia.pt",    
    "TUMOR": "Dr_Tumor.pt", "EFFUSION": "Dr_Effusion.pt",     
    "OPACITY": "Dr_Opacity.pt", "HEART": "Dr_Heart.pt"         
}

# ================= 3. CORE FUNCTIONS =================
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

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- HÀM HỎI CHATGPT ĐỂ LẤY NHÃN ---
def ask_gpt_for_label(api_key, image_path):
    try:
        client = OpenAI(api_key=api_key)
        base64_image = encode_image_to_base64(image_path)
        
        # Prompt ép kiểu trả về JSON
        labels_str = ", ".join([f"'{l}'" for l in ALLOWED_LABELS])
        prompt = f"""
        Bạn là bác sĩ chẩn đoán hình ảnh chuyên gia. Hãy xem phim X-quang này.
        Nhiệm vụ:
        1. Xác định các bệnh lý có trong ảnh.
        2. CHỈ ĐƯỢC CHỌN nhãn từ danh sách sau: [{labels_str}].
        3. Nếu bình thường, chọn 'Bình thường (Normal)'.
        
        Trả về kết quả định dạng JSON thuần túy (không markdown) như sau:
        {{
            "labels": ["Tên bệnh 1", "Tên bệnh 2"],
            "reasoning": "Giải thích ngắn gọn tại sao chọn (tiếng Việt)..."
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant. Output JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"labels": [], "reasoning": f"Lỗi: {str(e)}"}

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

def save_case(img_cv, patient_info="N/A"):
    img_id = datetime.now().strftime("%d%m%Y%H%M%S") 
    file_name = f"XRAY_{img_id}.jpg"
    save_path = os.path.join(IMAGES_DIR, file_name)
    try: cv2.imwrite(save_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
    except: pass
    new_record = {
        "ID": img_id, "Time": datetime.now().strftime("%d/%m/%Y %H:%M"), 
        "Result": "Đang chờ", "Image_Path": file_name, "Patient_Info": patient_info, 
        "Feedback_1": "Chưa đánh giá", "Label_1": "", "Feedback_2": "Chưa đánh giá", "Label_2": "", "GPT_Reasoning": ""
    }
    try:
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([pd.DataFrame([new_record]), df], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
    except: pass
    return img_id

def update_feedback_slot(selected_id, feedback_value, label_value, slot, gpt_reason=""):
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
        
        if gpt_reason:
             df.loc[df["ID"] == selected_id, "GPT_Reasoning"] = gpt_reason
             
        df.to_csv(LOG_FILE, index=False)
        return True
    except: return False

def get_final_label(row):
    if pd.notna(row["Label_2"]) and row["Label_2"] != "" and row["Feedback_2"] != "Chưa đánh giá": return row["Label_2"]
    elif pd.notna(row["Label_1"]) and row["Label_1"] != "" and row["Feedback_1"] != "Chưa đánh giá": return row["Label_1"]
    return ""

def export_dataset():
    if not os.path.exists(LOG_FILE): return "No data", None
    if os.path.exists(TRAIN_DATA_DIR): shutil.rmtree(TRAIN_DATA_DIR)
    os.makedirs(os.path.join(TRAIN_DATA_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DATA_DIR, "labels"), exist_ok=True)
    for en in LABEL_MAP.values(): os.makedirs(os.path.join(TRAIN_DATA_DIR, "classified", en), exist_ok=True)
    
    df = pd.read_csv(LOG_FILE)
    count = 0
    anatomy_model = MODELS.get("ANATOMY")
    
    for idx, row in df.iterrows():
        labels = get_final_label(row)
        img_src = os.path.join(IMAGES_DIR, str(row["Image_Path"]))
        if labels and os.path.exists(img_src):
            # Classify Folder
            for lbl in labels.split(";"):
                en_name = LABEL_MAP.get(lbl.strip())
                if en_name: shutil.copy(img_src, os.path.join(TRAIN_DATA_DIR, "classified", en_name, row["Image_Path"]))
            
            # YOLO Detection
            pri_lbl = labels.split(";")[0].strip()
            en_pre = LABEL_MAP.get(pri_lbl, "Unk")
            dst_img = os.path.join(TRAIN_DATA_DIR, "images", f"{en_pre}_{row['Image_Path']}")
            shutil.copy(img_src, dst_img)
            
            # Auto Label
            if anatomy_model:
                try:
                    res = anatomy_model(img_src, verbose=False)[0]
                    txt = ""
                    for box in res.boxes:
                        c, x, y, w, h = int(box.cls[0]), *box.xywhn[0].tolist()
                        txt += f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                    with open(dst_img.replace("images", "labels").replace(".jpg", ".txt"), "w") as f: f.write(txt)
                except: pass
            count += 1
            
    shutil.make_archive(TRAIN_DATA_DIR, 'zip', TRAIN_DATA_DIR)
    return f"Exported {count} files", f"{TRAIN_DATA_DIR}.zip"

# ================= 7. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("ĐIỀU KHIỂN")
    
    # -----------------------------------------------------------
    # BẮT BUỘC NHẬP KEY THỦ CÔNG (Không dùng secrets)
    # -----------------------------------------------------------
    api_key = st.text_input("🔑 OpenAI API Key:", type="password", help="Nhập Key để dùng tính năng AI Teacher")
    
    mode = st.radio("Chức năng:", ["🔍 Phân Tích & Upload", "📂 Hội Chẩn (AI Teacher)", "🛠️ Xuất Dataset"])

if mode == "🔍 Phân Tích & Upload":
    st.title("🏥 TẢI ẢNH LÊN HỆ THỐNG")
    uploaded_file = st.file_uploader("Tải ảnh X-quang:", type=["jpg", "png", "dcm"])
    
    def process_image(f):
        fname = f.name.lower()
        img_rgb, p_info = None, "Ẩn danh"
        if fname.endswith('dcm'): img_rgb, p_info = read_dicom_image(f)
        else: 
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
        
        if img_rgb is not None:
            save_case(img_rgb, p_info)
            return True, p_info
        return False, ""

    if uploaded_file and st.button("🚀 Xử lý & Lưu"):
        img_out, p_info = process_image(uploaded_file)
        if img_out: st.success("Đã lưu vào kho dữ liệu! Chuyển sang tab Hội Chẩn để gán nhãn.")

elif mode == "📂 Hội Chẩn (AI Teacher)":
    st.title("📂 HỘI CHẨN & AI GÁN NHÃN")
    
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df['ID'] = df['ID'].astype(str)
        df = df.iloc[::-1] # Mới nhất lên đầu
        
        # Chọn ca
        id_list = df["ID"].unique()
        selected_id = st.selectbox("👉 Chọn Mã hồ sơ:", id_list)
        
        if selected_id:
            record = df[df["ID"] == selected_id].iloc[0]
            col_img, col_tool = st.columns([1, 1])
            
            img_path = os.path.join(IMAGES_DIR, record["Image_Path"])
            
            with col_img:
                if os.path.exists(img_path): st.image(img_path, use_container_width=True)
            
            with col_tool:
                st.info(f"Bệnh nhân: {record['Patient_Info']}")
                
                # --- AI TEACHER BUTTON ---
                gpt_labels = []
                gpt_reason = ""
                
                # Nút bấm chỉ hoạt động khi có API Key
                if api_key:
                    if st.button("🧠 Xin ý kiến ChatGPT (Auto-Label)"):
                        with st.spinner("ChatGPT đang phân tích và chọn nhãn..."):
                            gpt_res = ask_gpt_for_label(api_key, img_path)
                            gpt_labels = gpt_res.get("labels", [])
                            gpt_reason = gpt_res.get("reasoning", "")
                            
                            if gpt_labels:
                                st.markdown(f"""
                                <div class="gpt-suggestion">
                                    <b>🤖 ChatGPT Gợi ý:</b> {', '.join(gpt_labels)}<br>
                                    <i>"{gpt_reason}"</i>
                                </div>
                                """, unsafe_allow_html=True)
                            else: st.error("ChatGPT không trả về nhãn nào hoặc lỗi (Kiểm tra lại Key).")
                else:
                    st.warning("⚠️ Vui lòng nhập OpenAI API Key ở cột bên trái để dùng tính năng này!")

                st.markdown("---")
                # --- FORM ĐÁNH GIÁ ---
                fb1 = record.get("Feedback_1", "Chưa đánh giá")
                lb1 = record.get("Label_1", "")
                
                # Tự động điền nếu có GPT
                default_labels = gpt_labels if gpt_labels else (lb1.split("; ") if lb1 else [])
                # Lọc lại để chắc chắn label nằm trong list cho phép
                valid_defaults = [l for l in default_labels if l in ALLOWED_LABELS]
                
                st.write("### 📝 Kết luận chuyên môn:")
                new_fb = st.radio("Đánh giá:", ["Chưa đánh giá", "✅ Đồng thuận", "❌ Sai (Sửa lại)"], 
                                  index=0 if fb1 == "Chưa đánh giá" else (1 if "Đồng thuận" in fb1 else 2))
                
                final_labels = st.multiselect("Bệnh lý xác định:", ALLOWED_LABELS, default=valid_defaults)
                
                if st.button("💾 LƯU KẾT QUẢ (TRAINING DATA)"):
                    lbl_str = "; ".join(final_labels)
                    # Lưu vào Slot 1 (hoặc logic 2 slot tùy bạn, ở đây làm đơn giản 1 slot chuẩn)
                    update_feedback_slot(selected_id, new_fb, lbl_str, 1, gpt_reason)
                    st.success("Đã lưu! Dữ liệu này sẽ được dùng để train.")
                    time.sleep(0.5)
                    st.rerun()

elif mode == "🛠️ Xuất Dataset":
    st.title("🛠️ XUẤT DATASET")
    if st.button("🚀 TẠO DATASET TỪ DỮ LIỆU ĐÃ GÁN NHÃN"):
        msg, zip_f = export_dataset()
        if zip_f:
            st.success(msg)
            with open(zip_f, "rb") as f: st.download_button("📥 Tải về", f, "dataset.zip")