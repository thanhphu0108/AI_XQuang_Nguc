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
import ast 
import google.generativeai as genai
from supabase import create_client, Client
import requests
from io import BytesIO

# ================= 1. CẤU HÌNH & CSS =================
st.set_page_config(page_title="AI Hospital (V31.3 - Vietnamese)", page_icon="🇻🇳", layout="wide")

st.markdown("""
<style>
    .main { background-color: #e9ecef; }
    .a4-paper {
        background-color: white; width: 100%; max-width: 800px; margin: 0 auto; padding: 40px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1); font-family: 'Times New Roman', serif; color: #000; border: 1px solid #ccc;
    }
    .hospital-header { text-align: center; border-bottom: 2px solid #000; padding-bottom: 15px; margin-bottom: 20px; }
    .hospital-header h1 { margin: 0; font-size: 22px; text-transform: uppercase; font-weight: bold; color: #002f6c; }
    .info-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
    .info-table td { padding: 5px; border-bottom: 1px dotted #999; vertical-align: bottom; }
    .section-title { background-color: #f0f2f5; font-weight: bold; padding: 8px; margin-top: 20px; border-left: 4px solid #002f6c; text-transform: uppercase; font-size: 14px; }
    .conclusion-box { border: 2px solid #333; padding: 15px; margin-top: 20px; text-align: center; font-weight: bold; }
    .stButton>button { width: 100%; font-weight: bold; height: 45px; }
    .step-badge { background-color: #002f6c; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; display: inline-block; margin-bottom: 10px; }
    
    /* Highlight cho Radio Button */
    div[role="radiogroup"] > label > div:first-child { background-color: #e3f2fd; }
</style>
""", unsafe_allow_html=True)

# --- TỪ ĐIỂN VIỆT HÓA ---
LABEL_MAP = {
    "Normal": "Bình thường",
    "Cardiomegaly": "Bóng tim to (Cardiomegaly)",
    "Pneumonia": "Viêm phổi (Pneumonia)",
    "Effusion": "Tràn dịch (Effusion)",
    "Pneumothorax": "Tràn khí (Pneumothorax)",
    "Nodule_Mass": "Nốt/Khối mờ (Nodule/Mass)",
    "Fibrosis_TB": "Xơ hóa/Lao (Fibrosis/TB)",
    "Fracture": "Gãy xương (Fracture)",
    "Pleural_Thickening": "Dày dính màng phổi",
    "Other": "Khác (Other)"
}
# Tạo list hiển thị cho Multiselect
VN_LABELS_LIST = list(LABEL_MAP.values())

TECHNICAL_OPTS = [
    "✅ Phim đạt chuẩn", 
    "⚠️ Chụp tại giường (AP View)", 
    "⚠️ Hít vào không đủ sâu", 
    "⚠️ Bệnh nhân xoay lệch", 
    "⚠️ Tia quá cứng (Phim đen)", 
    "⚠️ Tia quá mềm (Phim trắng)", 
    "⚠️ Dị vật/Áo chưa bỏ"
]

FEEDBACK_OPTS = [
    "Chưa đánh giá",
    "✅ Đồng thuận (AI Đúng)",
    "⚠️ Dương tính giả (AI Báo thừa)",
    "⚠️ Âm tính giả (AI Bỏ sót)",
    "❌ Sai hoàn toàn"
]

# --- KẾT NỐI SUPABASE ---
@st.cache_resource
def init_supabase():
    if "supabase" not in st.secrets:
        st.error("⚠️ Chưa cấu hình [supabase] trong secrets.toml.")
        return None
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        if not url or "http" not in url: return None
        return create_client(url, key)
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
    if supabase is None: return None
    try:
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
        bucket = "xray_images"
        supabase.storage.from_(bucket).upload(filename, buffer.tobytes(), {"content-type": "image/jpeg", "upsert": "true"})
        return supabase.storage.from_(bucket).get_public_url(filename)
    except:
        try: return supabase.storage.from_("xray_images").get_public_url(filename)
        except: return None

def save_log(data):
    if supabase is None: return False
    try:
        supabase.table("logs").upsert(data).execute()
        return True
    except Exception as e:
        st.error(f"❌ Lỗi lưu Data: {e}")
        return False

def get_logs():
    if supabase is None: return pd.DataFrame()
    try:
        response = supabase.table("logs").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(response.data)
    except: return pd.DataFrame()

# --- GEMINI ---
# --- GEMINI (V32.0 - AUTO SMART SELECT) ---
def ask_gemini(api_key, image, context="", note="", guide="", tags=[]):
    if not api_key: return {}
    
    # Cấu hình API
    try: genai.configure(api_key=api_key)
    except: return {"labels": [], "reasoning": "Sai API Key"}

    # --- DANH SÁCH MỤC TIÊU (Ưu tiên từ Xịn -> Nhanh) ---
    # Google hiện tại chưa có 2.5 hay 3.5, mới nhất là 2.0 Flash Exp
    model_priority = [
        "gemini-2.0-flash-exp",  # Mới nhất, xịn nhất (Thử trước)
        "gemini-1.5-pro",        # Phân tích sâu (Thử nhì)
        "gemini-1.5-flash"       # Nhanh, ổn định (Chống trượt)
    ]

    labels_str = ", ".join(ALLOWED_LABELS) # Danh sách bệnh tiếng Anh (để AI dễ hiểu)
    tech_note = ", ".join(tags) if tags else "Chuẩn."
    
    # Prompt tối ưu hóa
    prompt = f"""
    Role: Senior Radiologist.
    INPUTS: 
    - Clinical Context: "{context}"
    - Expert Note: "{note}"
    - Technical QA: "{tech_note}"
    - Guidance: "{guide}"
    
    TASK: Analyze Chest X-ray image. 
    1. Check technical quality.
    2. Detect abnormalities from this list: [{labels_str}]. If none, return 'Normal'.
    3. Provide reasoning in Vietnamese (Tiếng Việt).
    
    OUTPUT JSON FORMAT: {{ "labels": ["..."], "reasoning": "..." }}
    """

    # --- VÒNG LẶP AUTO SCAN ---
    last_error = ""
    for model_name in model_priority:
        try:
            # Tạo model
            model = genai.GenerativeModel(model_name)
            
            # Gửi ảnh và prompt
            response = model.generate_content(
                [prompt, image], 
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Nếu chạy thành công đến đây thì trả về kết quả ngay
            result = json.loads(response.text)
            result["used_model"] = model_name # Ghi dấu lại xem nó dùng model nào
            return result
            
        except Exception as e:
            # Nếu lỗi, bỏ qua, thử model tiếp theo trong danh sách
            last_error = str(e)
            continue 

    # Nếu thử hết cả 3 model mà vẫn lỗi
    return {"labels": [], "reasoning": f"Hệ thống bận hoặc lỗi API (Chi tiết: {last_error})", "used_model": "Failed"}
# --- HTML REPORT ---
def generate_html_report(findings_input, has_danger, patient_info, img_id, gemini_text=""):
    current_time = datetime.now().strftime('%H:%M ngày %d/%m/%Y')
    findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    
    if isinstance(findings_input, dict): findings_db = findings_input
    elif isinstance(findings_input, str):
        try: findings_db = ast.literal_eval(findings_input)
        except: pass
            
    def make_list(items, default):
        if not items: return f'<ul style="margin:5px 0 10px 20px;"><li>{default}</li></ul>'
        return f'<ul style="margin:5px 0 10px 20px; color:#c62828;"><li><b>PHÁT HIỆN:</b> {"; ".join(items)}</li></ul>'

    lung_html = make_list(findings_db.get("Lung"), "Hai trường phổi sáng đều. Không đám mờ.")
    pleura_html = make_list(findings_db.get("Pleura"), "Góc sườn hoành nhọn. Không tràn dịch.")
    heart_html = make_list(findings_db.get("Heart"), "Bóng tim không to. Trung thất cân đối.")
    bone_html = '<ul style="margin:5px 0 10px 20px;"><li>Khung xương lồng ngực cân đối.</li></ul>'
    
    is_abnormal = has_danger or (len(findings_db.get("Lung", [])) + len(findings_db.get("Pleura", [])) + len(findings_db.get("Heart", [])) > 0)
    
    if is_abnormal:
        conclusion_html = """<div style='color:#c62828; font-size:18px;'>🔴 KẾT LUẬN: CÓ HÌNH ẢNH BẤT THƯỜNG</div>"""
    else:
        conclusion_html = """<div style='color:#2e7d32; font-size:18px;'>✅ KẾT LUẬN: CHƯA GHI NHẬN BẤT THƯỜNG</div>"""

    gemini_block = ""
    if gemini_text:
        gemini_block = f"""<div style="margin-top:15px; padding:10px; background:#fffde7; border:1px dashed orange; font-style:italic;"><b>🤖 Gemini:</b> {gemini_text}</div>"""

    html = f"""
    <div class="a4-paper">
        <div class="hospital-header">
            <h1>PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH</h1>
            <p>Hệ thống AI Hỗ trợ Chẩn đoán X-quang Ngực</p>
        </div>
        <table class="info-table">
            <tr><td style="width:60%;"><strong>Họ tên:</strong> {patient_info}</td><td style="text-align:right;"><strong>Mã HS:</strong> {img_id}</td></tr>
            <tr><td><strong>Chỉ định:</strong> X-quang ngực thẳng (PA)</td><td style="text-align:right;"><strong>Ngày:</strong> {current_time}</td></tr>
        </table>
        <div class="section-title">I. MÔ TẢ HÌNH ẢNH</div>
        <strong>1. Nhu mô phổi:</strong>{lung_html}
        <strong>2. Màng phổi:</strong>{pleura_html}
        <strong>3. Tim - Trung thất:</strong>{heart_html}
        <strong>4. Hệ xương:</strong>{bone_html}
        <div class="section-title">II. KẾT LUẬN</div>
        <div class="conclusion-box">{conclusion_html}{gemini_block}</div>
        <div style="text-align:center; font-style:italic; font-size:12px; margin-top:50px;">(Chữ ký bác sĩ chuyên khoa)<br><br><br><b>BS. Chẩn Đoán Hình Ảnh</b></div>
    </div>
    """
    return html

# --- PROCESS & SAVE ---
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
        save_log({
            "id": img_id, 
            "created_at": datetime.now().isoformat(), 
            "image_url": img_url, 
            "result": "BẤT THƯỜNG" if has_danger else "BÌNH THƯỜNG", 
            "details": str(findings_db), 
            "patient_info": patient_info
        })

    return display_img, findings_db, has_danger, img_id, Image.fromarray(img_resized)

# ================= 3. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("ĐIỀU KHIỂN")
    api_key = st.text_input("🔑 Gemini API Key:", value=st.secrets.get("GEMINI_API_KEY", ""), type="password")
    mode = st.radio("Menu:", ["🔍 Phân Tích & In Phiếu", "📂 Hội Chẩn (Cloud)", "🛠️ Xuất Dataset"])

if mode == "🔍 Phân Tích & In Phiếu":
    st.title("🏥 TRỢ LÝ CHẨN ĐOÁN (A4)")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        uploaded_file = st.file_uploader("Chọn ảnh X-quang:", type=["jpg", "png", "jpeg", "dcm"])
        if uploaded_file and st.button("🚀 PHÂN TÍCH", type="primary"):
            with col2:
                with st.spinner("Đang xử lý..."):
                    img_out, findings, danger, img_id, pil_img = process_and_save(uploaded_file)
                    
                    if img_out is not None:
                        t1, t2 = st.tabs(["🖼️ Ảnh AI", "📄 Phiếu Kết Quả"])
                        with t1: st.image(img_out, caption=f"ID: {img_id}", use_container_width=True)
                        with t2:
                            gemini_txt = ""
                            if api_key:
                                res = ask_gemini(api_key, pil_img)
                                gemini_txt = res.get("reasoning", "")
                                if gemini_txt and supabase: save_log({"id": img_id, "ai_reasoning": gemini_txt})
                            html = generate_html_report(findings, danger, "Nguyễn Văn A", img_id, gemini_txt)
                            st.markdown(html, unsafe_allow_html=True)
                        if supabase: st.success("✅ Đã lưu vào Cloud!")
                    else: st.error("Lỗi xử lý")

elif mode == "📂 Hội Chẩn (Cloud)":
    st.title("📂 HỘI CHẨN & GÁN NHÃN")
    if supabase is None:
        st.error("⛔ Chưa kết nối Cloud.")
    else:
        df = get_logs()
        if not df.empty:
            df = df.fillna("")
            id_list = df['id'].tolist()
            selected_id = st.selectbox("👉 Chọn Mã Hồ Sơ:", id_list)
            
            if selected_id:
                record = df[df["id"] == selected_id].iloc[0]
                
                # --- LOAD ẢNH TRƯỚC (QUAN TRỌNG ĐỂ FIX LỖI GEMINI) ---
                pil_img = None
                if record.get('image_url'):
                    try:
                        response = requests.get(record['image_url'], timeout=5)
                        pil_img = Image.open(BytesIO(response.content))
                    except: pass
                
                t_work, t_paper = st.tabs(["👨‍⚕️ Bàn Làm Việc", "📄 Xem Phiếu A4"])
                
                with t_work:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if record.get('image_url'): st.image(record['image_url'], use_container_width=True)
                    
                    with c2:
                        st.info(f"BN: {record.get('patient_info')} | AI: {record.get('result')}")
                        if record.get('ai_reasoning'):
                            with st.expander("🤖 Đọc kết quả Gemini cũ"): st.write(record.get('ai_reasoning'))

                        # FORM NHẬP LIỆU
                        st.markdown("#### 📝 Thông tin Lâm sàng")
                        ctx = st.text_area("Bệnh cảnh:", value=record.get("clinical_context") or "", height=68)
                        note = st.text_area("Ý kiến chuyên gia:", value=record.get("expert_note") or "", height=68)
                        guide = st.text_area("Prompt cho AI:", value=record.get("prompt_guidance") or "", height=68)
                        
                        tags_str = record.get("technical_tags") or ""
                        def_tags = [t.strip() for t in tags_str.split(";")] if tags_str else []
                        tags = st.multiselect("Đánh giá Kỹ thuật:", TECHNICAL_OPTS, default=def_tags)
                        
                        # --- NÚT HỎI GEMINI (ĐÃ FIX) ---
                        if st.button("🧠 Hỏi lại Gemini"):
                            if not api_key: st.error("⚠️ Thiếu API Key!")
                            elif not pil_img: st.error("⚠️ Không tải được ảnh từ Cloud!")
                            else:
                                with st.spinner("Gemini đang đọc lại..."):
                                    res = ask_gemini(api_key, pil_img, ctx, note, guide, tags)
                                    txt = res.get("reasoning", "")
                                    if txt:
                                        save_log({"id": selected_id, "ai_reasoning": txt})
                                        st.success("Đã cập nhật Gemini mới!")
                                        time.sleep(1)
                                        st.rerun()

                        st.markdown("---")
                        # --- PHẦN GÁN NHÃN (LABELING) ---
                        st.markdown("#### 🏷️ Gán nhãn & Kết luận")
                        
                        fb1 = str(record.get("feedback_1") or "Chưa đánh giá")
                        
                        if fb1 == "Chưa đánh giá":
                            st.markdown('<div class="step-badge">VÒNG 1: SƠ BỘ</div>', unsafe_allow_html=True)
                            
                            # RADIO BUTTON CÓ DƯƠNG TÍNH GIẢ/ÂM TÍNH GIẢ
                            new_fb = st.radio("Đánh giá AI:", FEEDBACK_OPTS, index=0)
                            
                            lbl_str = record.get("label_1") or ""
                            def_lbls = [l.strip() for l in lbl_str.split(";")] if lbl_str else []
                            # MULTISELECT CÓ TIẾNG VIỆT
                            new_lbls = st.multiselect("Chốt bệnh lý:", VN_LABELS_LIST, default=def_lbls)
                            
                            saved_rating = record.get("prompt_rating", "Khá")
                            rating_opts = ["Tệ", "TB", "Khá", "Tốt", "Xuất sắc"]
                            val_idx = rating_opts.index(saved_rating) if saved_rating in rating_opts else 2
                            rating = st.select_slider("Chất lượng Prompt:", options=rating_opts, value=rating_opts[val_idx])

                            if st.button("💾 LƯU VÒNG 1"):
                                save_log({
                                    "id": selected_id, 
                                    "clinical_context": ctx, "expert_note": note, "prompt_guidance": guide,
                                    "technical_tags": "; ".join(tags), 
                                    "feedback_1": new_fb, "label_1": "; ".join(new_lbls),
                                    "prompt_rating": rating
                                })
                                st.success("Đã lưu Vòng 1!")
                                time.sleep(0.5)
                                st.rerun()
                        else:
                            st.info(f"✅ Vòng 1: {fb1}")
                            st.markdown('<div class="step-badge" style="background:#c62828">VÒNG 2: FINAL AUDIT</div>', unsafe_allow_html=True)
                            
                            new_fb2 = st.radio("Đánh giá cuối:", FEEDBACK_OPTS, index=0, key="fb2")
                            
                            lbl_str2 = record.get("label_2") or ""
                            def_lbls2 = [l.strip() for l in lbl_str2.split(";")] if lbl_str2 else []
                            new_lbls2 = st.multiselect("CHỐT BỆNH ÁN:", VN_LABELS_LIST, default=def_lbls2, key="lbl2")
                            
                            if st.button("💾 LƯU HỒ SƠ"):
                                save_log({"id": selected_id, "feedback_2": new_fb2, "label_2": "; ".join(new_lbls2)})
                                st.success("Đã chốt hồ sơ!")
                                time.sleep(0.5)
                                st.rerun()

                with t_paper:
                    raw_details = record.get("details", "")
                    is_danger = record.get("result") == "BẤT THƯỜNG"
                    p_info = record.get("patient_info", "N/A")
                    gemini_old = record.get("ai_reasoning", "")
                    html_repro = generate_html_report(raw_details, is_danger, p_info, selected_id, gemini_old)
                    st.markdown(html_repro, unsafe_allow_html=True)
        else:
            st.warning("📭 Chưa có dữ liệu trên Cloud.")

elif mode == "🛠️ Xuất Dataset":
    st.title("🛠️ DATASET")
    if supabase:
        df = get_logs()
        if not df.empty:
            st.dataframe(df)
            st.download_button("📥 Tải File CSV", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
        else:
            st.warning("Chưa có dữ liệu.")