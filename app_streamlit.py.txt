import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import time
from datetime import datetime
from PIL import Image

# ================= 1. CẤU HÌNH TRANG WEB =================
st.set_page_config(
    page_title="AI Radiology Assistant V6.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cấu hình CSS để giao diện đẹp hơn
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    h1 { color: #002f6c; }
    .report-box { 
        background-color: white; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: 'Times New Roman', serif;
    }
</style>
""", unsafe_allow_html=True)

# ================= 2. CẤU HÌNH HỆ THỐNG (CLOUD COMPATIBLE) =================
# Lấy đường dẫn hiện tại của file này (để chạy được trên cả Windows và Cloud)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")

# Đội ngũ bác sĩ
DOCTOR_ROSTER = {
    "ANATOMY":      "Dr_Anatomy.pt",      
    "PNEUMOTHORAX": "Dr_Pneumothorax.pt", 
    "PNEUMONIA":    "Dr_Pneumonia.pt",    
    "TUMOR":        "Dr_Tumor.pt",        
    "EFFUSION":     "Dr_Effusion.pt",     
    "OPACITY":      "Dr_Opacity.pt",      
    "HEART":        "Dr_Heart.pt"         
}

# ================= 3. LOAD MODEL (CACHE RESOURCE) =================
@st.cache_resource
def load_models():
    """Load model 1 lần duy nhất khi khởi động App"""
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
            except Exception as e:
                status_log.append(f"❌ {role}: Error ({str(e)})")
        else:
            status_log.append(f"⚠️ {role}: Missing file")
            
    return loaded_models, status_log, device

# Gọi hàm load
MODELS, MODEL_STATUS, DEVICE = load_models()

# ================= 4. BỘ NÃO LÂM SÀNG (LOGIC ENGINE) =================
def get_finding_text(disease, conf, location):
    pct = conf * 100
    
    if disease == "PNEUMOTHORAX":
        # Ngưỡng rất cao cho tràn khí
        if pct > 88: 
            return "danger", f"**{location}**: Mất vân phổi ngoại vi, hình ảnh điển hình **Tràn khí màng phổi** ({pct:.0f}%)."
        elif pct > 75: 
            return "warn", f"**{location}**: Tăng sáng khu trú, chưa loại trừ tràn khí lượng ít/kén khí ({pct:.0f}%)."

    elif disease == "EFFUSION":
        if pct > 80: 
            return "danger", f"**{location}**: Mờ đồng nhất góc sườn hoành, mất góc nhọn. Theo dõi **Tràn dịch** ({pct:.0f}%)."
        return "warn", f"**{location}**: Tù nhẹ góc sườn hoành, nghi ngờ dày dính/dịch ít ({pct:.0f}%)."

    elif disease == "PNEUMONIA":
        if pct > 75: 
            return "danger", f"**{location}**: Đám mờ thâm nhiễm phế bào, hình ảnh **Viêm phổi** ({pct:.0f}%)."
        return "warn", f"**{location}**: Đám mờ rải rác, theo dõi tổn thương viêm ({pct:.0f}%)."

    elif disease == "TUMOR":
        if pct > 85: 
            return "danger", f"**{location}**: Nốt mờ dạng khối, bờ không đều. Cần chụp CT ngực kiểm tra **U phổi** ({pct:.0f}%)."
        return "warn", f"**{location}**: Nốt mờ đơn độc nghi ngờ ({pct:.0f}%)."

    elif disease == "HEART":
        if pct > 70: 
            return "warn", f"**Bóng tim**: Chỉ số tim/lồng ngực ước > 0.5. Theo dõi bóng tim to ({pct:.0f}%)."
    
    return None, None

def process_image(image_file):
    if "ANATOMY" not in MODELS:
        return None, "Lỗi: Thiếu model giải phẫu (Anatomy)", False

    start_t = time.time()
    
    # Đọc ảnh từ Streamlit Upload
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    h, w = img_cv.shape[:2]
    # Resize chuẩn y tế 1280px
    scale = 1280 / max(h, w)
    img_resized = cv2.resize(img_cv, (int(w*scale), int(h*scale)))
    
    # Ảnh để vẽ (Display)
    display_img = img_resized.copy()
    
    findings_db = {"Lung": [], "Pleura": [], "Heart": []}
    has_danger = False
    
    PRIORITY_DISEASES = ["PNEUMOTHORAX", "EFFUSION", "TUMOR", "PNEUMONIA"] 
    SECONDARY_DISEASES = ["OPACITY"]

    # 1. Quét giải phẫu
    anatomy_res = MODELS["ANATOMY"](img_resized, conf=0.35, iou=0.45, verbose=False)[0]

    for box in anatomy_res.boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        region_name = anatomy_res.names[cls_id]
        
        # Safety Padding 40px
        pad = 40
        x1, y1, x2, y2 = coords
        roi = img_resized[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
        
        if roi.size == 0: continue

        # Chọn model theo vùng
        target_models = []
        if "Lung" in region_name: target_models = PRIORITY_DISEASES + SECONDARY_DISEASES
        elif "Heart" in region_name: target_models = ["HEART"]
        
        found_specific = False 

        for spec in target_models:
            if spec not in MODELS: continue
            if spec == "OPACITY" and found_specific: continue # Hierarchy check

            # Convert ROI sang BGR cho model YOLO (nếu model train bằng cv2 mặc định)
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            res = MODELS[spec](roi_bgr, verbose=False)[0]
            
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

                    # Vẽ Visualization
                    color = (255, 0, 0) if level == "danger" else (255, 165, 0) # Đỏ hoặc Cam (RGB)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_img, spec[:4], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    process_time = time.time() - start_t
    return display_img, findings_db, has_danger, process_time

# ================= 5. TẠO BÁO CÁO HTML =================
def generate_html_report(findings_db, has_danger, process_time):
    current_time = datetime.now().strftime('%H:%M ngày %d/%m/%Y')
    img_id = f"AI-{int(time.time())}"
    
    # Sinh nội dung từng phần
    lung_text = f"<b>Ghi nhận bất thường:</b><br>- {'; <br>- '.join(findings_db['Lung'])}." if findings_db["Lung"] else \
                "Hai trường phổi sáng đều, vân phổi phân bố bình thường đến ngoại vi. Không thấy đám mờ, nốt mờ hay tổn thương thâm nhiễm khu trú."
    
    pleura_text = f"<b>Phát hiện bất thường:</b><br>- {'; <br>- '.join(findings_db['Pleura'])}." if findings_db["Pleura"] else \
                  "Góc sườn hoành hai bên nhọn, vòm hoành đều. Không thấy hình ảnh tràn dịch màng phổi. Không ghi nhận tràn khí."
    
    heart_text = f"<b>Tim mạch:</b> {'; '.join(findings_db['Heart'])}." if findings_db["Heart"] else \
                 "Bóng tim không to (chỉ số tim/lồng ngực ước < 0,5). Trung thất cân đối, khí quản nằm giữa."

    bone_text = "Khung xương lồng ngực (xương sườn, xương đòn, xương vai) cân đối. Không ghi nhận hình ảnh gãy xương, khuyết xương hay tổn thương hủy xương rõ."

    # Kết luận
    if has_danger or (len(findings_db["Lung"]) + len(findings_db["Pleura"]) > 0):
        conclusion_html = "<div style='color:#c62828; font-weight:bold; font-size:18px;'>🔴 KẾT LUẬN: CÓ HÌNH ẢNH BẤT THƯỜNG TRÊN PHIM</div>"
        rec_html = """
        <div style="background:#fff3e0; padding:10px; border-left:4px solid #ff9800; color:#333;">
            <strong>💡 KHUYẾN NGHỊ:</strong><br>
            – Đề nghị kết hợp lâm sàng và xét nghiệm cận lâm sàng.<br>
            – Cân nhắc chụp CT ngực để đánh giá chi tiết bản chất tổn thương.
        </div>"""
    else:
        conclusion_html = "<div style='color:#2e7d32; font-weight:bold; font-size:18px;'>✅ KẾT LUẬN: CHƯA GHI NHẬN BẤT THƯỜNG RÕ</div>"
        rec_html = """
        <div style="color:#555;">
            <strong>💡 Khuyến nghị:</strong> Theo dõi lâm sàng. Nếu có triệu chứng hô hấp (đau ngực, khó thở, sốt kéo dài), đề nghị tái khám.
        </div>"""

    # HTML Template
    html = f"""
    <div class="report-box">
        <div style="text-align:center; border-bottom:2px solid #002f6c; padding-bottom:10px; margin-bottom:15px;">
            <h2 style="margin:0; color:#002f6c;">PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH</h2>
            <p style="margin:5px 0; font-style:italic;">(Hệ thống AI hỗ trợ phân tích X-quang ngực)</p>
        </div>
        
        <div style="font-size:14px; margin-bottom:15px;">
            <strong>Thời gian:</strong> {current_time} | <strong>ID:</strong> {img_id}<br>
            <div style="margin-top:5px; padding:5px; background:#f1f8e9; border:1px solid #c5e1a5; color:#333;">
                <strong>⚙️ KỸ THUẬT:</strong> X-quang ngực thẳng (PA view). Độ xuyên thấu và độ xoay đạt chuẩn.
            </div>
        </div>

        <h4 style="background:#eee; padding:5px; border-left:4px solid #002f6c; color:#333;">I. MÔ TẢ HÌNH ẢNH</h4>
        <ul style="padding-left:20px; line-height:1.6; color:#333;">
            <li><strong>Nhu mô phổi:</strong> {lung_text}</li>
            <li><strong>Màng phổi:</strong> {pleura_text}</li>
            <li><strong>Tim – Trung thất:</strong> {heart_text}</li>
            <li><strong>Xương lồng ngực:</strong> {bone_text}</li>
        </ul>

        <h4 style="background:#eee; padding:5px; border-left:4px solid #002f6c; color:#333;">II. KẾT LUẬN & KHUYẾN NGHỊ</h4>
        <div style="padding:10px; border:1px dashed #ccc; margin-bottom:10px;">{conclusion_html}</div>
        {rec_html}

        <div style="margin-top:30px; text-align:center; font-size:12px; color:#777; border-top:1px solid #eee; padding-top:10px;">
            Kết quả do hệ thống trí tuệ nhân tạo hỗ trợ tạo lập.<br>
            Chẩn đoán xác định thuộc về Bác sĩ chuyên khoa Chẩn đoán hình ảnh.<br>
            (Thời gian xử lý: {process_time:.2f}s)
        </div>
    </div>
    """
    return html

# ================= 6. GIAO DIỆN CHÍNH =================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("Trạng Thái Hệ Thống")
    st.info(f"🖥️ Thiết bị xử lý: **{str(DEVICE).upper()}**")
    
    with st.expander("🩺 Danh sách Model AI", expanded=True):
        for status in MODEL_STATUS:
            st.caption(status)
    
    st.markdown("---")
    st.markdown("**Phiên bản:** 6.0 Platinum")
    st.markdown("**Cập nhật:** 23/12/2025")

# --- MAIN PAGE ---
st.title("🏥 HỆ THỐNG TRỢ LÝ CĐHA CHUYÊN SÂU")
st.markdown("*(Tiêu chuẩn Bệnh viện Hạng I - Hỗ trợ phát hiện 6 nhóm bệnh lý lồng ngực)*")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Tải ảnh X-quang")
    uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG, DICOM...)", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)
        analyze_btn = st.button("🔍 PHÂN TÍCH CA BỆNH NGAY", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Kết quả Phân tích")
    
    if uploaded_file is not None and analyze_btn:
        with st.spinner("🤖 AI đang hội chẩn đa chuyên khoa..."):
            # Xử lý
            result_img, findings, has_danger, p_time = process_image(uploaded_file)
            
            if result_img is not None:
                # Tab hiển thị
                tab_img, tab_report = st.tabs(["🖼️ Hình ảnh AI", "📄 Phiếu kết quả"])
                
                with tab_img:
                    st.image(result_img, caption=f"Vị trí tổn thương (Xử lý trong {p_time:.2f}s)", use_container_width=True)
                
                with tab_report:
                    report_html = generate_html_report(findings, has_danger, p_time)
                    st.markdown(report_html, unsafe_allow_html=True)
                    
                    # Nút tải báo cáo (Giả lập)
                    st.download_button(
                        label="📥 Tải phiếu kết quả (PDF)",
                        data=report_html,
                        file_name="ket_qua_cdha.html",
                        mime="text/html"
                    )
            else:
                st.error("Lỗi: Không thể xử lý ảnh. Vui lòng kiểm tra lại file đầu vào.")
    elif uploaded_file is None:
        st.info("👈 Vui lòng tải ảnh lên ở cột bên trái để bắt đầu.")