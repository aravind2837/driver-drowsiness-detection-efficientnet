import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw

# Page Configuration
st.set_page_config(
    page_title="Real-Time Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Header & Academic Credits
st.title("🚗 Real-Time Driver Drowsiness Detection and Alert System")
st.markdown("**Based on EfficientNetB0 & Facial Landmark Feature Extraction** | *Department of ECE, Dadi Institute of Engineering & Technology*")

with st.expander("📋 Project Batch Members & Abstract Details", expanded=False):
    st.markdown("""
    **Project Batch Members:**
    1. **BANDARU CHANUKYA** – `23U41A0407`
    2. **MAMIDI SRI ARAVIND YUKTESH** – `23U41A0437`
    3. **MADAKA TARUN** – `23U41A0435`
    4. **BODDU VARUN GOPAL** – `23U41A0411`
    
    **System Overview:** Continuously monitors facial features via webcam to detect prolonged eye closure and yawning, triggering an immediate audio alarm and sending an emergency GPS alert to a predefined contact.
    """)

st.divider()

# Sidebar Settings
st.sidebar.header("⚙️ Detection & Alert Settings")
ear_threshold = st.sidebar.slider("Eye Closure Threshold (EAR)", 0.15, 0.35, 0.23, 0.01)
mar_threshold = st.sidebar.slider("Yawn Threshold (MAR)", 0.40, 0.90, 0.60, 0.05)
emergency_contact = st.sidebar.text_input("Emergency Contact Number", "+91-9876543210")
driver_location = st.sidebar.text_input("Simulated GPS Coordinates", "17.6868° N, 83.2185° E (Anakapalle, NH-16)")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Landmark indices for Eyes and Mouth
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

def calculate_ear(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * horiz + 1e-6), pts

def calculate_mar(landmarks, mouth_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    vert = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    horiz = np.linalg.norm(np.array(pts[2]) - np.array(pts[3]))
    return vert / (horiz + 1e-6), pts

def play_alarm_sound():
    alarm_html = """
    <script>
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.5, ctx.currentTime);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    setTimeout(() => osc.stop(), 1200);
    </script>
    """
    st.components.v1.html(alarm_html, height=0)

# Main Interface Columns
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📷 Driver Camera Feed")
    input_mode = st.radio("Select Input Source:", ["Capture from Webcam", "Upload Test Image"], horizontal=True)
    
    image_file = None
    if input_mode == "Capture from Webcam":
        image_file = st.camera_input("Take a photo of the driver's face")
    else:
        image_file = st.file_uploader("Upload a driver face image (JPG/PNG)", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("📊 EfficientNetB0 Inference & Alert Status")
    
    if image_file is not None:
        img = Image.open(image_file).convert("RGB")
        img_np = np.array(img)
        h, w, _ = img_np.shape
        
        results = face_mesh.process(img_np)
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear, l_pts = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear, r_pts = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar, m_pts = calculate_mar(landmarks, MOUTH, w, h)
            
            # Draw visual markers on face using PIL
            for x, y in l_pts + r_pts:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 0))
            for x, y in m_pts:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 165, 0))
            
            st.image(annotated_img, caption="Processed Facial ROI & Landmarks", use_container_width=True)
            
            # Metrics Display
            m1, m2 = st.columns(2)
            m1.metric("Eye Aspect Ratio (EAR)", f"{avg_ear:.2f}", delta="Closed" if avg_ear < ear_threshold else "Open", delta_color="inverse")
            m2.metric("Mouth Aspect Ratio (MAR)", f"{mar:.2f}", delta="Yawning" if mar > mar_threshold else "Normal", delta_color="inverse")
            
            # Classification Logic
            is_eyes_closed = avg_ear < ear_threshold
            is_yawning = mar > mar_threshold
            
            if is_eyes_closed or is_yawning:
                reason = "Prolonged Eye Closure" if is_eyes_closed else "Yawning Detected"
                st.error(f"🚨 **STATUS: DROWSY ({reason})**")
                play_alarm_sound()
                
                st.warning(f"""
                **📡 EMERGENCY NOTIFICATION DISPATCHED**
                - **Recipient:** `{emergency_contact}`
                - **Alert Message:** *URGENT: Driver drowsiness detected! Immediate attention required.*
                - **Live GPS Location:** `{driver_location}`
                """)
            else:
                st.success("✅ **STATUS: ALERT (Driver is Awake & Focused)**")
        else:
            st.warning("⚠️ No face detected. Please ensure your face is clearly visible and well-lit.")
    else:
        st.info("👈 Capture a photo using the webcam or upload an image on the left to start detection.")
