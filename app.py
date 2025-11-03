import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import pandas as pd
from datetime import datetime
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw

# ====================== SETUP ==========================
st.set_page_config(page_title="Face Attendance System", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🎓 AI-Based Face Recognition Attendance System</h1>
    <hr>
""", unsafe_allow_html=True)

# Initialize model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# CSV file for attendance
ATTENDANCE_FILE = "attendance_log.csv"
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Status", "Time", "Group Photo"]).to_csv(ATTENDANCE_FILE, index=False)

# ====================== STEP 1: UPLOAD STUDENTS ==========================
st.subheader("📁 Step 1: Upload Registered Student Photos")
student_files = st.file_uploader(
    "Upload all student images (filenames = names or roll numbers)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

known_faces, known_names = [], []

if student_files:
    for file in student_files:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        img = cv2.imread(temp.name)
        temp.close()
        faces = app.get(img)
        if faces:
            known_faces.append(faces[0].normed_embedding)
            name = os.path.splitext(file.name)[0]
            known_names.append(name)
    st.success(f"✅ {len(known_faces)} student profiles registered successfully!")

# ====================== STEP 2: UPLOAD GROUP PHOTO ==========================
st.subheader("🧑‍🏫 Step 2: Upload Group Photo for Attendance")
group_photo = st.file_uploader("Upload the group classroom photo", type=["jpg", "jpeg", "png"])

if group_photo and known_faces:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(group_photo.read())
    group_img = cv2.imread(temp.name)
    temp.close()

    # Detect faces
    group_faces = app.get(group_img)
    draw = Image.fromarray(cv2.cvtColor(group_img, cv2.COLOR_BGR2RGB))
    draw_img = ImageDraw.Draw(draw)

    recognized = []
    for face in group_faces:
        emb = face.normed_embedding
        sims = np.dot(known_faces, emb)
        idx = np.argmax(sims)
        name = known_names[idx] if sims[idx] > 0.35 else "Unknown"

        box = face.bbox.astype(int)
        color = "green" if name != "Unknown" else "red"
        draw_img.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
        draw_img.text((box[0], box[1] - 10), name, fill=color)
        recognized.append(name)

    # Display group image with results
    st.image(draw, caption="Detected Faces", use_container_width=True)

    # Attendance logic
    present_students = set([n for n in recognized if n != "Unknown"])
    all_students = set(known_names)
    absent_students = all_students - present_students

    st.markdown("### 🧾 AI-Detected Attendance Summary")

    # ================== MANUAL CHECKBOX SECTION ==================
    st.info("⚙️ If the AI missed someone, mark them manually below:")
    manual_present = []
    manual_absent = []

    for student in all_students:
        if student in present_students:
            checked = st.checkbox(f"{student} (Detected Present)", value=True)
        else:
            checked = st.checkbox(f"{student} (Detected Absent)", value=False)
        
        if checked:
            manual_present.append(student)
        else:
            manual_absent.append(student)

    # Final attendance after manual corrections
    present_students = set(manual_present)
    absent_students = set(manual_absent)

    # Display in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Present Students")
        st.write(list(present_students) if present_students else ["None"])
    with col2:
        st.error("❌ Absent Students")
        st.write(list(absent_students) if absent_students else ["None"])

    # Save attendance
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance_data = pd.read_csv(ATTENDANCE_FILE)

    new_records = []
    for name in all_students:
        status = "Present" if name in present_students else "Absent"
        new_records.append({
            "Name": name,
            "Status": status,
            "Time": current_time,
            "Group Photo": group_photo.name
        })

    df_new = pd.DataFrame(new_records)
    attendance_data = pd.concat([attendance_data, df_new], ignore_index=True)
    attendance_data.to_csv(ATTENDANCE_FILE, index=False)

    st.success("✅ Attendance Recorded Successfully!")

    # Download button for CSV
    csv = df_new.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download This Session Attendance",
        data=csv,
        file_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

elif group_photo and not known_faces:
    st.warning("⚠️ Please upload student photos first before uploading group image!")
