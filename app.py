# app.py
"""
Smart College Face Attendance — Multi-Capture Edition (Stable)
- Uses InsightFace buffalo_l exactly like your working version
- Multi-capture; detections merged across captures
- Flip-fallback + flipped-encodings for robust matching
- Cached embeddings; manual override + quick enroll unknown
- Records tab + full-day report

Run: streamlit run app.py
"""

import os
import time
import tempfile
from datetime import datetime, date
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart College Face Attendance", layout="wide")
ROOT_REG = "registered_faces"
ROOT_SAVE = "attendance_records"
os.makedirs(ROOT_REG, exist_ok=True)
os.makedirs(ROOT_SAVE, exist_ok=True)

MODEL_NAME = "buffalo_l"
CTX_ID = -1                   # CPU; set 0 if you have GPU
DET_SIZE = (640, 640)
INTERNAL_THRESHOLD = 0.36     # tuned for buffalo_l

# ---------------- STYLE ----------------
st.markdown(
    """
    <style>
    .title {color:#0b5ed7; text-align:center; font-weight:700; font-size:48px;}
    .subtitle {color:#6c757d; text-align:center; margin-top:0; margin-bottom:12px;}
    .card {background: rgba(255,255,255,0.02); padding:12px; border-radius:10px;}
    .meta {background: rgba(11,94,215,0.05); padding:8px; border-radius:6px;}
    .small {color: #8b949e; font-size:12px;}
    table.dataframe td, table.dataframe th {border: none;}
    .present {color: #0f9d58; font-weight:600;}
    .absent  {color: #d93025; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'> Digital ICFAI Attendance</div>", unsafe_allow_html=True)
st.write("")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model(name=MODEL_NAME, ctx=CTX_ID, det_size=DET_SIZE):
    fa = FaceAnalysis(name=name)
    fa.prepare(ctx_id=ctx, det_size=det_size)
    # warm-up once to avoid first-call miss/lag (important for stability)
    _ = fa.get(np.zeros((320, 320, 3), dtype=np.uint8))
    return fa

model = load_model()

# ---------------- REGISTERED EMBEDDINGS ----------------
@st.cache_data(ttl=3600)
def load_registered(augment_flip: bool = True) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Builds two banks:
      - reg_embs: embeddings of registered images
      - reg_embs_flip: embeddings from horizontally flipped registered images
    So we can match both normal and profile-ish faces reliably.
    """
    names, embs, embs_flip = [], [], []
    for fn in sorted(os.listdir(ROOT_REG)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(ROOT_REG, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = model.get(img)
        if not faces:
            continue
        emb = faces[0].normed_embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        names.append(os.path.splitext(fn)[0])
        embs.append(emb)

        if augment_flip:
            flip_img = cv2.flip(img, 1)
            faces_f = model.get(flip_img)
            if faces_f:
                ef = faces_f[0].normed_embedding.astype(np.float32)
                ef = ef / (np.linalg.norm(ef) + 1e-10)
                embs_flip.append(ef)
            else:
                embs_flip.append(emb)

    embs_mat = np.vstack(embs) if embs else np.zeros((0, 512), dtype=np.float32)
    embs_flip_mat = np.vstack(embs_flip) if embs_flip else None
    return names, embs_mat, embs_flip_mat

names, reg_embs, reg_embs_flip = load_registered(augment_flip=True)

# ---------------- HELPERS ----------------
def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)

def draw_box_label(pil_img: Image.Image, bbox, label: str, matched: bool):
    draw = ImageDraw.Draw(pil_img)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (18, 184, 134) if matched else (220, 53, 69)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)  # thin box
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    label_text = label or "Unknown"
    try:
        tb = draw.textbbox((x1, y1 - 18), label_text, font=font)
        tw = tb[2] - tb[0]; th = tb[3] - tb[1]
    except Exception:
        tw, th = draw.textsize(label_text, font=font)
    ly = y1 - th - 6 if (y1 - th - 6) > 0 else y2 + 6
    draw.rectangle([(x1 - 2, ly - 2), (x1 + tw + 6, ly + th + 2)], fill=(245,245,245))
    draw.text((x1 + 2, ly), label_text, fill=(0,0,0), font=font)

def match_face_vectorized_safe(face, threshold: float = INTERNAL_THRESHOLD) -> Tuple[str, float, bool]:
    """
    Stable matcher:
    - Use ORIGINAL face.normed_embedding from FaceAnalysis on the full frame
    - Compare against reg_embs (and reg_embs_flip bank) for robust match
    - No tiny re-detections; avoids intermittent misses
    """
    if reg_embs.shape[0] == 0:
        return None, 0.0, False

    # direct
    emb = normalize(face.normed_embedding.astype(np.float32))
    sims = reg_embs @ emb
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_name = names[best_idx]

    if best_score >= threshold:
        return best_name, best_score, True

    # try flipped-encodings bank
    if reg_embs_flip is not None:
        sims_f = reg_embs_flip @ emb
        best_idx_f = int(np.argmax(sims_f))
        best_score_f = float(sims_f[best_idx_f])
        if best_score_f >= threshold:
            return names[best_idx_f], best_score_f, True
        # track better of two even if below threshold
        if best_score_f > best_score:
            best_name, best_score = names[best_idx_f], best_score_f

    return best_name, best_score, False

def ensure_dirs():
    os.makedirs(ROOT_REG, exist_ok=True)
    os.makedirs(ROOT_SAVE, exist_ok=True)

def save_enrolled_image(pil_crop: Image.Image, label: str) -> str:
    ensure_dirs()
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in label.strip())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{safe}_{ts}.jpg"
    path = os.path.join(ROOT_REG, fname)
    pil_crop.convert("RGB").save(path, format="JPEG", quality=90)
    return path

def format_time_12h(dt: datetime) -> str:
    return dt.strftime("%I-%M%p")

def format_time_display(dt: datetime) -> str:
    return dt.strftime("%I:%M %p")

def build_filename(dt: datetime, period_label: str, subject: str, faculty: str) -> str:
    date_part = dt.strftime("%Y-%m-%d")
    time_part = format_time_12h(dt)
    subj = (subject or "subject").replace(" ", "_")
    fac = (faculty or "faculty").replace(" ", "_")
    period_s = period_label.replace(" ", "_")
    return f"{date_part}_{time_part}_{period_s}_{subj}_{fac}.csv"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Session details")
    subject = st.text_input("Subject name")
    faculty = st.text_input("FACULTY NAME")
    period = st.selectbox("Period", [f"Period {i}" for i in range(1,9)])
    mode = st.radio("Photo source", ["Webcam (capture)", "Upload photos"])
    num_clicks = st.number_input("Number of captures (per lecture)", min_value=1, max_value=10, value=3, step=1)
    auto_time = datetime.now()
    time_default = format_time_display(auto_time)
    time_input = st.text_input("Time (HH:MM AM/PM) — override if needed", value=time_default)
    st.markdown("---")
    st.write(f"Registered students: **{len(names)}**")
    if st.button("Reload registry"):
        load_registered.clear()
        st.rerun()

if len(names) == 0:
    st.warning("No registered faces found in `registered_faces/`. Add headshots and click Reload.")
    st.stop()

# ---------------- TABS ----------------
tab_take, tab_records = st.tabs(["📸 Take Attendance", "📄 Attendance Records (Today)"])

# ---------------- TAB: Take Attendance ----------------
with tab_take:
    st.subheader("Capture / Upload group photo(s)")

    captured_images = []

    if mode == "Webcam (capture)":
        st.info(f"Use the camera widgets below to capture up to {num_clicks} photos. After capturing, click 'Process captures' to detect faces across all photos.")
        cam_cols = st.columns(2)
        cam_widgets = []
        for i in range(num_clicks):
            col = cam_cols[i % 2]
            with col:
                cam = st.camera_input(f"Take photo #{i+1}")
                cam_widgets.append(cam)
        for cam in cam_widgets:
            if cam:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(cam.read())
                    tmp.flush()
                    img = cv2.imread(tmp.name)
                    if img is not None:
                        captured_images.append(img)
    else:
        st.info(f"Upload up to {num_clicks} group photos and then click 'Process captures'")
        uploaded = st.file_uploader(
            "Upload group photos (multiple)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if uploaded:
            for up in uploaded[:num_clicks]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(up.read())
                    tmp.flush()
                    img = cv2.imread(tmp.name)
                    if img is not None:
                        captured_images.append(img)

    # Process captures
    if st.button("Process captures"):
        if not captured_images:
            st.warning("No photos captured or uploaded. Please capture/upload photos first.")
        else:
            t0_all = time.time()
            all_detected = set()
            unknown_previews = []
            last_annotated = None
            process_times = []

            for idx_img, group_bgr in enumerate(captured_images, start=1):
                t0 = time.time()
                H, W = group_bgr.shape[:2]
                maxdim = 900
                if max(H, W) > maxdim:
                    scale = maxdim / max(H, W)
                    group_bgr = cv2.resize(group_bgr, (int(W*scale), int(H*scale)))

                faces = model.get(group_bgr)
                pil_img = pil_from_bgr(group_bgr)

                for face in faces:
                    # Stable matching: direct + flipped-encodings bank
                    best_name, best_score, matched = match_face_vectorized_safe(face, threshold=INTERNAL_THRESHOLD)

                    # Optional extra rescue (your original small flip-of-crop), keep it because it worked for you
                    if not matched:
                        try:
                            x1, y1, x2, y2 = [int(v) for v in face.bbox]
                            crop_bgr = group_bgr[max(0, y1):y2, max(0, x1):x2]
                            if crop_bgr.size != 0:
                                flip = cv2.flip(crop_bgr, 1)
                                faces_f = model.get(flip)
                                if faces_f:
                                    emb_f = normalize(faces_f[0].normed_embedding.astype(np.float32))
                                    sims_f = reg_embs @ emb_f
                                    bi = int(np.argmax(sims_f))
                                    bs = float(sims_f[bi])
                                    if bs >= INTERNAL_THRESHOLD:
                                        best_name, best_score, matched = names[bi], bs, True
                        except Exception:
                            pass

                    if matched and best_name is not None:
                        all_detected.add(best_name)
                    else:
                        try:
                            x1, y1, x2, y2 = [int(v) for v in face.bbox]
                            crop = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))
                            unknown_previews.append({"crop": crop, "score": float(best_score)})
                        except Exception:
                            pass

                    draw_box_label(pil_img, face.bbox, best_name if matched else "Unknown", matched)

                last_annotated = pil_img
                process_times.append(time.time() - t0)

            st.success(f"Processed {len(captured_images)} photo(s) in {time.time() - t0_all:.2f} sec (avg {np.mean(process_times):.2f}s per image)")

            if last_annotated is not None:
                st.image(last_annotated, caption="Annotated last processed photo (others processed too)", use_container_width=True)

            # Build attendance baseline (union across captures)
            attendance_rows = []
            for nm in names:
                attendance_rows.append({"Student Name": nm.replace("_", " "), "Status": "Present" if nm in all_detected else "Absent"})

            st.markdown("<div class='meta'>", unsafe_allow_html=True)
            st.write(f"**Subject:** {subject or '—'}  |  **Faculty:** {faculty or '—'}  |  **Period:** {period}  |  **Date:** {date.today().isoformat()}  |  **Time:** {time_input}")
            st.markdown("</div>")

            # Manual override UI
            st.subheader("Attendance — verify & edit")
            edited = []
            present_count = 0
            for i, row in enumerate(attendance_rows):
                checked = st.checkbox(row["Student Name"], value=(row["Status"] == "Present"), key=f"chk_{i}")
                status = "Present" if checked else "Absent"
                if status == "Present":
                    present_count += 1
                edited.append({"Student Name": row["Student Name"], "Status": status})
            df_display = pd.DataFrame(edited)
            st.markdown(f"**Summary:** ✅ Present: **{present_count}**  |  ❌ Absent: **{len(names) - present_count}**")
            st.table(df_display[["Student Name", "Status"]])

            # Save attendance (structured)
            if st.button("Save attendance for this period"):
                try:
                    dt_display = datetime.strptime(time_input.strip(), "%I:%M %p")
                    now_dt = datetime.combine(date.today(), dt_display.time())
                except Exception:
                    now_dt = datetime.now()
                fname = build_filename(now_dt, period, subject, faculty)
                today_str = date.today().isoformat()
                folder = os.path.join(ROOT_SAVE, today_str, period.replace(" ", "_"))
                os.makedirs(folder, exist_ok=True)
                fpath = os.path.join(folder, fname)
                df_save = df_display.copy()
                df_save["Subject"] = subject
                df_save["Faculty"] = faculty
                df_save["Period"] = period
                df_save["Date"] = today_str
                df_save["Time"] = format_time_display(now_dt)
                df_save["SavedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df_save.to_csv(fpath, index=False)
                st.success(f"Saved → {os.path.relpath(fpath)}")
                with open(fpath, "rb") as fh:
                    st.download_button("Download CSV", fh.read(), file_name=fname, mime="text/csv")

            # Enroll unknown faces
            if unknown_previews:
                st.markdown("---")
                st.subheader("Enroll Unknown Faces (quick)")
                enroll_cols = st.columns(3)
                enroll_inputs = []
                for idx, item in enumerate(unknown_previews[:12]):
                    col = enroll_cols[idx % 3]
                    with col:
                        st.image(item["crop"], width=160)
                        st.caption(f"AI score: {item['score']:.2f}")
                        name_input = st.text_input(f"Label for face #{idx+1}", key=f"enroll_name_{idx}")
                        enroll_inputs.append((idx, item["crop"], name_input))

                if st.button("Enroll labeled faces into registry"):
                    saved = 0
                    for idx, crop, label in enroll_inputs:
                        if label and label.strip():
                            save_enrolled_image(crop, label.strip())
                            saved += 1
                    if saved > 0:
                        st.success(f"Enrolled {saved} faces. Reloading registry...")
                        load_registered.clear()
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.warning("No labels entered. Please type a name for each face you want to enroll.")

            st.caption(f"Unique detected students across captures: {len(all_detected)}")
    else:
        st.info("Capture (webcam) or upload a photo to start attendance. After capturing/uploading the desired photos, click 'Process captures' to detect across all photos.")

# ---------------- TAB: Attendance Records (Today) ----------------
with tab_records:
    st.subheader("📄 Attendance Records — Today (grouped by period)")
    today_str = date.today().isoformat()
    today_folder = os.path.join(ROOT_SAVE, today_str)
    if not os.path.isdir(today_folder):
        st.info("No attendance saved for today yet.")
    else:
        period_folders = sorted([p for p in os.listdir(today_folder) if os.path.isdir(os.path.join(today_folder, p))])
        if not period_folders:
            st.info("No attendance saved for today yet.")
        else:
            sel_period = st.selectbox("Choose period (or All)", ["All"] + period_folders)
            files_to_show = []
            if sel_period == "All":
                for p in period_folders:
                    pf = os.path.join(today_folder, p)
                    for f in sorted(os.listdir(pf), reverse=True):
                        if f.lower().endswith(".csv"):
                            files_to_show.append(os.path.join(pf, f))
            else:
                rec_folder = os.path.join(today_folder, sel_period)
                csvs = sorted([f for f in os.listdir(rec_folder) if f.lower().endswith(".csv")], reverse=True)
                files_to_show = [os.path.join(rec_folder, f) for f in csvs]

            if not files_to_show:
                st.info("No saved records to display.")
            else:
                for fpath in files_to_show:
                    try:
                        df_view = pd.read_csv(fpath)
                    except Exception:
                        continue
                    subj = df_view["Subject"].iloc[0] if "Subject" in df_view.columns else ""
                    fac = df_view["Faculty"].iloc[0] if "Faculty" in df_view.columns else ""
                    period_label = df_view["Period"].iloc[0] if "Period" in df_view.columns else ""
                    dt_date = df_view["Date"].iloc[0] if "Date" in df_view.columns else today_str
                    dt_time = df_view["Time"].iloc[0] if "Time" in df_view.columns else ""
                    st.markdown("<div class='meta'>", unsafe_allow_html=True)
                    st.write(f"**Subject:** {subj or '—'}  |  **Faculty:** {fac or '—'}  |  **Period:** {period_label}  |  **Date:** {dt_date}  |  **Time:** {dt_time}")
                    st.markdown("</div>")
                    if {"Student Name", "Status"}.issubset(df_view.columns):
                        def render_student_table(df_local):
                            rows_html = "<table style='width:100%; border-collapse:collapse;'>"
                            for _, r in df_local.iterrows():
                                name = r["Student Name"]
                                status = r["Status"]
                                status_html = f"<span class='present'>✅ Present</span>" if str(status).strip().lower() == "present" else f"<span class='absent'>❌ Absent</span>"
                                rows_html += f"<tr><td style='padding:6px 8px; width:70%;'>{name}</td><td style='padding:6px 8px; text-align:right; width:30%;'>{status_html}</td></tr>"
                            rows_html += "</table>"
                            return rows_html
                        st.markdown(render_student_table(df_view[["Student Name","Status"]]), unsafe_allow_html=True)
                    else:
                        st.dataframe(df_view)
                    with open(fpath, "rb") as fh:
                        st.download_button("Download this record (CSV)", fh.read(), file_name=os.path.basename(fpath), mime="text/csv")
                    st.markdown("---")

                st.markdown("### Generate full-day report")
                st.write("Combine all saved period CSVs for today into one Day_Attendance file (manual action).")
                if st.button("Generate Full-Day Report (merge today's periods)"):
                    merged_rows = []
                    for p in period_folders:
                        pf = os.path.join(today_folder, p)
                        for f in sorted(os.listdir(pf)):
                            if not f.lower().endswith(".csv"):
                                continue
                            fp = os.path.join(pf, f)
                            try:
                                dfa = pd.read_csv(fp)
                            except Exception:
                                continue
                            for _, r in dfa.iterrows():
                                student = r.get("Student Name") or r.get("Name") or ""
                                status = r.get("Status") or ""
                                merged_rows.append({
                                    "Student Name": student,
                                    "Subject": r.get("Subject", ""),
                                    "Faculty": r.get("Faculty", ""),
                                    "Period": r.get("Period", p),
                                    "Time": r.get("Time", ""),
                                    "Date": r.get("Date", today_str),
                                    "Status": status
                                })
                    if merged_rows:
                        merged_df = pd.DataFrame(merged_rows)
                        day_fname = f"Day_Attendance_{today_str}.csv"
                        day_path = os.path.join(ROOT_SAVE, today_str, day_fname)
                        merged_df.to_csv(day_path, index=False)
                        st.success(f"Day report created: {os.path.relpath(day_path)}")
                        with open(day_path, "rb") as fh:
                            st.download_button("Download Day Report (CSV)", fh.read(), file_name=day_fname, mime="text/csv")
                    else:
                        st.info("No period records found to merge for today.")
