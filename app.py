import streamlit as st
import numpy as np, cv2
from PIL import Image
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Group Face Recognition", layout="wide")
st.title("👥 Group Face Recognition (Green = Matched | Red = Unmatched)")

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app = load_model()

group_file = st.file_uploader("Upload a Group Photo", type=["jpg","jpeg","png"])
face_files = st.file_uploader("Upload Individual Face Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if st.button("Run Recognition"):
    if not group_file or not face_files:
        st.warning("Please upload both a group photo and at least one individual face.")
    else:
        group_img = Image.open(group_file).convert("RGB")
        group_np = np.array(group_img)
        group_faces = app.get(group_np)
        total_faces = len(group_faces)
        if total_faces == 0:
            st.error("No faces detected in the group photo.")
        else:
            known_embeddings = []
            for f in face_files:
                try:
                    img = Image.open(f).convert("RGB")
                    arr = np.array(img)
                    faces = app.get(arr)
                    if faces:
                        known_embeddings.append(faces[0].normed_embedding)
                except Exception as e:
                    st.write("Skipping one face:", e)

            matched_count = 0
            annotated = group_np.copy()
            for face in group_faces:
                emb = face.normed_embedding
                matched = any(np.dot(emb, ref) > 0.45 for ref in known_embeddings)
                color = (0,255,0) if matched else (255,0,0)
                if matched: matched_count += 1
                box = face.bbox.astype(int)
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)

            summary = f"Total Faces: {total_faces} | Matched: {matched_count}"
            st.image(annotated, caption=summary, channels="BGR", use_container_width=True)
            st.success(summary)
