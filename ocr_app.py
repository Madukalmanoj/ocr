import streamlit as st
import easyocr
import cv2
import json
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import tempfile
import time

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(page_title="Advanced OCR System", layout="wide")
st.title("📄 Advanced OCR System (EasyOCR + TrOCR)")

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write("Device:", device)

# =====================================================
# LOAD MODELS (Cached)
# =====================================================

@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    model.to(device)
    model.eval()

    return reader, processor, model


reader, processor, model = load_models()

# =====================================================
# IMAGE ENHANCEMENT (UNCHANGED)
# =====================================================

def multi_enhance(image):
    versions = [image]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (8,8))
    cl = clahe.apply(gray)
    versions.append(cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR))

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    versions.append(cv2.filter2D(image, -1, kernel))

    return versions

# =====================================================
# MULTI-SCALE DETECTION (UNCHANGED)
# =====================================================

def multi_scale_detect(image):
    scales = [0.75, 1.0, 1.5, 2.0]
    detections = []

    for scale in scales:
        resized = cv2.resize(image, None, fx=scale, fy=scale)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb)

        for (bbox, text, conf) in results:
            if conf < 0.35:
                continue

            bbox = (np.array(bbox) / scale).astype(int)

            x1 = int(np.min(bbox[:,0]))
            y1 = int(np.min(bbox[:,1]))
            x2 = int(np.max(bbox[:,0]))
            y2 = int(np.max(bbox[:,1]))

            detections.append({
                "box": [x1, y1, x2, y2],
                "text_easy": text.strip(),
                "conf_easy": float(conf)
            })

    return detections

# =====================================================
# IOU
# =====================================================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xB <= xA or yB <= yA:
        return 0.0

    interArea = (xB - xA) * (yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

def fuse_boxes(detections):
    detections = sorted(detections, key=lambda x: -x["conf_easy"])
    final = []

    for d in detections:
        keep = True
        for f in final:
            if iou(d["box"], f["box"]) > 0.4:
                keep = False
                break
        if keep:
            final.append(d)

    return final

# =====================================================
# TROCR
# =====================================================

def recognize_trocr(crop):
    if crop.size == 0:
        return ""

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=rgb, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# =====================================================
# CONSENSUS (UNCHANGED)
# =====================================================

def consensus(easy_text, trocr_text):
    easy_text = easy_text.upper().strip()
    trocr_text = trocr_text.upper().strip()

    if not trocr_text:
        return easy_text
    if not easy_text:
        return trocr_text

    if easy_text in trocr_text:
        return trocr_text
    if trocr_text in easy_text:
        return easy_text

    return trocr_text if len(trocr_text) > len(easy_text) else easy_text

# =====================================================
# LINE RECONSTRUCTION (UNCHANGED)
# =====================================================

def reconstruct_lines(results):

    results = sorted(results, key=lambda r: (r["box"][1], r["box"][0]))
    lines = []
    used = [False] * len(results)

    for i, r1 in enumerate(results):
        if used[i]:
            continue

        x1, y1, x2, y2 = r1["box"]
        h1 = y2 - y1
        cy1 = (y1 + y2) / 2

        current_line = [r1]
        used[i] = True

        for j, r2 in enumerate(results):
            if used[j]:
                continue

            x3, y3, x4, y4 = r2["box"]
            h2 = y4 - y3
            cy2 = (y3 + y4) / 2

            if abs(cy1 - cy2) < max(h1, h2) * 0.6:
                current_line.append(r2)
                used[j] = True

        current_line = sorted(current_line, key=lambda r: r["box"][0])

        seen = set()
        words = []
        for r in current_line:
            w = r["text"]
            if w not in seen:
                words.append(w)
                seen.add(w)

        line_text = " ".join(words)

        lines.append({
            "text": line_text,
            "confidence": float(np.mean([r["confidence"] for r in current_line]))
        })

    return lines

# =====================================================
# OCR PIPELINE (UNCHANGED)
# =====================================================

def extract_text_from_image(image_np):

    enhanced_versions = multi_enhance(image_np)

    detections = []
    for v in enhanced_versions:
        detections.extend(multi_scale_detect(v))

    detections = fuse_boxes(detections)

    final_results = []

    for d in detections:
        x1,y1,x2,y2 = d["box"]
        crop = image_np[y1:y2, x1:x2]

        trocr_text = recognize_trocr(crop)
        final_text = consensus(d["text_easy"], trocr_text)

        final_results.append({
            "text": final_text,
            "confidence": float(d["conf_easy"]),
            "box": [x1, y1, x2, y2]
        })

    lines = reconstruct_lines(final_results)

    output = image_np.copy()

    for r in final_results:
        x1,y1,x2,y2 = r["box"]
        cv2.rectangle(output,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(output,r["text"],(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    return output, lines

# =====================================================
# STREAMLIT UI (IMAGE + VIDEO)
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "png", "jpeg", "mp4"]
)

if uploaded_file:

    if "image" in uploaded_file.type:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", width="stretch")

        if st.button("Run OCR"):

            with st.spinner("Running OCR..."):
                output_img, lines = extract_text_from_image(image_np)

            st.image(output_img, caption="OCR Result", width="stretch")
            st.json(lines)

    elif "video" in uploaded_file.type:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output_frame, _ = extract_text_from_image(frame)

            stframe.image(output_frame, channels="RGB", width="stretch")

        cap.release()
