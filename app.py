import streamlit as st
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString
from pdf2image import convert_from_bytes
import pytesseract
import math

st.set_page_config(layout="wide")

st.title("🏢 NT Butų Scanner PRO")

uploaded = st.file_uploader("Upload planą (PNG / JPG / PDF)", type=["png", "jpg", "jpeg", "pdf"])

if uploaded:
    file_bytes = uploaded.read()

    # =========================
    # LOAD IMAGE
    # =========================
    if uploaded.name.endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=300)
        img = np.array(pages[0])
    else:
        file_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    st.image(img, caption="Planas", use_container_width=True)

    # =========================
    # NORTH
    # =========================
    st.subheader("🧭 Šiaurės kryptis")

    direction_choice = st.selectbox(
        "Pasirink kryptį",
        ["Up (↑)", "Right (→)", "Down (↓)", "Left (←)"]
    )

    if direction_choice == "Up (↑)":
        NORTH = np.array([0, -1])
    elif direction_choice == "Down (↓)":
        NORTH = np.array([0, 1])
    elif direction_choice == "Left (←)":
        NORTH = np.array([-1, 0])
    else:
        NORTH = np.array([1, 0])

    # =========================
    # DETECT WALLS → BUTAI
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5), np.uint8)
    walls = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    inv = cv2.bitwise_not(walls)

    h, w = inv.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    flood = inv.copy()
    cv2.floodFill(flood, mask, (0,0), 0)

    apartments_mask = inv - flood

    contours, _ = cv2.findContours(
        apartments_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 5000 < area < 200000:
            approx = cv2.approxPolyDP(cnt, 5, True)
            pts = [(p[0][0], p[0][1]) for p in approx]

            if len(pts) >= 4:
                polygons.append(Polygon(pts))

    # =========================
    # OCR → BUTŲ NUMERIAI
    # =========================
    def detect_labels(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        labels = []

        for i, text in enumerate(data["text"]):
            t = text.strip()

            if t.startswith("B") and len(t) <= 3:
                x = data["left"][i]
                y = data["top"][i]

                labels.append({
                    "text": t,
                    "pos": np.array([x, y])
                })

        return labels

    labels = detect_labels(img)

    # =========================
    # MATCH BUTAS ↔ LABEL
    # =========================
    def match_apartments(polygons, labels):
        names = []

        for poly in polygons:
            c = np.array([poly.centroid.x, poly.centroid.y])

            best = "UNKNOWN"
            min_dist = 1e9

            for lab in labels:
                dist = np.linalg.norm(c - lab["pos"])

                if dist < min_dist:
                    min_dist = dist
                    best = lab["text"]

            names.append(best)

        return names

    names = match_apartments(polygons, labels)

    # =========================
    # VECTOR MATH
    # =========================
    def unit(v):
        return v / np.linalg.norm(v)

    def normal(v):
        return np.array([-v[1], v[0]])

    def angle(v1, v2):
        return math.degrees(
            math.acos(np.clip(np.dot(unit(v1), unit(v2)), -1, 1))
        )

    def classify(a):
        dirs = ["N","NE","E","SE","S","SW","W","NW"]
        return dirs[int((a+22.5)//45)%8]

    def get_external_edges(poly):
        edges = []
        coords = list(poly.exterior.coords)

        for i in range(len(coords)-1):
            e = LineString([coords[i], coords[i+1]])

            is_external = True

            for other in polygons:
                if other != poly and e.buffer(5).intersects(other):
                    is_external = False
                    break

            if is_external and e.length > 40:
                edges.append(e)

        return edges

    # =========================
    # CALCULATE
    # =========================
    results = []

    for i, poly in enumerate(polygons):
        dirs = set()

        for edge in get_external_edges(poly):
            x1, y1 = edge.coords[0]
            x2, y2 = edge.coords[1]

            v = np.array([x2 - x1, y2 - y1])
            n = normal(v)

            ang = angle(n, NORTH)
            dirs.add(classify(ang))

        results.append({
            "Apartment": names[i],
            "Directions": ", ".join(sorted(dirs))
        })

    df = pd.DataFrame(results)

    st.subheader("📊 Rezultatai")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Atsisiųsti CSV",
        csv,
        "butai.csv",
        "text/csv"
    )
