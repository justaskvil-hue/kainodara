import streamlit as st
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString
from pdf2image import convert_from_bytes
import math

st.set_page_config(layout="wide")

st.title("🏢 NT Butų Scanner")

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
    # NORTH (PAPRASTAS)
    # =========================
    st.subheader("🧭 Šiaurės kryptis")

    direction_choice = st.selectbox(
        "Pasirink kryptį (kur plane yra šiaurė)",
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
    # SCALE
    # =========================
    st.subheader("📐 Scale")
    ppm = st.number_input("Pixels per meter", value=100)

    # =========================
    # PREPROCESS
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # =========================
    # FIND APARTMENTS (FIXED)
    # =========================
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 2000 < area < 50000:

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 0

            if 0.3 < aspect_ratio < 3:

                approx = cv2.approxPolyDP(cnt, 5, True)
                pts = [(p[0][0], p[0][1]) for p in approx]

                if len(pts) >= 4:
                    polygons.append(Polygon(pts))

    # =========================
    # HELPERS
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
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return dirs[int((a + 22.5) // 45) % 8]

    def get_external_edges(poly):
        edges = []
        coords = list(poly.exterior.coords)

        for i in range(len(coords) - 1):
            e = LineString([coords[i], coords[i + 1]])

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

        area_m2 = poly.area / (ppm ** 2)

        results.append({
            "Apartment": f"A{i+1}",
            "Area_m2": round(area_m2, 2),
            "Directions": ", ".join(sorted(dirs))
        })

    df = pd.DataFrame(results)

    st.subheader("📊 Rezultatai")
    st.dataframe(df, use_container_width=True)

    # =========================
    # DOWNLOAD
    # =========================
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Atsisiųsti CSV",
        csv,
        "butai.csv",
        "text/csv"
    )
