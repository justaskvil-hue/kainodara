import streamlit as st
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString
from pdf2image import convert_from_bytes
import math

st.set_page_config(layout="wide")
st.title("🏢 NT Auto Scanner (FIXED)")

uploaded = st.file_uploader("Upload planą", type=["png","jpg","jpeg","pdf"])

if uploaded:
    file_bytes = uploaded.read()

    if uploaded.name.endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=300)
        img = np.array(pages[0])
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    st.image(img, use_container_width=True)

    # =========================
    # NORTH
    # =========================
    direction = st.selectbox("Šiaurė", ["Up","Right","Down","Left"])

    NORTH = {
        "Up": np.array([0,-1]),
        "Down": np.array([0,1]),
        "Left": np.array([-1,0]),
        "Right": np.array([1,0])
    }[direction]

    # =========================
    # GRAYSCALE
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # stipresnis kontrastas
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    st.image(edges, caption="DEBUG edges")

    # =========================
    # FIND CONTOURS
    # =========================
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # filtrai – svarbiausia dalis
        if 15000 < area < 300000:
            pts = [(p[0][0], p[0][1]) for p in cnt]

            if len(pts) >= 4:
                polygons.append(Polygon(pts))

    # rūšiavimas
    polygons = sorted(polygons, key=lambda p: p.centroid.x)

    st.write(f"🏠 Butų: {len(polygons)}")

    # =========================
    # LANGŲ DETECTION
    # =========================
    def get_windows(poly):
        edges = []
        coords = list(poly.exterior.coords)

        for i in range(len(coords)-1):
            p1 = coords[i]
            p2 = coords[i+1]

            line = LineString([p1, p2])

            if line.length < 50:
                continue

            mx = int((p1[0]+p2[0])/2)
            my = int((p1[1]+p2[1])/2)

            patch = gray[max(0,my-6):my+6, max(0,mx-6):mx+6]

            if patch.size == 0:
                continue

            if np.mean(patch) > 170:
                edges.append(line)

        return edges

    def classify(v):
        dirs = ["N","NE","E","SE","S","SW","W","NW"]
        ang = math.degrees(math.atan2(v[1], v[0]))
        ang = (ang+360)%360
        return dirs[int((ang+22.5)//45)%8]

    # =========================
    # RESULTS
    # =========================
    results = []

    for i, poly in enumerate(polygons):
        dirs = set()

        for e in get_windows(poly):
            x1,y1 = e.coords[0]
            x2,y2 = e.coords[1]

            v = np.array([x2-x1, y2-y1])
            dirs.add(classify(v))

        results.append({
            "Apartment": f"B{i+1}",
            "Directions": ", ".join(sorted(dirs)) if dirs else "?"
        })

    df = pd.DataFrame(results)

    st.dataframe(df)
