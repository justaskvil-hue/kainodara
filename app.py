import streamlit as st
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString
from pdf2image import convert_from_bytes
import math

st.set_page_config(layout="wide")
st.title("🏢 NT Butų Scanner")

uploaded = st.file_uploader("Upload planą", type=["png","jpg","jpeg","pdf"])

if uploaded:
    file_bytes = uploaded.read()

    # =========================
    # LOAD IMAGE
    # =========================
    if uploaded.name.endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=300)
        img = np.array(pages[0])
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    st.image(img, use_container_width=True)

    # =========================
    # NORTH
    # =========================
    direction = st.selectbox("🧭 Šiaurė", ["Up","Right","Down","Left"])

    NORTH = {
        "Up": np.array([0,-1]),
        "Down": np.array([0,1]),
        "Left": np.array([-1,0]),
        "Right": np.array([1,0])
    }[direction]

    # =========================
    # PREPROCESS
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # paprastas threshold (stabilus)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # mažesnis kernel → nesujungia visko
    kernel = np.ones((3,3), np.uint8)
    walls = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # =========================
    # FLOOD FILL
    # =========================
    inv = cv2.bitwise_not(walls)

    h, w = inv.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    flood = inv.copy()
    cv2.floodFill(flood, mask, (1,1), 255)

    apartments_mask = cv2.bitwise_not(flood)

    # DEBUG – labai svarbu
    st.image(apartments_mask, caption="DEBUG butų zonos")

    # =========================
    # FIND APARTMENTS
    # =========================
    contours,_ = cv2.findContours(
        apartments_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # lengvesnis filtras
        if area > 3000:
            approx = cv2.approxPolyDP(cnt,5,True)
            pts = [(p[0][0],p[0][1]) for p in approx]

            if len(pts) >= 4:
                polygons.append(Polygon(pts))

    st.write(f"🏠 Butų: {len(polygons)}")

    # =========================
    # VECTOR FUNKCIJOS
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

    # =========================
    # EXTERNAL EDGES
    # =========================
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
    # RESULTS
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
            "Apartment": f"A{i+1}",
            "Area_px": int(poly.area),
            "Directions": ", ".join(sorted(dirs))
        })

    df = pd.DataFrame(results)

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 CSV",
        df.to_csv(index=False),
        "butai.csv"
    )
