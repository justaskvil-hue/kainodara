import streamlit as st
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString
from pdf2image import convert_from_bytes
import math

st.set_page_config(layout="wide")
st.title("🏢 NT Scanner PRO")

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
    # PLOTŲ INPUT
    # =========================
    st.subheader("📐 Plotai (iš lentelės)")
    area_input = st.text_area("Pvz: 46.92, 29.06, 29.97, 59.32, 55.95")

    areas = []
    if area_input:
        try:
            areas = [float(x.strip().replace(",", ".")) for x in area_input.split(",")]
        except:
            st.warning("Blogas formatas")

    # =========================
    # IMAGE PROCESSING
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,2
    )

    kernel = np.ones((3,3), np.uint8)
    walls = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    inv = cv2.bitwise_not(walls)

    h, w = inv.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    flood = inv.copy()
    cv2.floodFill(flood, mask, (0,0), 0)

    apartments_mask = inv - flood

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

        if 6000 < area < 150000:
            approx = cv2.approxPolyDP(cnt,5,True)
            pts = [(p[0][0],p[0][1]) for p in approx]

            if len(pts)>=4:
                polygons.append(Polygon(pts))

    polygons = sorted(polygons, key=lambda p: p.centroid.x)

    st.write(f"🏠 Butų: {len(polygons)}")

    # =========================
    # MATH
    # =========================
    def unit(v): return v/np.linalg.norm(v)
    def normal(v): return np.array([-v[1],v[0]])

    def angle(v1,v2):
        return math.degrees(
            math.acos(np.clip(np.dot(unit(v1),unit(v2)),-1,1))
        )

    def classify(a):
        dirs=["N","NE","E","SE","S","SW","W","NW"]
        return dirs[int((a+22.5)//45)%8]

    # =========================
    # LANGŲ DETECTION (KEY PART)
    # =========================
    def get_window_edges(poly, img):
        edges=[]
        coords=list(poly.exterior.coords)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(len(coords)-1):
            p1 = coords[i]
            p2 = coords[i+1]

            line = LineString([p1,p2])

            if line.length < 40:
                continue

            x1,y1 = int(p1[0]), int(p1[1])
            x2,y2 = int(p2[0]), int(p2[1])

            mx = int((x1+x2)/2)
            my = int((y1+y2)/2)

            patch = gray[max(0,my-6):my+6, max(0,mx-6):mx+6]

            if patch.size == 0:
                continue

            avg = np.mean(patch)

            # langai = šviesesni nei sienos
            if avg > 170:
                edges.append(line)

        return edges

    # =========================
    # RESULTS
    # =========================
    results=[]

    for i, poly in enumerate(polygons):
        dirs=set()

        window_edges = get_window_edges(poly, img)

        for e in window_edges:
            x1,y1=e.coords[0]
            x2,y2=e.coords[1]

            v=np.array([x2-x1,y2-y1])
            n=normal(v)

            dirs.add(classify(angle(n,NORTH)))

        area_val = areas[i] if i < len(areas) else None

        results.append({
            "Apartment": f"B{i+1}",
            "Area_m2": area_val,
            "Directions": ", ".join(sorted(dirs)) if dirs else "?"
        })

    df=pd.DataFrame(results)

    st.dataframe(df)

    st.download_button("📥 CSV", df.to_csv(index=False), "butai.csv")
