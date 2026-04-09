import streamlit as st
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString
from pdf2image import convert_from_bytes
import pytesseract
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
    direction = st.selectbox("Šiaurė", ["Up","Right","Down","Left"])

    NORTH = {
        "Up": np.array([0,-1]),
        "Down": np.array([0,1]),
        "Left": np.array([-1,0]),
        "Right": np.array([1,0])
    }[direction]

    # =========================
    # WALL DETECTION
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,2
    )

    kernel = np.ones((5,5), np.uint8)
    walls = cv2.dilate(thresh, kernel, iterations=2)

    inv = cv2.bitwise_not(walls)

    h,w = inv.shape
    mask = np.zeros((h+2,w+2), np.uint8)

    flood = inv.copy()
    cv2.floodFill(flood, mask, (0,0), 0)

    apartments_mask = inv - flood

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

        if 8000 < area < 300000:
            approx = cv2.approxPolyDP(cnt,5,True)
            pts = [(p[0][0],p[0][1]) for p in approx]

            if len(pts)>=4:
                polygons.append(Polygon(pts))

    # =========================
    # SORT LEFT → RIGHT
    # =========================
    polygons = sorted(polygons, key=lambda p: p.centroid.x)

    # =========================
    # OCR TABLE (dešinė pusė)
    # =========================
    h, w, _ = img.shape
    table_crop = img[:, int(w*0.7):]  # paimam dešinę dalį

    gray_table = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(gray_table, output_type=pytesseract.Output.DICT)

    areas = []

    for i, text in enumerate(data["text"]):
        t = text.strip().replace(",", ".")

        try:
            val = float(t)
            if 20 < val < 200:  # butų plotai
                areas.append(val)
        except:
            pass

    areas = sorted(areas, reverse=True)

    # =========================
    # VECTOR MATH
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

    def get_edges(poly):
        edges=[]
        coords=list(poly.exterior.coords)

        for i in range(len(coords)-1):
            e=LineString([coords[i],coords[i+1]])

            ext=True
            for o in polygons:
                if o!=poly and e.buffer(5).intersects(o):
                    ext=False
                    break

            if ext and e.length>50:
                edges.append(e)

        return edges

    # =========================
    # RESULTS
    # =========================
    results=[]

    for i, poly in enumerate(polygons):
        dirs=set()

        for e in get_edges(poly):
            x1,y1=e.coords[0]
            x2,y2=e.coords[1]

            v=np.array([x2-x1,y2-y1])
            n=normal(v)

            dirs.add(classify(angle(n,NORTH)))

        area_val = areas[i] if i < len(areas) else None

        results.append({
            "Apartment": f"B{i+1}",
            "Area_m2": area_val,
            "Directions": ", ".join(sorted(dirs))
        })

    df = pd.DataFrame(results)

    st.write(f"🏠 Butų: {len(df)}")
    st.dataframe(df)

    st.download_button(
        "📥 CSV",
        df.to_csv(index=False),
        "butai.csv"
    )
