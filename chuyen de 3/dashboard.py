import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
    color: white;
}

/* ===== TITLE ===== */
h1 {
    color: #0f172a;
    font-weight: 600;
}

/* ===== KPI CARD ===== */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
}

/* ===== SELECT ===== */
div[data-baseweb="select"] > div {
    background: white !important;
    border: 1px solid #cbd5e1 !important;
}

/* ===== TAG ===== */
span[data-baseweb="tag"] {
    background-color: #2563eb !important;
    color: white !important;
}

/* ===== TAB ===== */
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #2563eb;
    color: #2563eb;
}

</style>
""", unsafe_allow_html=True)
sns.set(style="whitegrid")

st.title("Dashboard phân tích dữ liệu TMĐT")

# =========================
# TẠO THƯ MỤC CHUẨN
# =========================
IMG_DIR = "images"
REPORT_DIR = "reports"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 👉 XÓA ẢNH CŨ (mỗi lần chạy)
import shutil

for f in os.listdir(IMG_DIR):
    path = os.path.join(IMG_DIR, f)

    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/superstore_cleaned.csv")
    df_clean = pd.read_csv("data/superstore_final.csv")

    df_clean["Order Date"] = pd.to_datetime(df_clean["Order Date"])
    df_clean["Year"] = df_clean["Order Date"].dt.year
    df_clean["Month"] = df_clean["Order Date"].dt.month

    return df_raw, df_clean

df_raw, df = load_data()

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Bộ lọc")

selected_year = st.sidebar.selectbox(
    "Chọn năm", sorted(df["Year"].unique())
)

selected_regions = st.sidebar.multiselect(
    "Chọn khu vực",
    sorted(df["Region"].unique()),
    default=df["Region"].unique()
)

temp_df = df[
    (df["Year"] == selected_year) &
    (df["Region"].isin(selected_regions))
]

selected_product = st.sidebar.selectbox(
    "Chọn sản phẩm",
    ["All"] + sorted(temp_df["Product Name"].unique())
)

filtered_df = temp_df.copy()
if selected_product != "All":
    filtered_df = filtered_df[
        filtered_df["Product Name"] == selected_product
    ]

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([
    "Data Cleaning",
    "Dashboard",
    "Model"
])

# =========================
# TAB 1: DATA CLEANING
# =========================
with tab1:
    st.subheader("Dữ liệu trước khi làm sạch")
    st.write(df_raw.shape)
    st.dataframe(df_raw.head())

    st.subheader("Dữ liệu sau khi làm sạch")
    st.write(df.shape)
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    col1.metric("Ban đầu", df_raw.shape[0])
    col2.metric("Sau clean", df.shape[0])
    col3.metric("Đã xóa", df_raw.shape[0] - df.shape[0])

    st.write(
        "Tỷ lệ giữ lại:",
        round(len(df) / len(df_raw) * 100, 2),
        "%"
    )

# =========================
# TAB 2: DASHBOARD
# =========================
with tab2:

    # KPI
    total_sales = int(filtered_df["Sales"].sum())
    total_orders = filtered_df["Order ID"].nunique()
    avg_sales = int(filtered_df["Sales"].mean())

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng doanh thu", f"{total_sales:,}")
    col2.metric("Số đơn hàng", total_orders)
    col3.metric("Trung bình / đơn", f"{avg_sales:,}")

    # ===== BIỂU ĐỒ =====
    col1, col2 = st.columns(2)

    # 📈 Trend
    with col1:
        st.subheader("Xu hướng doanh thu")

        trend = filtered_df.groupby("Month")["Sales"].sum().reset_index()

        fig1, ax1 = plt.subplots()
        sns.lineplot(data=trend, x="Month", y="Sales", marker="o", ax=ax1)
        st.pyplot(fig1)

        fig1_path = os.path.join(IMG_DIR, "trend.png")
        fig1.savefig(fig1_path)

    # 🔥 Top sản phẩm
    with col2:
        st.subheader("Top sản phẩm")

        top_products = (
            filtered_df.groupby("Product Name")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig2, ax2 = plt.subplots()
        sns.barplot(data=top_products, x="Sales", y="Product Name", ax=ax2)
        st.pyplot(fig2)

        fig2_path = os.path.join(IMG_DIR, "top.png")
        fig2.savefig(fig2_path)

    # 🌍 Region
    st.subheader("Doanh thu theo khu vực")

    region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()

    fig3, ax3 = plt.subplots()
    ax3.pie(region_sales["Sales"], labels=region_sales["Region"], autopct="%1.1f%%")
    st.pyplot(fig3)

    fig3_path = os.path.join(IMG_DIR, "region.png")
    fig3.savefig(fig3_path)

    st.subheader("Dữ liệu chi tiết")
    st.dataframe(filtered_df.head(20))

# =========================
# TAB 3: MODEL
# =========================
with tab3:
    st.subheader("Dự báo doanh thu")

    df_model = filtered_df.sort_values("Order Date").copy()
    df_model["Time"] = np.arange(len(df_model))

    X = df_model[["Time"]]
    y = df_model["Sales"]

    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X, y)

    pred_lr = lr.predict(X)
    pred_rf = rf.predict(X)

    fig4, ax4 = plt.subplots()
    ax4.plot(y.values, label="Thực tế")
    ax4.plot(pred_lr, label="Linear Regression")
    ax4.plot(pred_rf, label="Random Forest")
    ax4.legend()

    st.pyplot(fig4)

    fig4_path = os.path.join(IMG_DIR, "forecast.png")
    fig4.savefig(fig4_path)

# =========================
# EXPORT PDF
# =========================
st.subheader("Xuất báo cáo PDF")

if st.button("Tạo PDF Report"):

    today = datetime.now().strftime("%Y-%m-%d")
    file_name = os.path.join(REPORT_DIR, f"report_{today}.pdf")

    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("BÁO CÁO TMĐT", styles["Title"]))
    content.append(Spacer(1,10))

    content.append(Paragraph(f"Tổng doanh thu: {total_sales:,}", styles["Normal"]))
    content.append(Paragraph(f"Số đơn: {total_orders}", styles["Normal"]))
    content.append(Paragraph(f"TB/đơn: {avg_sales:,}", styles["Normal"]))
    content.append(Spacer(1,20))

    content.append(Image(fig1_path, width=400, height=200))
    content.append(Image(fig2_path, width=400, height=200))
    content.append(Image(fig3_path, width=400, height=200))
    content.append(Image(fig4_path, width=400, height=200))

    doc.build(content)

    st.success(f"Đã lưu: {file_name}")