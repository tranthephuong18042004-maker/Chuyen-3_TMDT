import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import shutil
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.ticker as ticker

# =========================
# 1. CONFIG & CSS (UI/UX CHUYÊN NGHIỆP)
# =========================
st.set_page_config(page_title="E-Commerce Pro Dashboard", page_icon="", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
    h1 { color: #1e3a8a; font-weight: 800; text-align: center; margin-bottom: 30px; text-transform: uppercase;}
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-5px); box-shadow: 0 8px 12px rgba(0,0,0,0.1); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", context="paper")

st.title(" DASHBOARD PHÂN TÍCH TMĐT TOÀN DIỆN")

# =========================
# 2. SETUP THƯ MỤC ẢNH
# =========================
IMG_DIR = "images"
REPORT_DIR = "reports"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Làm sạch thư mục ảnh mỗi lần F5 web
for f in os.listdir(IMG_DIR):
    path = os.path.join(IMG_DIR, f)
    if os.path.isfile(path): os.remove(path)

# =========================
# 3. LOAD DỮ LIỆU
# =========================
@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/superstore_cleaned.csv")
    df = pd.read_csv("data/superstore_final.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter_Year"] = df["Order Date"].dt.to_period("Q").astype(str)
    if "Profit" in df.columns:
        df["Profit Margin (%)"] = (df["Profit"] / df["Sales"]) * 100
    return df_raw, df

df_raw, df = load_data()

# =========================
# 4. THANH SIDEBAR (ĐỈNH CAO BỘ LỌC)
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081648.png", width=120)
    st.header(" LỌC DỮ LIỆU")
    selected_year = st.selectbox(" Chọn năm", ["Tất cả"] + sorted(df["Year"].unique().tolist(), reverse=True))
    selected_regions = st.multiselect(" Chọn khu vực", sorted(df["Region"].unique()), default=df["Region"].unique())

# Áp dụng bộ lọc Data
filtered_df = df.copy()
if selected_year != "Tất cả":
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]
filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]

with st.sidebar:
    st.markdown("---")
    st.header(" LỌC BIỂU ĐỒ HIỂN THỊ")
    
    # DANH SÁCH 10 BIỂU ĐỒ PHÂN TÍCH (Biểu đồ thứ 11 là AI nằm ở Tab riêng)
    all_charts = [
        "1. Xu hướng Doanh thu (Tháng)",
        "2. Tăng trưởng Doanh thu (Quý)",
        "3. Top 10 Sản phẩm Doanh thu cao",
        "4. Tỷ trọng Doanh thu (Khu vực)",
        "5. Lợi nhuận ròng theo Nhóm hàng",
        "6. Tỷ suất Lợi nhuận (%) theo Nhóm",
        "7. Tác động Giảm giá -> Lợi nhuận (Line)",
        "8. Điểm chết Giảm giá (Bar)",
        "9. Phân khúc Khách hàng (Sales & Profit)",
        "10. Top 10 Khách hàng VIP"
    ]
    
    selected_charts = st.multiselect(
        "Chọn các biểu đồ bạn muốn xem và xuất PDF:",
        all_charts,
        default=["1. Xu hướng Doanh thu (Tháng)", "3. Top 10 Sản phẩm Doanh thu cao", "6. Tỷ suất Lợi nhuận (%) theo Nhóm", "10. Top 10 Khách hàng VIP"] # Mặc định show 4 cái đẹp nhất
    )

# =========================
# 5. KPI CHÍNH
# =========================
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum() if "Profit" in filtered_df.columns else 0
total_orders = filtered_df["Order ID"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Doanh thu ($)", f"{total_sales:,.0f}")
col2.metric("Lợi nhuận ($)", f"{total_profit:,.0f}")
col3.metric("Số lượng Đơn hàng", f"{total_orders:,}")
col4.metric("Giá trị TB/Đơn ($)", f"{(total_sales/total_orders if total_orders else 0):,.0f}")
st.markdown("---")

# =========================
# 6. KHAI BÁO 10 HÀM VẼ BIỂU ĐỒ ĐỘC LẬP
# =========================
def plot_1(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    trend = data.groupby("Month")["Sales"].sum().reset_index()
    sns.lineplot(data=trend, x="Month", y="Sales", marker="o", color="#2980b9", ax=ax)
    ax.set_title("1. Xu hướng Doanh thu theo Tháng", pad=15, fontweight='bold')
    return fig

def plot_2(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    qt = data.groupby("Quarter_Year")["Sales"].sum().reset_index()
    sns.barplot(data=qt, x="Quarter_Year", y="Sales", color="#3498db", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("2. Biến động Doanh thu theo Quý", pad=15, fontweight='bold')
    return fig

def plot_3(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    top = data.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(data=top, x="Sales", y="Product Name", palette="viridis", ax=ax)
    ax.set_ylabel("")
    ax.set_title("3. Top 10 Sản phẩm bán chạy", pad=15, fontweight='bold')
    return fig

def plot_4(data):
    fig, ax = plt.subplots(figsize=(6, 6))
    reg = data.groupby("Region")["Sales"].sum()
    ax.pie(reg.values, labels=reg.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
    ax.set_title("4. Tỷ trọng Doanh thu Khu vực", pad=15, fontweight='bold')
    return fig

def plot_5(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = data.groupby(["Category", "Sub-Category"])["Profit"].sum().reset_index().sort_values("Profit", ascending=False)
    sns.barplot(data=sc, x="Profit", y="Sub-Category", hue="Category", dodge=False, ax=ax)
    ax.axvline(0, color='black', lw=1.5)
    ax.set_title("5. Lợi nhuận ròng theo Nhóm hàng", pad=15, fontweight='bold')
    return fig

def plot_6(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    mg = data.groupby("Sub-Category")[["Sales", "Profit"]].sum()
    mg["Margin"] = (mg["Profit"] / mg["Sales"]) * 100
    mg = mg.sort_values("Margin", ascending=False).reset_index()
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in mg["Margin"]]
    sns.barplot(data=mg, x="Margin", y="Sub-Category", palette=colors, ax=ax)
    ax.axvline(0, color='black', lw=1.5)
    ax.set_title("6. Tỷ suất Lợi nhuận (%) theo Nhóm", pad=15, fontweight='bold')
    return fig

def plot_7(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=data, x="Discount", y="Profit", errorbar=None, marker="o", color="red", lw=2, ax=ax)
    ax.axhline(0, color='black', ls='--', lw=1.5)
    ax.set_title("7. Tác động của Giảm giá đến Lợi nhuận", pad=15, fontweight='bold')
    return fig

def plot_8(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    dp = data.groupby("Discount")["Profit"].mean().reset_index()
    sns.barplot(data=dp, x="Discount", y="Profit", palette=['#f39c12' if p > 0 else '#c0392b' for p in dp["Profit"]], ax=ax)
    ax.axhline(0, color='black', lw=1.5)
    ax.set_title("8. Điểm chết Giảm giá", pad=15, fontweight='bold')
    return fig

def plot_9(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    seg = data.groupby("Segment")[["Sales", "Profit"]].sum().reset_index().melt(id_vars="Segment", var_name="Metric", value_name="Value")
    sns.barplot(data=seg, x="Segment", y="Value", hue="Metric", palette=["#3498db", "#2ecc71"], ax=ax)
    ax.set_title("9. Hiệu quả theo Phân khúc KH", pad=15, fontweight='bold')
    return fig

def plot_10(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    top_c = data.groupby("Customer Name")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False).head(10).reset_index()
    top_c_m = top_c.melt(id_vars="Customer Name", var_name="Metric", value_name="Value")
    sns.barplot(data=top_c_m, x="Value", y="Customer Name", hue="Metric", palette=["#34495e", "#27ae60"], ax=ax)
    ax.set_ylabel("")
    ax.set_title("10. Đóng góp Top 10 VIP", pad=15, fontweight='bold')
    return fig

# Mapping dictionary
chart_funcs = {
    "1. Xu hướng Doanh thu (Tháng)": plot_1,
    "2. Tăng trưởng Doanh thu (Quý)": plot_2,
    "3. Top 10 Sản phẩm Doanh thu cao": plot_3,
    "4. Tỷ trọng Doanh thu (Khu vực)": plot_4,
    "5. Lợi nhuận ròng theo Nhóm hàng": plot_5,
    "6. Tỷ suất Lợi nhuận (%) theo Nhóm": plot_6,
    "7. Tác động Giảm giá -> Lợi nhuận (Line)": plot_7,
    "8. Điểm chết Giảm giá (Bar)": plot_8,
    "9. Phân khúc Khách hàng (Sales & Profit)": plot_9,
    "10. Top 10 Khách hàng VIP": plot_10
}

# =========================
# 7. CHIA TABS VÀ RENDER
# =========================
tab1, tab2, tab3 = st.tabs([" Lưới Biểu đồ (Visuals)", " AI Dự báo (Model)", " Dữ liệu & Báo cáo PDF"])

with tab1:
    if not selected_charts:
        st.info(" Vui lòng chọn ít nhất 1 biểu đồ ở thanh Sidebar để hiển thị!")
    else:
        # Tự động chia biểu đồ thành lưới 2 cột
        cols = st.columns(2)
        for idx, chart_name in enumerate(selected_charts):
            with cols[idx % 2]:
                fig = chart_funcs[chart_name](filtered_df)
                fig.tight_layout()
                st.pyplot(fig)
                # Lưu file ảnh tự động (để chuẩn bị cho PDF)
                safe_name = chart_name.split(".")[0] # Lấy số làm tên file
                fig.savefig(os.path.join(IMG_DIR, f"chart_{safe_name}.png"), dpi=300, bbox_inches='tight')

with tab2:
    st.subheader(" Dự báo Doanh thu (Biểu đồ thứ 11)")
    df_model = filtered_df.sort_values("Order Date").copy()
    df_model["Time"] = np.arange(len(df_model))
    
    if len(df_model) > 10:
        X, y = df_model[["Time"]], df_model["Sales"]
        lr, rf = LinearRegression().fit(X, y), RandomForestRegressor(n_estimators=100, random_state=0).fit(X, y)
        
        fig11, ax11 = plt.subplots(figsize=(12, 5))
        ax11.plot(y.values, label="Thực tế", alpha=0.5, color="gray")
        ax11.plot(lr.predict(X), label="Linear Regression", linestyle="--", color="blue")
        ax11.plot(rf.predict(X), label="Random Forest", color="orange", alpha=0.8)
        ax11.legend()
        ax11.set_title("11. AI So sánh Dự báo Doanh thu", fontweight='bold')
        fig11.tight_layout()
        st.pyplot(fig11)
        fig11.savefig(os.path.join(IMG_DIR, "chart_11_AI.png"), dpi=300, bbox_inches='tight')
    else:
        st.warning("Dữ liệu quá ít để AI học. Vui lòng bỏ bớt bộ lọc.")

with tab3:
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader(" Trạng thái Data")
        st.write(f"- Dòng ban đầu: **{len(df_raw):,}**")
        st.write(f"- Dòng hiện tại (sau lọc): **{len(filtered_df):,}**")
        st.dataframe(filtered_df.head(20), height=250)
        
    with colB:
        st.subheader(" Xuất Báo cáo Cuối kỳ")
        st.info("File PDF xuất ra sẽ chứa CHÍNH XÁC những biểu đồ bạn đang hiển thị ở Tab 1 và Tab 2.")
        
        if st.button(" TẠO REPORT PDF NGAY", use_container_width=True, type="primary"):
            with st.spinner("Đang tổng hợp báo cáo..."):
                today = datetime.now().strftime("%Y-%m-%d_%H%M")
                file_name = os.path.join(REPORT_DIR, f"Report_{today}.pdf")

                doc = SimpleDocTemplate(file_name)
                styles = getSampleStyleSheet()
                content = []

                # Trang bìa / Tóm tắt
                content.append(Paragraph("BÁO CÁO KINH DOANH CHUYÊN SÂU", styles["Title"]))
                content.append(Spacer(1,15))
                content.append(Paragraph(f"Thời gian: {datetime.now().strftime('%d/%m/%Y')} | Năm phân tích: {selected_year}", styles["Normal"]))
                content.append(Spacer(1,10))
                content.append(Paragraph(f"- Tổng doanh thu: ${total_sales:,.0f}", styles["Heading3"]))
                content.append(Paragraph(f"- Tổng lợi nhuận: ${total_profit:,.0f}", styles["Heading3"]))
                content.append(Spacer(1,20))

                # Tự động nhặt tất cả ảnh biểu đồ đã được sinh ra để in vào PDF
                saved_images = sorted(os.listdir(IMG_DIR))
                for idx, img_file in enumerate(saved_images):
                    if img_file.endswith(".png"):
                        content.append(Image(os.path.join(IMG_DIR, img_file), width=420, height=210))
                        content.append(Spacer(1, 15))
                        # Ngắt trang sau mỗi 3 biểu đồ
                        if (idx + 1) % 3 == 0:
                            content.append(PageBreak())

                doc.build(content)
                st.success(f" Báo cáo đã hoàn tất: **{file_name}**")
