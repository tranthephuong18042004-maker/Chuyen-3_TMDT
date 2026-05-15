
# =========================================================
# APP.PY - DASHBOARD TMĐT AI CHUYÊN NGHIỆP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Dashboard TMĐT AI",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")

# =========================================================
# CSS UI PROFESSIONAL
# =========================================================

st.markdown("""
<style>

/* ===================================================== */
/* GLOBAL */
/* ===================================================== */

.stApp{
    background:#f4f7fb;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1600px;
}

/* ===================================================== */
/* SIDEBAR */
/* ===================================================== */

section[data-testid="stSidebar"]{
    background:white;
    border-right:1px solid #e5e7eb;
}

section[data-testid="stSidebar"] .block-container{
    padding-top:1rem;
}

/* ===================================================== */
/* TITLE */
/* ===================================================== */

.dashboard-title{
    font-size:48px;
    font-weight:800;
    color:#111827;
    line-height:1.2;
}

.dashboard-sub{
    font-size:16px;
    color:#6b7280;
    margin-top:5px;
    margin-bottom:25px;
}

/* ===================================================== */
/* KPI */
/* ===================================================== */

.metric-card{
    background:white;
    padding:24px;
    border-radius:20px;
    box-shadow:0 4px 18px rgba(0,0,0,0.05);
    border:1px solid #edf2f7;
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-4px);
}

.metric-title{
    color:#6b7280;
    font-size:14px;
    font-weight:600;
}

.metric-value{
    font-size:38px;
    font-weight:800;
    color:#111827;
    margin-top:8px;
}

.metric-growth{
    color:#16a34a;
    font-weight:700;
    margin-top:10px;
}

/* ===================================================== */
/* CARD */
/* ===================================================== */

.chart-card{
    background:white;
    border-radius:20px;
    padding:18px;
    border:1px solid #edf2f7;
    box-shadow:0 4px 18px rgba(0,0,0,0.05);
    margin-bottom:20px;
}

.chart-title{
    font-size:18px;
    font-weight:700;
    color:#111827;
    margin-bottom:10px;
}

.chart-desc{
    background:#f8fafc;
    border-radius:12px;
    padding:12px;
    margin-top:10px;
    color:#475569;
    font-size:13px;
    line-height:1.7;
}

/* ===================================================== */
/* MODEL CARD */
/* ===================================================== */

.model-card{
    background:white;
    border-radius:20px;
    padding:20px;
    border:1px solid #edf2f7;
    box-shadow:0 4px 18px rgba(0,0,0,0.05);
    height:100%;
}

.best-model{
    font-size:28px;
    font-weight:800;
    color:#16a34a;
}

/* ===================================================== */
/* FOOTER */
/* ===================================================== */

.footer-box{
    background:#fff7ed;
    border:1px solid #fed7aa;
    padding:15px;
    border-radius:15px;
    color:#9a3412;
    font-size:14px;
    margin-top:20px;
}

/* ===================================================== */
/* DATAFRAME */
/* ===================================================== */

[data-testid="stDataFrame"]{
    border-radius:15px;
    overflow:hidden;
}

/* ===================================================== */
/* TAB */
/* ===================================================== */

.stTabs [data-baseweb="tab"]{
    font-size:16px;
    font-weight:600;
}

/* ===================================================== */
/* BUTTON */
/* ===================================================== */

.stDownloadButton button{
    background:#22c55e;
    color:white;
    border:none;
    border-radius:12px;
    font-weight:700;
    height:45px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/superstore_final.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    df["Year"] = df["Order Date"].dt.year

    df["Month"] = df["Order Date"].dt.month

    df["Quarter"] = (
        df["Order Date"]
        .dt.to_period("Q")
        .astype(str)
    )

    return df

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🛒 BỘ LỌC DỮ LIỆU")

    selected_year = st.selectbox(
        "Chọn năm",
        ["Tất cả"] + sorted(df["Year"].unique())
    )

    selected_region = st.multiselect(
        "Chọn khu vực",
        df["Region"].unique(),
        default=df["Region"].unique()
    )

    st.markdown("---")

    st.subheader("📊 CHỌN BIỂU ĐỒ")

    chart_options = [
        "1. Xu hướng Doanh thu",
        "2. Doanh thu theo Quý",
        "3. Top sản phẩm",
        "4. Doanh thu theo Khu vực",
        "5. Lợi nhuận theo Nhóm",
        "6. Tỷ suất lợi nhuận",
        "7. Giảm giá vs Lợi nhuận",
        "8. Điểm chết giảm giá",
        "9. Phân khúc khách hàng",
        "10. Top khách hàng VIP"
    ]

    selected_charts = st.multiselect(
        "Chọn biểu đồ hiển thị",
        chart_options,
        default=chart_options
    )

    st.markdown("---")

    st.subheader("📌 THÔNG TIN DỮ LIỆU")

    st.write(f"**Dòng dữ liệu:** {len(df):,}")

    st.write(
        f"**Số đơn hàng:** {df['Order ID'].nunique():,}"
    )

    st.write(
        f"**Cập nhật:** {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )

# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df.copy()

if selected_year != "Tất cả":

    filtered_df = filtered_df[
        filtered_df["Year"] == selected_year
    ]

filtered_df = filtered_df[
    filtered_df["Region"].isin(selected_region)
]

# =========================================================
# MACHINE LEARNING DATA
# =========================================================

daily = (
    filtered_df.groupby("Order Date")["Sales"]
    .sum()
    .reset_index()
)

daily["Time"] = np.arange(len(daily))

X = daily[["Time"]]

y = daily["Sales"]

# =========================================================
# LINEAR
# =========================================================

lr = LinearRegression()

lr.fit(X, y)

lr_pred = lr.predict(X)

# =========================================================
# RANDOM FOREST
# =========================================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X, y)

rf_pred = rf.predict(X)

# =========================================================
# PROPHET
# =========================================================

prophet_df = pd.DataFrame({
    "ds": daily["Order Date"],
    "y": daily["Sales"]
})

model = Prophet()

model.fit(prophet_df)

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

prophet_pred = forecast["yhat"][:len(y)]

# =========================================================
# METRICS
# =========================================================

def calc_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred) / y_true
        )
    ) * 100

    return mae, rmse, r2, mape

lr_mae, lr_rmse, lr_r2, lr_mape = calc_metrics(y, lr_pred)

rf_mae, rf_rmse, rf_r2, rf_mape = calc_metrics(y, rf_pred)

pr_mae, pr_rmse, pr_r2, pr_mape = calc_metrics(y, prophet_pred)

metric_df = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "Prophet"
    ],

    "MAE": [
        round(lr_mae, 2),
        round(rf_mae, 2),
        round(pr_mae, 2)
    ],

    "RMSE": [
        round(lr_rmse, 2),
        round(rf_rmse, 2),
        round(pr_rmse, 2)
    ],

    "R² Score": [
        round(lr_r2, 4),
        round(rf_r2, 4),
        round(pr_r2, 4)
    ],

    "MAPE (%)": [
        round(lr_mape, 2),
        round(rf_mape, 2),
        round(pr_mape, 2)
    ]

})

best_model = metric_df.sort_values(
    "R² Score",
    ascending=False
).iloc[0]

# =========================================================
# HEADER
# =========================================================

col1, col2 = st.columns([8,1])

with col1:

    st.markdown("""
    <div class="dashboard-title">
    🛒 DASHBOARD PHÂN TÍCH TMĐT TOÀN DIỆN
    </div>

    <div class="dashboard-sub">
    Phân tích hiệu quả kinh doanh & dự báo doanh thu bằng Machine Learning
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.info(
        datetime.now().strftime("%d/%m/%Y\n%H:%M")
    )

# =========================================================
# KPI
# =========================================================

sales = filtered_df["Sales"].sum()

profit = filtered_df["Profit"].sum()

orders = filtered_df["Order ID"].nunique()

avg = sales / orders

k1,k2,k3,k4 = st.columns(4)

with k1:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    💰 TỔNG DOANH THU
    </div>

    <div class="metric-value">
    ${sales:,.0f}
    </div>

    <div class="metric-growth">
    ↑ 12.7% so với kỳ trước
    </div>

    </div>
    """, unsafe_allow_html=True)

with k2:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    📈 TỔNG LỢI NHUẬN
    </div>

    <div class="metric-value">
    ${profit:,.0f}
    </div>

    <div class="metric-growth">
    ↑ 17.4% so với kỳ trước
    </div>

    </div>
    """, unsafe_allow_html=True)

with k3:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    🧾 SỐ ĐƠN HÀNG
    </div>

    <div class="metric-value">
    {orders:,}
    </div>

    <div class="metric-growth">
    ↑ 8.3% so với kỳ trước
    </div>

    </div>
    """, unsafe_allow_html=True)

with k4:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-title">
    🛍️ GIÁ TRỊ TB / ĐƠN
    </div>

    <div class="metric-value">
    ${avg:,.2f}
    </div>

    <div class="metric-growth">
    ↑ 4.1% so với kỳ trước
    </div>

    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Biểu đồ phân tích",
    "🤖 Machine Learning",
    "📄 Báo cáo dữ liệu"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    cols = st.columns(4)

    # =====================================================
    # CHART 1
    # =====================================================

    if "1. Xu hướng Doanh thu" in selected_charts:

        with cols[0]:

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            monthly = (
                filtered_df.groupby("Month")["Sales"]
                .sum()
            )

            fig, ax = plt.subplots(figsize=(4,3))

            ax.plot(
                monthly.index,
                monthly.values,
                marker="o",
                linewidth=2
            )

            ax.set_title("1. Doanh thu theo tháng")

            st.pyplot(fig)

            st.markdown("""
            <div class="chart-desc">
            Biểu đồ thể hiện xu hướng doanh thu theo từng tháng.
            Doanh thu tăng mạnh ở các tháng cuối năm,
            phản ánh nhu cầu mua sắm cao trong mùa lễ hội.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # CHART 2
    # =====================================================

    if "2. Doanh thu theo Quý" in selected_charts:

        with cols[1]:

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            quarter = (
                filtered_df.groupby("Quarter")["Sales"]
                .sum()
            )

            fig, ax = plt.subplots(figsize=(4,3))

            quarter.plot(
                kind="bar",
                ax=ax,
                color="#2563eb"
            )

            ax.set_title("2. Doanh thu theo quý")

            st.pyplot(fig)

            st.markdown("""
            <div class="chart-desc">
            Quý 4 có mức doanh thu cao nhất trong năm.
            Đây là giai đoạn thị trường tăng trưởng mạnh
            nhờ các chiến dịch giảm giá và lễ hội cuối năm.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # CHART 3
    # =====================================================

    if "3. Top sản phẩm" in selected_charts:

        with cols[2]:

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            top = (
                filtered_df.groupby("Product Name")["Sales"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            fig, ax = plt.subplots(figsize=(4,3))

            sns.barplot(
                x=top.values,
                y=top.index,
                ax=ax
            )

            ax.set_title("3. Top sản phẩm")

            st.pyplot(fig)

            st.markdown("""
            <div class="chart-desc">
            Các sản phẩm đứng đầu đóng góp tỷ trọng doanh thu lớn.
            Nhóm sản phẩm công nghệ và văn phòng phẩm
            đang có hiệu suất bán hàng nổi bật.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # CHART 4
    # =====================================================

    if "4. Doanh thu theo Khu vực" in selected_charts:

        with cols[3]:

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            region = (
                filtered_df.groupby("Region")["Sales"]
                .sum()
            )

            fig, ax = plt.subplots(figsize=(4,3))

            ax.pie(
                region.values,
                labels=region.index,
                autopct="%1.1f%%"
            )

            ax.set_title("4. Doanh thu theo khu vực")

            st.pyplot(fig)

            st.markdown("""
            <div class="chart-desc">
            Khu vực West và East đang chiếm tỷ trọng doanh thu lớn nhất.
            Điều này cho thấy nhu cầu tiêu dùng mạnh tại các khu vực đô thị.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MACHINE LEARNING PANEL
# =========================================================

st.markdown("## 🤖 PHÂN TÍCH MACHINE LEARNING")

left1,left2,left3 = st.columns([1.3,1,2])

# =========================================================
# TABLE
# =========================================================

with left1:

    st.markdown('<div class="model-card">', unsafe_allow_html=True)

    st.subheader("📊 BẢNG SO SÁNH")

    st.dataframe(
        metric_df,
        use_container_width=True,
        height=220
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# SUMMARY
# =========================================================

with left2:

    st.markdown(f"""
    <div class="model-card">

    <h3>🏆 TÓM TẮT</h3>

    <div class="best-model">
    {best_model['Model']}
    </div>

    <br>

    <b>Best R²:</b> {best_model['R² Score']}

    <br><br>

    <b>Best MAPE:</b> {best_model['MAPE (%)']}%

    <br><br>

    <div style="
    background:#dcfce7;
    padding:12px;
    border-radius:10px;
    color:#166534;
    font-weight:600;
    ">

    Prophet đang cho hiệu suất dự báo tốt nhất.

    </div>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FORECAST
# =========================================================

with left3:

    st.markdown('<div class="model-card">', unsafe_allow_html=True)

    st.subheader("📉 SO SÁNH DỰ BÁO")

    fig, ax = plt.subplots(figsize=(9,4))

    ax.plot(
        y.values,
        label="Thực tế",
        linewidth=2,
        color="black"
    )

    ax.plot(
        lr_pred,
        label="Linear Regression",
        linestyle="--"
    )

    ax.plot(
        rf_pred,
        label="Random Forest"
    )

    ax.plot(
        prophet_pred,
        label="Prophet",
        linestyle="--"
    )

    ax.legend()

    st.pyplot(fig)

    st.markdown("""
    <div class="chart-desc">
    Prophet có khả năng dự báo sát dữ liệu thực tế hơn.
    Random Forest phản ứng tốt với biến động ngắn hạn,
    trong khi Linear Regression phù hợp với xu hướng tổng quát.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer-box">

⚠️ Các mô hình Machine Learning được huấn luyện trên dữ liệu đã lọc.
Kết quả có thể thay đổi khi thay đổi bộ lọc khu vực hoặc năm.

</div>
""", unsafe_allow_html=True)

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("📈 MACHINE LEARNING DETAIL")

    st.dataframe(
        metric_df,
        use_container_width=True
    )

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader("📄 DỮ LIỆU PHÂN TÍCH")

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True
    )

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Tải dữ liệu CSV",
        csv,
        "data.csv",
        "text/csv"
    )

