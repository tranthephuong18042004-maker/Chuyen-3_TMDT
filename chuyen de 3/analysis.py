# =========================================================
# ANALYSIS - TMĐT (BÁO CÁO PHÂN TÍCH & DỰ BÁO)
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

# =========================
# THIẾT LẬP STYLE
# =========================
sns.set(style="whitegrid")

# =========================
# TẠO THƯ MỤC
# =========================
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# =========================
# LOAD DATA
# =========================
file_path = os.path.join("data", "superstore_final.csv")

if not os.path.exists(file_path):
    print("❌ Không tìm thấy file:", file_path)
    exit()

df = pd.read_csv(file_path)

# =========================
# XỬ LÝ DỮ LIỆU
# =========================
df["Order Date"] = pd.to_datetime(df["Order Date"])

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Quarter_Year"] = df["Order Date"].dt.to_period("Q").astype(str)

if "Profit" in df.columns and "Sales" in df.columns:
    df["Profit Margin (%)"] = (
        df["Profit"] / df["Sales"]
    ) * 100

print("=" * 50)
print("DỮ LIỆU ĐÃ TẢI THÀNH CÔNG")
print("Kích thước dữ liệu:", df.shape)
print("=" * 50)

# =========================================================
# 1. DOANH THU THEO THÁNG
# =========================================================
monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(10, 5))

sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values,
    marker="o"
)

plt.title("Xu hướng doanh thu theo tháng")
plt.xlabel("Tháng")
plt.ylabel("Doanh thu")

plt.tight_layout()

plt.savefig(
    "images/01_monthly_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 2. TOP 10 SẢN PHẨM DOANH THU CAO
# =========================================================
top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title("Top 10 sản phẩm doanh thu cao")
plt.xlabel("Doanh thu")
plt.ylabel("Sản phẩm")

plt.tight_layout()

plt.savefig(
    "images/02_top_products.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 3. DOANH THU THEO KHU VỰC
# =========================================================
region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(7, 7))

plt.pie(
    region_sales.values,
    labels=region_sales.index,
    autopct="%1.1f%%"
)

plt.title("Tỷ trọng doanh thu theo khu vực")

plt.tight_layout()

plt.savefig(
    "images/03_region_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 4. MÔ HÌNH DỰ BÁO DOANH THU
# =========================================================

print("\n" + "=" * 50)
print("HUẤN LUYỆN MÔ HÌNH")
print("=" * 50)

# Sắp xếp theo thời gian
df = df.sort_values("Order Date")

# Tạo biến thời gian
df["Time"] = np.arange(len(df))

X = df[["Time"]]
y = df["Sales"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# LINEAR REGRESSION
# =========================================================
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

# Metric
lr_mae = mean_absolute_error(y_test, lr_pred)

lr_rmse = np.sqrt(
    mean_squared_error(y_test, lr_pred)
)

lr_r2 = r2_score(y_test, lr_pred)

# =========================================================
# RANDOM FOREST
# =========================================================
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# Metric
rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_pred)
)

rf_r2 = r2_score(y_test, rf_pred)

# =========================================================
# BẢNG SO SÁNH MODEL
# =========================================================
metrics_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        round(lr_mae, 2),
        round(rf_mae, 2)
    ],

    "RMSE": [
        round(lr_rmse, 2),
        round(rf_rmse, 2)
    ],

    "R² Score": [
        round(lr_r2, 4),
        round(rf_r2, 4)
    ]
})

print("\nBẢNG ĐÁNH GIÁ MÔ HÌNH")
print(metrics_df)

# Xuất CSV metric
metrics_df.to_csv(
    "reports/model_metrics.csv",
    index=False
)

# =========================================================
# BIỂU ĐỒ SO SÁNH MÔ HÌNH
# =========================================================
plt.figure(figsize=(12, 5))

plt.plot(
    y_test.values,
    label="Thực tế",
    linewidth=2
)

plt.plot(
    lr_pred,
    label="Linear Regression",
    linestyle="--"
)

plt.plot(
    rf_pred,
    label="Random Forest",
    alpha=0.8
)

plt.title("So sánh mô hình dự báo doanh thu")

plt.xlabel("Số mẫu")
plt.ylabel("Doanh thu")

plt.legend()

plt.tight_layout()

plt.savefig(
    "images/04_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 5. LỢI NHUẬN THEO DANH MỤC
# =========================================================
plt.figure(figsize=(12, 6))

subcat_profit = (
    df.groupby(["Category", "Sub-Category"])["Profit"]
    .sum()
    .reset_index()
)

subcat_profit = subcat_profit.sort_values(
    by="Profit",
    ascending=False
)

sns.barplot(
    data=subcat_profit,
    x="Profit",
    y="Sub-Category",
    hue="Category",
    dodge=False
)

plt.title("Lợi nhuận theo nhóm sản phẩm")

plt.xlabel("Tổng lợi nhuận")
plt.ylabel("Nhóm sản phẩm")

plt.axvline(
    0,
    color="black",
    linewidth=1
)

plt.tight_layout()

plt.savefig(
    "images/05_profit_subcategory.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 6. DISCOUNT VÀ PROFIT
# =========================================================
plt.figure(figsize=(10, 5))

sns.lineplot(
    data=df,
    x="Discount",
    y="Profit",
    errorbar=None,
    marker="o",
    linewidth=2
)

plt.title("Ảnh hưởng của Discount đến Profit")

plt.xlabel("Discount")
plt.ylabel("Profit")

plt.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.5
)

plt.tight_layout()

plt.savefig(
    "images/06_discount_profit.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 7. PHÂN KHÚC KHÁCH HÀNG
# =========================================================
segment_stats = (
    df.groupby("Segment")[["Sales", "Profit"]]
    .sum()
    .reset_index()
)

segment_melted = segment_stats.melt(
    id_vars="Segment",
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=segment_melted,
    x="Segment",
    y="Value",
    hue="Metric"
)

plt.title("Doanh thu và lợi nhuận theo phân khúc khách hàng")

plt.xlabel("Phân khúc")
plt.ylabel("Giá trị")

plt.tight_layout()

plt.savefig(
    "images/07_segment_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 8. DOANH THU THEO QUÝ
# =========================================================
plt.figure(figsize=(14, 6))

quarterly_sales = (
    df.groupby("Quarter_Year")["Sales"]
    .sum()
    .reset_index()
)

ax1 = sns.barplot(
    data=quarterly_sales,
    x="Quarter_Year",
    y="Sales"
)

plt.title("Biến động doanh thu theo quý")

plt.xlabel("Quý")
plt.ylabel("Doanh thu")

plt.xticks(rotation=45)

for p in ax1.patches:

    ax1.annotate(
        f'{p.get_height()/1000:.0f}K',

        (
            p.get_x() + p.get_width()/2,
            p.get_height()
        ),

        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "images/08_quarterly_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 9. PROFIT MARGIN
# =========================================================
plt.figure(figsize=(14, 8))

margin_df = (
    df.groupby("Sub-Category")[["Sales", "Profit"]]
    .sum()
)

margin_df["Margin"] = (
    margin_df["Profit"] /
    margin_df["Sales"]
) * 100

margin_df = (
    margin_df.sort_values(
        "Margin",
        ascending=False
    )
    .reset_index()
)

colors = [
    "#2ecc71" if x > 0 else "#e74c3c"
    for x in margin_df["Margin"]
]

ax2 = sns.barplot(
    data=margin_df,
    x="Margin",
    y="Sub-Category",
    palette=colors
)

plt.title("Tỷ suất lợi nhuận theo nhóm sản phẩm")

plt.xlabel("Profit Margin (%)")
plt.ylabel("Nhóm sản phẩm")

plt.axvline(
    0,
    color="black",
    linewidth=1.5
)

for p in ax2.patches:

    width = p.get_width()

    label_pos = (
        width + 1
        if width > 0
        else width - 1
    )

    ha = (
        "left"
        if width > 0
        else "right"
    )

    ax2.annotate(
        f"{width:.1f}%",

        (
            label_pos,
            p.get_y() + p.get_height()/2
        ),

        ha=ha,
        va="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "images/09_profit_margin.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 10. DISCOUNT VÀ PROFIT TRUNG BÌNH
# =========================================================
plt.figure(figsize=(12, 6))

discount_profit = (
    df.groupby("Discount")["Profit"]
    .mean()
    .reset_index()
)

ax3 = sns.barplot(
    data=discount_profit,
    x="Discount",
    y="Profit",
    palette=[
        "#3498db" if p > 0 else "#e74c3c"
        for p in discount_profit["Profit"]
    ]
)

plt.title("Profit trung bình theo mức Discount")

plt.xlabel("Discount")
plt.ylabel("Profit trung bình")

plt.axhline(
    0,
    color="black",
    linewidth=1.5
)

plt.tight_layout()

plt.savefig(
    "images/10_discount_avg_profit.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 11. TOP KHÁCH HÀNG
# =========================================================
plt.figure(figsize=(14, 7))

top_customers = (
    df.groupby("Customer Name")[["Sales", "Profit"]]
    .sum()
    .sort_values("Sales", ascending=False)
    .head(10)
    .reset_index()
)

top_customers_melted = top_customers.melt(
    id_vars="Customer Name",
    var_name="Metric",
    value_name="Value"
)

ax4 = sns.barplot(
    data=top_customers_melted,
    x="Value",
    y="Customer Name",
    hue="Metric"
)

plt.title("Top 10 khách hàng theo doanh thu")

plt.xlabel("Giá trị")
plt.ylabel("Khách hàng")

for p in ax4.patches:

    width = p.get_width()

    if width > 0:

        ax4.annotate(
            f'${width:,.0f}',

            (
                width + 200,
                p.get_y() + p.get_height()/2
            ),

            ha='left',
            va='center',
            fontsize=8
        )

plt.legend(title="")

plt.tight_layout()

plt.savefig(
    "images/11_top_customers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# HOÀN TẤT
# =========================================================
print("\n" + "=" * 50)
print("PHÂN TÍCH HOÀN TẤT")
print("Toàn bộ biểu đồ đã lưu trong thư mục images")
print("Metric mô hình đã lưu trong reports")
print("=" * 50)
