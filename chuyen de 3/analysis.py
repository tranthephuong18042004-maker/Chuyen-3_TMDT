# =========================================================
# ANALYSIS - TMĐT (FULL BÁO CÁO CUỐI KỲ)
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as ticker

# Thiết lập style chung
sns.set(style="whitegrid")

# =========================
# LOAD DATA (FIX PATH)
# =========================
file_path = os.path.join("data", "superstore_final.csv")

if not os.path.exists(file_path):
    print("❌ Không tìm thấy file:", file_path)
    exit()

df = pd.read_csv(file_path)
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Tạo thư mục lưu ảnh (nếu chưa có)
os.makedirs("images", exist_ok=True)

# Thêm các cột phục vụ cho phân tích VIP phía sau
df["Quarter_Year"] = df["Order Date"].dt.to_period("Q").astype(str)
if "Profit" in df.columns and "Sales" in df.columns:
    df["Profit Margin (%)"] = (df["Profit"] / df["Sales"]) * 100

print("Dữ liệu:", df.shape)

# =========================
# 1. DOANH THU THEO THÁNG
# =========================
monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(10,5))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o")
plt.title("Xu hướng doanh thu theo tháng")
plt.xlabel("Tháng")
plt.ylabel("Doanh thu")

plt.savefig("images/trend.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 2. TOP SẢN PHẨM
# =========================
top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(8,5))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top sản phẩm bán chạy")

plt.savefig("images/top.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 3. DOANH THU THEO KHU VỰC
# =========================
region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(6,6))
plt.pie(region_sales.values, labels=region_sales.index, autopct="%1.1f%%")
plt.title("Doanh thu theo khu vực")

plt.savefig("images/region.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 4. MODEL DỰ BÁO
# =========================
df = df.sort_values("Order Date")
df["Time"] = np.arange(len(df))

X = df[["Time"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

# Đánh giá
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))

print("\nĐánh giá model:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Vẽ
plt.figure(figsize=(10,5))
plt.plot(y.values, label="Thực tế")
plt.plot(pred, label="Dự đoán")
plt.legend()
plt.title("Dự báo doanh thu")

plt.savefig("images/forecast.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 5. LỢI NHUẬN THEO DANH MỤC (CATEGORY & SUB-CATEGORY)
# =========================
plt.figure(figsize=(12, 6))
# Tính tổng lợi nhuận theo Sub-Category và sắp xếp
subcat_profit = df.groupby(["Category", "Sub-Category"])["Profit"].sum().reset_index()
subcat_profit = subcat_profit.sort_values(by="Profit", ascending=False)

# Vẽ biểu đồ ngang cho dễ đọc tên
sns.barplot(data=subcat_profit, x="Profit", y="Sub-Category", hue="Category", dodge=False)
plt.title("Lợi nhuận ròng theo Từng nhóm Sản phẩm (Sub-Category)", fontsize=14, pad=15)
plt.xlabel("Tổng Lợi Nhuận ($)")
plt.ylabel("Nhóm Sản Phẩm")
plt.axvline(0, color='black', linestyle='-', linewidth=1) # Kẻ đường số 0 để thấy rõ phần Lỗ
plt.tight_layout()

plt.savefig("images/profit_subcategory.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 6. TÁC ĐỘNG CỦA GIẢM GIÁ (DISCOUNT) ĐẾN LỢI NHUẬN
# =========================
plt.figure(figsize=(9, 5))
# Dùng lineplot để xem xu hướng Lợi nhuận thay đổi ra sao khi Discount tăng
sns.lineplot(data=df, x="Discount", y="Profit", errorbar=None, marker="o", color="red", linewidth=2)
plt.title("Mối tương quan giữa Tỷ lệ Giảm giá và Lợi nhuận trung bình", fontsize=14, pad=15)
plt.xlabel("Tỷ lệ Giảm giá (Discount)")
plt.ylabel("Lợi nhuận Trung bình ($)")
plt.axhline(0, color='black', linestyle='--', linewidth=1.5) # Đường ranh giới hòa vốn
plt.tight_layout()

plt.savefig("images/discount_profit.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 7. HIỆU QUẢ THEO PHÂN KHÚC KHÁCH HÀNG (SEGMENT)
# =========================
# Gộp nhóm để lấy Doanh thu và Lợi nhuận
segment_stats = df.groupby("Segment")[["Sales", "Profit"]].sum().reset_index()

# Chuyển đổi cấu trúc data để vẽ biểu đồ cột kép (Grouped Bar Chart)
segment_melted = segment_stats.melt(id_vars="Segment", var_name="Chỉ số", value_name="Giá trị")

plt.figure(figsize=(8, 5))
sns.barplot(data=segment_melted, x="Segment", y="Giá trị", hue="Chỉ số", palette=["#3498db", "#2ecc71"])
plt.title("So sánh Doanh thu và Lợi nhuận theo Phân khúc Khách hàng", fontsize=14, pad=15)
plt.xlabel("Phân khúc (Segment)")
plt.ylabel("Tổng Giá trị ($)")
plt.tight_layout()

plt.savefig("images/segment_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# CÁC BIỂU ĐỒ CHUYÊN SÂU BỔ SUNG (VIP PRO)
# =========================================================

# =========================
# 8. TĂNG TRƯỞNG DOANH THU THEO QUÝ (TÍNH MÙA VỤ)
# =========================
plt.figure(figsize=(14, 6))
quarterly_sales = df.groupby("Quarter_Year")["Sales"].sum().reset_index()

ax1 = sns.barplot(data=quarterly_sales, x="Quarter_Year", y="Sales", color="#3498db")
plt.title("Biến động Doanh thu theo Quý (Phân tích tính Mùa vụ)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Quý - Năm", fontsize=12)
plt.ylabel("Tổng Doanh Thu ($)", fontsize=12)
plt.xticks(rotation=45)

# Gắn nhãn số liệu (định dạng K cho nghìn đô)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()/1000:.0f}K', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig("images/vip1_quarterly_trend.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 9. TỶ SUẤT LỢI NHUẬN (PROFIT MARGIN) THEO DANH MỤC
# =========================
plt.figure(figsize=(14, 8))
margin_df = df.groupby("Sub-Category")[["Sales", "Profit"]].sum()
margin_df["Margin"] = (margin_df["Profit"] / margin_df["Sales"]) * 100
margin_df = margin_df.sort_values("Margin", ascending=False).reset_index()

# Tạo mảng màu: Xanh nếu Margin > 0, Đỏ nếu Margin < 0
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in margin_df["Margin"]]

ax2 = sns.barplot(data=margin_df, x="Margin", y="Sub-Category", palette=colors)
plt.title("Hiệu quả Sinh lời: Tỷ suất Lợi nhuận (%) theo Nhóm Sản phẩm", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Tỷ suất Lợi nhuận (%)", fontsize=12)
plt.ylabel("Nhóm Sản phẩm", fontsize=12)
plt.axvline(0, color='black', linewidth=1.5)

# Gắn nhãn % lên thanh biểu đồ
for p in ax2.patches:
    width = p.get_width()
    label_pos = width + 1 if width > 0 else width - 1
    ha = 'left' if width > 0 else 'right'
    ax2.annotate(f'{width:.1f}%', 
                 (label_pos, p.get_y() + p.get_height() / 2), 
                 ha=ha, va='center', fontsize=10, fontweight='bold', color='#2c3e50')

plt.tight_layout()
plt.savefig("images/vip2_profit_margin.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 10. ĐIỂM CHẾT GIẢM GIÁ (DISCOUNT vs AVERAGE PROFIT)
# =========================
plt.figure(figsize=(12, 6))
discount_profit = df.groupby("Discount")["Profit"].mean().reset_index()

ax3 = sns.barplot(data=discount_profit, x="Discount", y="Profit", 
                  palette=['#f39c12' if p > 0 else '#c0392b' for p in discount_profit["Profit"]])
plt.title("Cái bẫy Giảm giá: Mức Discount nào khiến Doanh nghiệp Lỗ?", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Mức Giảm Giá (Discount %)", fontsize=12)
plt.ylabel("Lợi nhuận Trung bình mỗi đơn hàng ($)", fontsize=12)
plt.axhline(0, color='black', linewidth=1.5)

plt.tight_layout()
plt.savefig("images/vip3_discount_trap.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 11. TỪ ĐIỂN KHÁCH HÀNG VIP (TOP 10 CUSTOMERS)
# =========================
plt.figure(figsize=(14, 7))
top_customers = df.groupby("Customer Name")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False).head(10).reset_index()

# Biểu đồ cột kép kết hợp (Sales và Profit)
top_customers_melted = top_customers.melt(id_vars="Customer Name", var_name="Metric", value_name="Value")

ax4 = sns.barplot(data=top_customers_melted, x="Value", y="Customer Name", hue="Metric", palette=["#34495e", "#27ae60"])
plt.title("Đóng góp của Top 10 Khách hàng VIP (Doanh thu & Lợi nhuận)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Tổng Giá trị ($)", fontsize=12)
plt.ylabel("")

# Dán nhãn
for p in ax4.patches:
    width = p.get_width()
    if width > 0: # Chỉ in số lớn hơn 0
        ax4.annotate(f'${width:,.0f}', 
                     (width + 200, p.get_y() + p.get_height() / 2), 
                     ha='left', va='center', fontsize=9, fontweight='bold', color='black')

plt.legend(title="")
plt.tight_layout()
plt.savefig("images/vip4_top_customers.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Đã chạy xong! Toàn bộ 11 biểu đồ đã được lưu trong thư mục 'images'.")
