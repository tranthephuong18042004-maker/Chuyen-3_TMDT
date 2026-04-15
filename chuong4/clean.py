#=======làm sạch dữ liệu =========================

import pandas as pd

def clean_data(input_file, output_file):
    print("Đang đọc dữ liệu...")

    df = pd.read_csv(input_file, encoding='utf-8-sig')

    print("Dữ liệu ban đầu:", df.shape)

    #===== xử lý dữ liệu =========================

    # Xóa khoảng trắng tên cột
    df.columns = df.columns.str.strip()

    # Xóa dòng trùng
    df = df.drop_duplicates()

    # Xóa dữ liệu thiếu
    df = df.dropna()

   #====== xử lý kiểu dữ liệu =========================

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    # Xóa dòng lỗi
    df = df.dropna(subset=["Order Date", "Sales"])

    # Xóa dữ liệu bất thường
    df = df[df["Sales"] > 0]

   #====== FEATURE ENGINEERING =========================

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month

    print("Sau khi làm sạch:", df.shape)

    #====== lưu file ==========================

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Đã lưu file sạch: {output_file}")


#===== chạy hàm làm sạch dữ liệu =========================
if __name__ == "__main__":
    clean_data("superstore_cleaned.csv", "superstore_final.csv")