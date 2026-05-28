# HE THONG PHAN TICH DU LIEU THUONG MAI DIEN TU VA DU BAO DOANH THU

## 1. Gioi thieu

Du an xay dung mot he thong phan tich du lieu thuong mai dien tu hoan chinh, bao gom cac thanh phan:

* Lam sach du lieu
* Phan tich du lieu
* Truc quan hoa
* Dashboard tuong tac
* Du bao doanh thu bang Machine Learning

He thong giup doanh nghiep:

* Hieu ro hieu suat kinh doanh
* Phan tich hanh vi khach hang
* Dua ra du bao doanh thu
* Ho tro quyet dinh chien luoc

---

## 2. Cau truc thu muc

```
CHUYEN DE 3/
│
├── data/
│   ├── superstore_cleaned.csv
│   └── superstore_final.csv
│
├── images/
│
├── reports/
│
├── analysis.py
├── app.py
├── charts.py
├── clean.py
├── model.py
│
└── README.md
```

Mo ta:

* data: chua du lieu dau vao va du lieu sau khi lam sach
* images: luu tat ca bieu do duoc sinh ra tu qua trinh phan tich
* reports: luu bao cao PDF va ket qua danh gia mo hinh
* clean.py: xu ly va lam sach du lieu
* analysis.py: phan tich du lieu va danh gia mo hinh
* charts.py: dinh nghia cac ham ve bieu do
* app.py: dashboard Streamlit
* model.py: mo rong xu ly mo hinh (neu su dung)
* README.md: mo ta du an

---

## 3. Dataset

Du lieu su dung: Superstore Dataset

Thong tin:

* Du lieu giao dich ban hang
* Bao gom cac truong:

  * Order ID
  * Order Date
  * Product Name
  * Category
  * Sales
  * Profit
  * Customer Name
  * Region
  * Segment

Sau khi lam sach:

* Du lieu hop le, khong co gia tri null quan trong
* Co them cac cot thoi gian phuc vu phan tich

---

## 4. Cong nghe su dung

* Python: ngon ngu lap trinh chinh
* Pandas: xu ly du lieu
* NumPy: tinh toan so hoc
* Matplotlib, Seaborn: truc quan hoa du lieu
* Scikit-learn: xay dung mo hinh machine learning
* Prophet: du bao chuoi thoi gian
* Streamlit: xay dung dashboard
* ReportLab: xuat bao cao PDF

---

## 5. Quy trinh xu ly du lieu

### 5.1 Lam sach du lieu (clean.py)

Cac buoc chinh:

* Xoa du lieu trung lap
* Xoa du lieu thieu
* Chuan hoa dinh dang du lieu:

  * Order Date -> datetime
  * Sales -> numeric
* Loai bo du lieu khong hop le (Sales <= 0)

Tao them cac cot:

* Year: phuc vu loc du lieu
* Month: phuc vu phan tich xu huong

Ket qua:

* File superstore_final.csv
* Anh bang du lieu
* Bao cao PDF tu dong

---

### 5.2 Phan tich du lieu (analysis.py)

Cac noi dung phan tich:

* Doanh thu theo thang
* Top san pham doanh thu cao
* Doanh thu theo khu vuc
* Doanh thu theo quy
* Loi nhuan theo danh muc
* Ty suat loi nhuan
* Anh huong cua discount den profit
* Phan khuc khach hang
* Top khach hang

Tat ca bieu do duoc luu vao thu muc images.

---

### 5.3 Machine Learning

Su dung 2 mo hinh chinh:

* Linear Regression
* Random Forest

Quy trinh:

* Tao bien thoi gian (Time)
* Train/Test split
* Train model
* Du doan doanh thu

Danh gia mo hinh:

* MAE
* RMSE
* R2 Score

Ket qua duoc luu vao:

reports/model_metrics.csv

---

## 6. Dashboard (app.py)

Dashboard duoc xay dung bang Streamlit voi cac tinh nang:

### 6.1 Bo loc du lieu

* Loc theo nam
* Loc theo khu vuc
* Lua chon bieu do hien thi

---

### 6.2 KPI tong quan

* Tong doanh thu
* Tong loi nhuan
* So don hang
* Gia tri trung binh moi don

---

### 6.3 Bieu do phan tich

* Doanh thu theo thang
* Doanh thu theo quy
* Top san pham
* Doanh thu theo khu vuc

---

### 6.4 Machine Learning

Su dung 3 mo hinh:

* Linear Regression
* Random Forest
* Prophet

Dashboard cung cap:

* So sanh mo hinh
* Hien thi chi so danh gia
* Bieu do du bao

---

### 6.5 Bao cao du lieu

* Hien thi bang du lieu
* Tai file CSV truc tiep

---

## 7. Luong he thong

```
Du lieu goc
    ->
Lam sach du lieu (clean.py)
    ->
Du lieu sach
    ->
Phan tich va truc quan (analysis.py, charts.py)
    ->
Machine Learning
    ->
Dashboard (app.py)
    ->
Bao cao (PDF, CSV)
```

---

## 8. Huong dan chay du an

### Buoc 1: Cai dat thu vien

```
pip install pandas numpy matplotlib seaborn scikit-learn prophet streamlit reportlab
```

---

### Buoc 2: Lam sach du lieu

```
python clean.py
```

---

### Buoc 3: Phan tich du lieu

```
python analysis.py
```

---

### Buoc 4: Chay dashboard

```
streamlit run app.py
```

---

## 9. Ket qua dat duoc

* Xay dung pipeline xu ly du lieu hoan chinh
* Truc quan hoa du lieu ro rang
* Xay dung mo hinh du bao doanh thu
* Dashboard tuong tac chuyen nghiep
* Xuat bao cao tu dong

---

## 10. Mo rong

Co the phat trien them:

* Ket noi co so du lieu thuc te
* Trien khai len cloud
* Them mo hinh deep learning
* Xay dung API du lieu

---

## 11. Ket luan

Du an mo ta day du quy trinh phan tich du lieu trong thuc te:

Data Cleaning -> Analysis -> Visualization -> Modeling -> Dashboard -> Reporting

He thong co the ap dung truc tiep trong bai toan thuong mai dien tu hoac mo rong sang cac linh vuc khac.
