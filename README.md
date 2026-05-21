# Lọc Email Spam sử dụng Machine Learning

## 1. Chức năng

Dự án phân loại email thành 2 nhóm:

- `spam`: email rác, quảng cáo, lừa đảo, nội dung không mong muốn
- `ham`: email hợp lệ, bình thường

Mô hình sử dụng:

- TF-IDF để chuyển văn bản thành vector số
- Multinomial Naive Bayes để phân loại email

## 2. Cài thư viện

```bash
pip install -r requirements.txt
```

## 3. Huấn luyện mô hình

```bash
python train_model.py
```

Sau khi chạy, chương trình tạo:

```text
spam_filter_model.pkl
confusion_matrix.png
```

## 4. Dự đoán bằng terminal

```bash
python predict_cli.py
```

## 5. Chạy giao diện web

```bash
streamlit run app.py
```

## 6. Cấu trúc file

```text

├── spam.csv
├── train_model.py
├── predict_cli.py
├── app.py
├── requirements.txt
└── README.md
```

