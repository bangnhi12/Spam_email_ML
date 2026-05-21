import re
import joblib
import streamlit as st

MODEL_PATH = "spam_filter_model.pkl"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " link ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Spam Email Filter",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Công cụ lọc Email Spam sử dụng Machine Learning")

st.write(
    "Nhập nội dung email để hệ thống dự đoán email đó là **Spam** hay **Ham**."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Chưa tìm thấy file spam_filter_model.pkl. Hãy chạy `python train_model.py` trước.")
    st.stop()

email_content = st.text_area(
    "Nội dung email:",
    height=180,
    placeholder="Ví dụ: Congratulations! You won a free prize. Click here now!"
)

if st.button("Kiểm tra email"):
    if not email_content.strip():
        st.warning("Vui lòng nhập nội dung email.")
    else:
        cleaned = clean_text(email_content)
        prediction = model.predict([cleaned])[0]

        proba = model.predict_proba([cleaned])[0]
        labels = model.classes_
        score = dict(zip(labels, proba))

        spam_score = score.get("spam", 0)
        ham_score = score.get("ham", 0)

        st.subheader("Kết quả dự đoán")

        if prediction == "spam":
            st.error("Email này có khả năng là SPAM.")
        else:
            st.success("Email này có khả năng là email hợp lệ.")

        st.write(f"**Xác suất Ham:** {ham_score:.2%}")
        st.write(f"**Xác suất Spam:** {spam_score:.2%}")

        st.progress(float(spam_score))

st.divider()


