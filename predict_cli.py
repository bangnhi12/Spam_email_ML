import re
import joblib

MODEL_PATH = "spam_filter_model.pkl"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " link ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Chưa có model. Hãy chạy trước:")
        print("python train_model.py")
        return

    print("=" * 60)
    print("CÔNG CỤ KIỂM TRA EMAIL SPAM")
    print("=" * 60)

    while True:
        email = input("\nNhập nội dung email, hoặc nhập exit để thoát:\n> ")

        if email.lower().strip() == "exit":
            break

        cleaned = clean_text(email)
        prediction = model.predict([cleaned])[0]

        proba = model.predict_proba([cleaned])[0]
        labels = model.classes_
        score = dict(zip(labels, proba))

        spam_score = score.get("spam", 0)
        ham_score = score.get("ham", 0)

        print("\nKết quả:", prediction.upper())
        print(f"Xác suất HAM : {ham_score:.2%}")
        print(f"Xác suất SPAM: {spam_score:.2%}")


if __name__ == "__main__":
    main()
