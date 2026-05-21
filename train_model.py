import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

DATASET_PATH = "spam.csv"
MODEL_PATH = "spam_filter_model.pkl"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " link ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)

    if "label" not in data.columns or "message" not in data.columns:
        raise ValueError("File CSV phải có 2 cột: label,message")

    data = data.dropna(subset=["label", "message"])
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data["message"] = data["message"].astype(str).apply(clean_text)

    valid_labels = {"spam", "ham"}
    data = data[data["label"].isin(valid_labels)]

    if data.empty:
        raise ValueError("Dataset rỗng hoặc nhãn không hợp lệ. Nhãn phải là spam hoặc ham.")

    return data


def train():
    data = load_dataset(DATASET_PATH)

    X = data["message"]
    y = data["label"]

    label_counts = y.value_counts()
    use_stratify = y if label_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=use_stratify
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1
        )),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print()
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    disp.plot()
    plt.title("Confusion Matrix - Spam Filter")
    plt.savefig("confusion_matrix.png")
    plt.close()

    joblib.dump(model, MODEL_PATH)

    print("=" * 60)
    print(f"Đã lưu model: {MODEL_PATH}")
    print("Đã lưu biểu đồ: confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    train()
