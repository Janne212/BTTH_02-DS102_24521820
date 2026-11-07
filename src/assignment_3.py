import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils import LoadImagesFromFolder
from feature_extraction import ExtractFeature

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss


print("Đang đọc dữ liệu từ tập train và test...")
train_path = r"DS102\BTTH 02\data\train"
test_path = r"DS102\BTTH 02\data\test"

train_images, train_labels, class_names = LoadImagesFromFolder(train_path)
test_images, test_labels, test_class_names = LoadImagesFromFolder(test_path)

print("Đang rút trích đặc trưng ảnh...")
X_train = np.array([ExtractFeature(image) for image in train_images])
X_test = np.array([ExtractFeature(image) for image in test_images])

y_train = np.array(train_labels)
y_test = np.array(test_labels)

def to_binary(X, y, names, binary_labels):
    ids = [names.index(c) for c in binary_labels]
    mask = np.isin(y, ids)
    return X[mask], (y[mask] == ids[1]).astype(int)

def fit_logistic_regression(X, y, epochs=100):
    logistic = LogisticRegression(fit_intercept=False, max_iter=epochs)
    loss_history = []

    with tqdm(range(epochs)) as pb:
        for epoch in pb:
            logistic.fit(X, y)
            loss_value = log_loss(y, logistic.predict_proba(X), labels=logistic.classes_)
            loss_history.append(loss_value)

            pb.set_description(
                f"Training | Epoch {epoch + 1}/{epochs} | Loss: {loss_value:.4f}"
            )

    return logistic, loss_history

def fit_softmax_regression(X, y, epochs=100):
    softmax = LogisticRegression(fit_intercept=False, solver='lbfgs', max_iter=epochs)
    loss_history = []

    with tqdm(range(epochs)) as pb:
        for epoch in pb:
            softmax.fit(X, y)
            loss_value = log_loss(y, softmax.predict_proba(X), labels=softmax.classes_)
            loss_history.append(loss_value)

            pb.set_description(
                f"Training | Epoch {epoch + 1}/{epochs} | Loss: {loss_value:.4f}"
            )
            
    return softmax, loss_history

def evaluate_model(y_test, y_predicted, average="macro"):
    print(f"Accuracy:  {accuracy_score(y_test, y_predicted):.4f}")
    print(f"Precision: {precision_score(y_test, y_predicted, average=average):.4f}")
    print(f"Recall:    {recall_score(y_test, y_predicted, average=average):.4f}")
    print(f"F1 score:  {f1_score(y_test, y_predicted, average=average):.4f}")

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / "visualize"
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_labels = ["architecture", "art"]
    X_train_bin, y_train_bin = to_binary(X_train, y_train, class_names, binary_labels)
    X_test_bin, y_test_bin = to_binary(X_test, y_test, test_class_names, binary_labels)

    print("Đang chạy Logistic Regression...")
    logistic, l_loss_history = fit_logistic_regression(X_train_bin, y_train_bin)
    evaluate_model(y_test_bin, logistic.predict(X_test_bin), average="binary")
    np.save(output_dir / "library_loss_history_logistic.npy", l_loss_history)

    print("Đang chạy Softmax Regression...")
    softmax, s_loss_history = fit_softmax_regression(X_train, y_train)
    evaluate_model(y_test, softmax.predict(X_test), average="macro")
    np.save(output_dir / "library_loss_history_softmax.npy", s_loss_history)