import numpy as np
import cv2 as cv
from pathlib import Path
from data_utils import LoadImagesFromFolder
from feature_extraction import ExtractFeature
from softmax_regression import SoftmaxRegression

def convert_to_one_hot(y: list[int] | np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    K = np.max(y) + 1
    N = y.shape[0]
    oh_y = np.zeros((N, K))
    oh_y[np.arange(N), y] = 1

    return oh_y

train_path = r"DS102\BTTH 02\data\train"
test_path = r"DS102\BTTH 02\data\test"

print("Đang đọc dữ liệu từ tập train và test...")
train_images, train_labels, class_names = LoadImagesFromFolder(train_path)
test_images, test_labels, test_class_names = LoadImagesFromFolder(test_path)

print("Đang rút trích đặc trưng ảnh...")
X_train = np.array([ExtractFeature(image) for image in train_images])
X_test = np.array([ExtractFeature(image) for image in test_images])

y_train = convert_to_one_hot(train_labels)
y_test = convert_to_one_hot(test_labels)

train_set = (X_train, y_train) 
test_set = (X_test, y_test)

if __name__ == "__main__":
    train_images, train_labels = train_set
    model = SoftmaxRegression(epochs=100, learning_rate=0.2)
    model.fit(train_images, train_labels)

    test_images, test_labels = test_set
    metrics = model.evaluate(test_images, test_labels)
    for key, value in metrics.items():
        print(f"{key}: {value}" )
        
    output_dir = Path(__file__).resolve().parent.parent / "visualize"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "loss_history_softmax.npy", model.loss_history)