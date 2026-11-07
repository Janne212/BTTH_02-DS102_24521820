import numpy as np
from pathlib import Path
from data_utils import LoadImagesFromFolder
from feature_extraction import ExtractFeature
from logistic_regression import LogisticRegression

train_path = r"DS102\BTTH 02\data\train"
test_path = r"DS102\BTTH 02\data\test"

print("Đang đọc dữ liệu từ tập train và test...")
train_images, train_labels, class_names = LoadImagesFromFolder(train_path)
test_images, test_labels, test_class_names = LoadImagesFromFolder(test_path)

train_set = (train_images, train_labels) 
test_set = (test_images, test_labels)

def filter_data(data, criterion, class_names, binary_labels):
    images, labels = data

    class_index = class_names.index(criterion)
    selected_images = [
        ExtractFeature(image)
        for image, label in zip(images, labels)
        if label == class_index
    ]
    selected_labels = np.full(
        len(selected_images), binary_labels.index(criterion), dtype=int
    )

    return np.asarray(selected_images), selected_labels

def merge_data(data_set: list):
    images, labels = data_set[0]

    for ds in data_set[1:]:
        i, l = ds
        images = np.concatenate([images, i], axis=0)
        labels = np.concatenate([labels, l], axis=0)

    return (images, labels)

print("Đang rút trích đặc trưng từ ảnh và lọc dữ liệu cho bài toán phân loại nhị phân...")
binary_labels = ["architecture", "art"]

train_set_0 = filter_data(train_set, "architecture", class_names, binary_labels)
train_set_1 = filter_data(train_set, "art", class_names, binary_labels)
train_set = merge_data([
    train_set_0,
    train_set_1
])

test_set_0 = filter_data(test_set, "architecture", test_class_names, binary_labels)
test_set_1 = filter_data(test_set, "art", test_class_names, binary_labels)
test_set = merge_data([
    test_set_0,
    test_set_1
])

if __name__ == "__main__":
    train_images, train_labels = train_set
    model = LogisticRegression(learning_rate=0.2, epochs=100)
    model.fit(train_images, train_labels)

    test_images, test_labels = test_set
    metrics = model.evaluate(test_images, test_labels)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    output_dir = Path(__file__).resolve().parent.parent / "visualize"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "loss_history_logistic.npy", model.loss_history)
