import argparse
import math
import shutil
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

test_size = 0.2
random_state = 42
img_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

def split_data(raw_path: Path, train_path: Path, test_path: Path, random_state: int = random_state):
    raw_path = raw_path.resolve()
    train_path = train_path.resolve()
    test_path = test_path.resolve()

    if not raw_path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu raw: {raw_path}")

    class_dirs = sorted(
        (path for path in raw_path.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not class_dirs:
        raise ValueError(f"Không tồn tại thư mục con trong dữ liệu raw: {raw_path}")

    # Chuyển dữ liệu từ cấu trúc thư mục thành danh sách để đưa vào train_test_split
    image_paths = []
    labels = []
    for class_dir in class_dirs:
        class_images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in img_extensions
        )
        image_paths.extend(class_images)
        labels.extend([class_dir.name] * len(class_images))

    # Chia dữ liệu train-test
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Kiểm tra và tạo các thư mục train và test nếu chưa tồn tại
    for output_path in (train_path, test_path):
        if output_path.exists() and not output_path.is_dir():
            raise NotADirectoryError(f"Đường dẫn không phải là thư mục: {output_path}")

    for output_path in (train_path, test_path):
        output_path.mkdir(parents=True, exist_ok=True)

    # Sao chép các ảnh vào các thư mục train và test tương ứng
    for image_path, label in zip(train_images, train_labels):
        destination = train_path / label / image_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination)

    for image_path, label in zip(test_images, test_labels):
        destination = test_path / label / image_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination)

    # Thống kê số lượng ảnh trong các tập train và test
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    print(f"Split complete: train={len(train_images)}, test={len(test_images)}")
    for class_dir in class_dirs:
        name = class_dir.name
        print(f"{name}: train={train_counts[name]}, test={test_counts[name]}")

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"

    parser = argparse.ArgumentParser(description="Chia dữ liệu thô thành train-test (80:20) với cấu trúc thư mục tương ứng")
    parser.add_argument("--raw-dir", type=Path, default=data_dir / "raw")
    parser.add_argument("--train-dir", type=Path, default=data_dir / "train")
    parser.add_argument("--test-dir", type=Path, default=data_dir / "test")
    parser.add_argument("--seed", type=int, default=random_state)

    args = parser.parse_args()

    split_data(
        raw_path=args.raw_dir,
        train_path=args.train_dir,
        test_path=args.test_dir,
        random_state=args.seed
    )