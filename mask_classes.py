"""Single class order shared by training and all inference entry points."""
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ("mask", "no_mask", "incorrect_mask")
CLASS_DIRECTORIES = {
    "mask": "with_mask",
    "no_mask": "without_mask",
    "incorrect_mask": "incorrect_mask",
}
DISPLAY_LABELS = {"mask": "Mask", "no_mask": "No Mask", "incorrect_mask": "Incorrect Mask"}
CLASS_COLORS = {"mask": (0, 255, 0), "no_mask": (0, 0, 255), "incorrect_mask": (0, 165, 255)}
DEFAULT_MODEL = "mask_detector_3class_refined.model"
FACE_PADDING = {"sides": 0.25, "top": 0.10, "bottom": 0.50}


def expand_face_box(box, width, height):
    """Retain masks under the chin or hanging from an ear in the classifier ROI."""
    x1, y1, x2, y2 = (int(value) for value in box)
    face_width, face_height = x2 - x1, y2 - y1
    return (max(0, int(np.floor(x1 - face_width * FACE_PADDING["sides"]))),
            max(0, int(np.floor(y1 - face_height * FACE_PADDING["top"]))),
            min(width, int(np.ceil(x2 + face_width * FACE_PADDING["sides"]))),
            min(height, int(np.ceil(y2 + face_height * FACE_PADDING["bottom"]))))


def validate_model(model, model_path):
    """Reject old two-class models and ambiguous/mismatched label metadata."""
    if tuple(model.input_shape[1:]) != (224, 224, 3) or model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("Model cần đầu vào (224, 224, 3) và đầu ra 3 nhãn. "
                         "Chạy train_mask_detector.py để tạo model 3 lớp.")
    path = Path(str(model_path) + ".labels.json")
    if not path.is_file():
        raise ValueError("Thiếu file thứ tự nhãn: {}. Dùng model và metadata từ cùng lần train.".format(path))
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Thứ tự nhãn của model không khớp: {}".format(list(CLASS_NAMES)))
    if "face_padding" in metadata and metadata["face_padding"] != FACE_PADDING:
        raise ValueError("Cách cắt vùng mặt của model không khớp cấu hình nhận diện.")


def decode_prediction(prediction):
    """Return the winning display label, confidence and OpenCV BGR color."""
    scores = np.asarray(prediction)
    if scores.shape != (len(CLASS_NAMES),) or not np.isfinite(scores).all():
        raise ValueError("Expected three finite class probabilities.")
    index = int(np.argmax(scores))
    name = CLASS_NAMES[index]
    return DISPLAY_LABELS[name], float(scores[index]), CLASS_COLORS[name]
