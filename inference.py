"""CPU inference using the exported model and the original face crop rules."""
import hashlib
import json
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np
import onnxruntime as ort

from mask_classes import CLASS_NAMES, FACE_PADDING, expand_face_box


class Predictor:
    def __init__(self, model_path, face_directory):
        model_path, face_directory = Path(model_path), Path(face_directory)
        metadata = json.loads(Path(str(model_path) + '.labels.json').read_text(encoding='utf-8'))
        if metadata.get('class_names') != list(CLASS_NAMES) or metadata.get('face_padding') != FACE_PADDING:
            raise ValueError('Model labels/crop configuration does not match inference.')
        self.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        cv2.setNumThreads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].shape[1:] != [224, 224, 3]
                or inputs[0].type != 'tensor(float)' or len(outputs) != 1 or outputs[0].shape[-1] != 3):
            raise ValueError('Expected float32 NHWC input and three class outputs.')
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.face_net = cv2.dnn.readNet(str(face_directory / 'deploy.prototxt'),
            str(face_directory / 'res10_300x300_ssd_iter_140000.caffemodel'))
        self.lock = Lock()
        self.classify(np.zeros((1, 224, 224, 3), dtype=np.float32))

    def classify(self, rgb_batch):
        batch = np.asarray(rgb_batch, dtype=np.float32) / 127.5 - 1.0
        probabilities = self.session.run([self.output_name], {self.input_name: batch})[0]
        if (probabilities.shape != (len(batch), len(CLASS_NAMES))
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)):
            raise ValueError('Model returned invalid class probabilities.')
        return probabilities

    def predict(self, frame):
        faces, _ = self.predict_with_metrics(frame)
        return faces

    def predict_with_metrics(self, frame):
        height, width = frame.shape[:2]
        faces, boxes = [], []
        metrics = {'face_detection_ms': 0.0, 'face_preprocess_ms': 0.0,
                   'classification_ms': 0.0, 'queue_ms': 0.0}
        waiting = time.perf_counter()
        with self.lock:
            metrics['queue_ms'] = (time.perf_counter() - waiting) * 1000
            started = time.perf_counter()
            self.face_net.setInput(cv2.dnn.blobFromImage(
                frame, 1.0, (300, 300), (104.0, 177.0, 123.0)))
            detections = self.face_net.forward()
            metrics['face_detection_ms'] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            for detection in detections[0, 0]:
                if not np.isfinite(detection[2:7]).all() or detection[2] <= 0.5:
                    continue
                x1, y1, x2, y2 = (detection[3:7] * np.array([width, height, width, height])).astype(int)
                x1, x2 = np.clip([x1, x2], 0, width)
                y1, y2 = np.clip([y1, y2], 0, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_box = expand_face_box((x1, y1, x2, y2), width, height)
                cx1, cy1, cx2, cy2 = crop_box
                face = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
                faces.append(cv2.resize(face, (224, 224)))
                boxes.append(([int(v) for v in (x1, y1, x2, y2)], list(crop_box)))
            if len(faces) > 32:
                raise ValueError('Too many faces in one frame (maximum 32).')
            metrics['face_preprocess_ms'] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            probabilities = self.classify(faces) if faces else []
            metrics['classification_ms'] = (time.perf_counter() - started) * 1000 if faces else 0.0
        result = []
        for (box, crop_box), scores in zip(boxes, probabilities):
            index = int(np.argmax(scores))
            result.append({'box': box, 'classifier_box': crop_box,
                'label': CLASS_NAMES[index], 'confidence': float(scores[index]),
                'probabilities': {name: float(scores[i]) for i, name in enumerate(CLASS_NAMES)}})
        metrics['inference_ms'] = metrics['face_detection_ms'] + metrics['classification_ms']
        return result, {name: round(value, 3) for name, value in metrics.items()}
