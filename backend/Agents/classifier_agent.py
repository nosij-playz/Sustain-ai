import os
from typing import List, Dict, Optional

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class ClassifierAgent:
    """Lightweight wrapper to reuse the YOLO model from the waste-detection weights.

    The agent expects `params` with keys:
      - `image_path` (str): path to a local image file to classify
      - `conf` (float, optional): confidence threshold, default 0.5
    """
    def __init__(self, model_path: Optional[str] = None):
        # Keep all model assets inside the waste-dispo repo.
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_paths = [
            os.path.join(root, "weights", "best.pt"),
            os.path.join(root, "weights", "best.onnx"),
            os.path.join(root, "weights", "last.pt"),
        ]

        if model_path:
            default_paths.insert(0, model_path)

        chosen = None
        for p in default_paths:
            if p and os.path.exists(p):
                chosen = p
                break

        self.model_path = chosen
        self.model = None
        if YOLO is not None and self.model_path:
            try:
                self.model = YOLO(self.model_path)
            except Exception:
                self.model = None

    def run(self, params: Dict) -> Dict:
        image_path = (params or {}).get("image_path")
        conf = float((params or {}).get("conf", 0.5))

        if not image_path or not os.path.exists(image_path):
            return {"success": False, "error": "No image file found at path."}

        if YOLO is None:
            return {"success": False, "error": "ultralytics.YOLO not available (install ultralytics)."}

        if self.model is None:
            if self.model_path is None:
                return {"success": False, "error": "No model weights found. Place weights in waste-detection/weights/."}
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                return {"success": False, "error": f"Failed to load model: {e}"}

        try:
            results = self.model.predict(image_path, conf=conf)
            names = getattr(self.model, "names", {}) or {}
            detections: List[Dict] = []
            # results is an iterable; combine all boxes
            for res in results:
                boxes = getattr(res, "boxes", None)
                if not boxes:
                    continue
                for i in range(len(boxes.cls)):
                    try:
                        cls_idx = int(boxes.cls[i])
                        label = names.get(cls_idx, str(cls_idx))
                        conf_score = float(boxes.conf[i])
                        xyxy = tuple(map(float, boxes.xyxy[i])) if hasattr(boxes, "xyxy") else None
                    except Exception:
                        # best effort fallback
                        label = str(getattr(boxes, "cls", i))
                        conf_score = float(getattr(boxes, "conf", 0.0))
                        xyxy = None

                    detections.append({"label": label, "confidence": conf_score, "bbox": xyxy})

            return {"success": True, "image": image_path, "detections": detections}
        except Exception as e:
            return {"success": False, "error": str(e)}
