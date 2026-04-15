"""
YOLOv8 を使ったイノシシ検知モジュール
カメラフレームからイノシシを検出し、バウンディングボックスと信頼度を返す
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """単一の検知結果"""
    class_name: str
    confidence: float
    # バウンディングボックス (x1, y1, x2, y2) ピクセル座標
    bbox: tuple  # (x1, y1, x2, y2)


@dataclass
class BoarDetector:
    """YOLOv8 を用いたイノシシ検知器"""

    model_path: str
    confidence_threshold: float
    target_classes: List[str]

    # 内部状態 (初期化後に設定)
    _model: object = field(default=None, init=False, repr=False)

    def load_model(self) -> None:
        """YOLOv8 モデルを読み込む"""
        # ultralytics は重いので遅延インポート
        from ultralytics import YOLO

        path = Path(self.model_path)
        if not path.exists():
            logger.warning(
                "モデルファイルが見つかりません: %s\n"
                "カスタムモデルを 'models/boar_yolov8.pt' に配置するか、\n"
                "config.yaml の model_path を変更してください。",
                self.model_path,
            )
            raise FileNotFoundError(f"モデルが見つかりません: {self.model_path}")

        logger.info("YOLOv8 モデルを読み込み中: %s", self.model_path)
        self._model = YOLO(str(path))
        logger.info("モデル読み込み完了")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        1フレームからイノシシを検出する

        Args:
            frame: BGR フォーマットの画像 (OpenCV)

        Returns:
            検知されたイノシシのリスト (空リスト = 検知なし)
        """
        if self._model is None:
            raise RuntimeError("モデルが読み込まれていません。load_model() を先に呼んでください。")

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id].lower()
                confidence = float(box.conf[0])

                if self._is_target(class_name):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(
                        Detection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                        )
                    )
                    logger.debug(
                        "検知: class=%s conf=%.2f bbox=(%d,%d,%d,%d)",
                        class_name, confidence, x1, y1, x2, y2,
                    )

        return detections

    def _is_target(self, class_name: str) -> bool:
        """検知クラス名がターゲット対象かどうか判定する"""
        return any(
            target.lower() in class_name or class_name in target.lower()
            for target in self.target_classes
        )

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """
        フレームに検知結果を描画する

        Args:
            frame:      元フレーム (変更しない)
            detections: 検知結果リスト

        Returns:
            描画済みフレームのコピー
        """
        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            label = f"{det.class_name} {det.confidence:.0%}"

            # 赤色のバウンディングボックス
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # ラベル背景
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                output,
                (x1, y1 - text_h - 8),
                (x1 + text_w + 4, y1),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return output
