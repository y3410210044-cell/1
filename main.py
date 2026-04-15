"""
イノシシ検知 & SwitchBot 罠作動システム
========================================
カメラ映像からイノシシを検知し、SwitchBot 経由で罠を作動させる

使い方:
    python main.py                    # デフォルト設定 (config.yaml)
    python main.py --config my.yaml   # 設定ファイルを指定
    python main.py --list-devices     # SwitchBot デバイス一覧を表示
    python main.py --no-display       # 映像ウィンドウなしで起動
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from detector import BoarDetector
from switchbot import SwitchBotClient


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    log_file = log_cfg.get("file", "")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def open_camera(cfg: dict) -> cv2.VideoCapture:
    source = cfg["source"]
    # 数値文字列ならデバイス番号に変換
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: {source}")

    w = cfg.get("width", 0)
    h = cfg.get("height", 0)
    if w and h:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    fps = cfg.get("fps", 30)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def save_detection_image(frame, detections, save_dir: str) -> None:
    """検知時のフレームをタイムスタンプ付きで保存する"""
    if not save_dir:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = Path(save_dir) / f"boar_{ts}.jpg"
    cv2.imwrite(str(filename), frame)
    logging.getLogger(__name__).info("検知画像を保存: %s", filename)


def list_devices(cfg: dict) -> None:
    """登録済み SwitchBot デバイス一覧を表示する"""
    sb = cfg["switchbot"]
    client = SwitchBotClient(
        token=sb["token"],
        secret=sb["secret"],
        timeout=sb.get("timeout_seconds", 10),
    )
    devices = client.get_devices()
    if devices is None:
        print("デバイス一覧の取得に失敗しました。token/secret を確認してください。")
        return
    if not devices:
        print("登録済みデバイスがありません。")
        return
    print(f"{'デバイスID':<40} {'名前':<30} {'タイプ'}")
    print("-" * 80)
    for dev in devices:
        print(
            f"{dev.get('deviceId', ''):<40} "
            f"{dev.get('deviceName', ''):<30} "
            f"{dev.get('deviceType', '')}"
        )


def run(config: dict, display: bool) -> None:
    logger = logging.getLogger(__name__)

    # --- 検知器の初期化 ---
    det_cfg = config["detection"]
    detector = BoarDetector(
        model_path=det_cfg["model_path"],
        confidence_threshold=det_cfg["confidence_threshold"],
        target_classes=det_cfg["target_classes"],
    )
    detector.load_model()

    # --- SwitchBot クライアントの初期化 ---
    sb_cfg = config["switchbot"]
    switchbot = SwitchBotClient(
        token=sb_cfg["token"],
        secret=sb_cfg["secret"],
        timeout=sb_cfg.get("timeout_seconds", 10),
    )

    # --- カメラを開く ---
    cap = open_camera(config["camera"])
    logger.info("カメラ起動完了。監視を開始します。(終了: Q キー または Ctrl+C)")

    consecutive_count = 0
    required_frames = det_cfg.get("consecutive_frames_required", 3)
    cooldown_seconds = det_cfg.get("cooldown_seconds", 300)
    last_trigger_time = 0.0
    save_dir = config.get("logging", {}).get("save_detections_dir", "")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("フレームの取得に失敗しました。カメラ接続を確認してください。")
                time.sleep(1)
                continue

            # --- イノシシ検知 ---
            detections = detector.detect(frame)

            now = time.time()
            in_cooldown = (now - last_trigger_time) < cooldown_seconds

            if detections:
                consecutive_count += 1
                logger.info(
                    "イノシシ検知 (%d/%d フレーム) conf=%.2f%s",
                    consecutive_count,
                    required_frames,
                    max(d.confidence for d in detections),
                    " [クールダウン中]" if in_cooldown else "",
                )

                # 連続検知フレーム数が閾値を超えたら罠を作動
                if consecutive_count >= required_frames and not in_cooldown:
                    logger.warning(
                        "!! イノシシ確定 !! SwitchBot を作動させます: device=%s command=%s",
                        sb_cfg["device_id"],
                        sb_cfg["command"],
                    )
                    success = switchbot.send_command(
                        device_id=sb_cfg["device_id"],
                        command=sb_cfg["command"],
                    )
                    if success:
                        logger.info("罠作動成功。クールダウン開始 (%d 秒)", cooldown_seconds)
                        last_trigger_time = now
                        consecutive_count = 0
                    else:
                        logger.error("罠作動失敗。次のフレームで再試行します。")

                # 検知画像を保存
                if save_dir:
                    annotated = detector.draw_detections(frame, detections)
                    save_detection_image(annotated, detections, save_dir)

            else:
                # 連続検知カウントをリセット
                if consecutive_count > 0:
                    logger.debug("検知なし。カウントリセット (%d → 0)", consecutive_count)
                consecutive_count = 0

            # --- 映像表示 ---
            if display:
                annotated = detector.draw_detections(frame, detections)

                # ステータス表示
                status = "監視中"
                color = (0, 255, 0)
                if in_cooldown:
                    remain = int(cooldown_seconds - (now - last_trigger_time))
                    status = f"クールダウン中 ({remain}s)"
                    color = (0, 165, 255)
                elif detections:
                    status = f"イノシシ検知! ({consecutive_count}/{required_frames})"
                    color = (0, 0, 255)

                cv2.putText(
                    annotated, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                )
                cv2.imshow("イノシシ検知システム", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Q キーで終了します。")
                    break

    except KeyboardInterrupt:
        logger.info("Ctrl+C で終了します。")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        logger.info("システム停止。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="イノシシ検知 & SwitchBot 罠作動システム"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="設定ファイルのパス (デフォルト: config.yaml)"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="SwitchBot デバイス一覧を表示して終了"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="映像ウィンドウを表示しない (ヘッドレス動作)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("設定ファイル読み込み完了: %s", args.config)

    if args.list_devices:
        list_devices(config)
        return

    run(config, display=not args.no_display)


if __name__ == "__main__":
    main()
