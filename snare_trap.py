"""
くくり罠 自動作動システム
==========================
設計思想 (docs/snare_design.md 参照):
  電子系はミリ秒で発火できないため、AIを「引き金」ではなく「安全装置」にする。

モード:
  interlock (推奨) : 踏板→バネの純機械トリガーは残し、ソレノイドで常時ロック。
                     AIがイノシシ/シカを確認し禁止対象がいない間だけ通電解錠する。
  direct           : 餌で静止させ、足先が発火ゾーンに滞在したらソレノイドで直接発火。

使い方:
    python snare_trap.py                          # config_snare.yaml で起動
    python snare_trap.py --config my.yaml
    python snare_trap.py --headless               # 画面なし (Raspberry Pi)
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests
import yaml

from detector import BoarDetector
from latch import SolenoidLatch

logger = logging.getLogger(__name__)


# ---------------- ユーティリティ ----------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_cfg.get("file"):
        Path(log_cfg["file"]).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_cfg["file"], encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def rect_to_px(rect, w: int, h: int) -> tuple:
    """相対座標 [x1,y1,x2,y2] → ピクセル座標"""
    return (int(rect[0] * w), int(rect[1] * h), int(rect[2] * w), int(rect[3] * h))


def point_in_rect(pt, rect_px) -> bool:
    x, y = pt
    x1, y1, x2, y2 = rect_px
    return x1 <= x <= x2 and y1 <= y <= y2


def foot_point(bbox) -> tuple:
    """足先の代理点 = バウンディングボックス下端中央"""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


def notify(cfg: dict, message: str, image=None) -> None:
    """LINE Notify / Webhook へ通報 (best-effort)"""
    n = cfg.get("notify", {})
    if n.get("save_snapshot") and image is not None:
        d = Path(cfg.get("logging", {}).get("event_dir", "events"))
        d.mkdir(parents=True, exist_ok=True)
        fn = d / f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(fn), image)
        logger.info("イベント画像保存: %s", fn)
    token = n.get("line_notify_token")
    if token:
        try:
            requests.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {token}"},
                data={"message": message},
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.error("LINE 通知失敗: %s", exc)
    url = n.get("webhook_url")
    if url:
        try:
            requests.post(url, json={"content": message}, timeout=10)
        except requests.RequestException as exc:
            logger.error("Webhook 通知失敗: %s", exc)


# ---------------- メインループ ----------------

def run(cfg: dict, headless: bool) -> None:
    snare = cfg["snare"]
    mode = snare.get("mode", "interlock")

    det_cfg = cfg["detection"]
    target_names = {c.lower() for c in det_cfg["target_classes"]}
    forbidden_names = {c.lower() for c in det_cfg["forbidden_classes"]}

    # 検知器は対象+禁止の全クラスを返すよう設定し、後段で振り分ける
    detector = BoarDetector(
        model_path=det_cfg["model_path"],
        confidence_threshold=det_cfg["confidence_threshold"],
        target_classes=sorted(target_names | forbidden_names),
    )
    detector.load_model()

    latch = SolenoidLatch(
        pin=snare["gpio_pin"],
        max_unlock_seconds=snare.get("max_unlock_seconds", 30.0),
    )

    cam = cfg["camera"]
    cap = cv2.VideoCapture(cam["source"] if not str(cam["source"]).isdigit() else int(cam["source"]))
    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けません: {cam['source']}")
    if cam.get("width"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])

    unlock_grace = snare.get("unlock_grace_seconds", 8.0)
    dwell_required = snare.get("direct_dwell_frames", 4)
    confirm_required = snare.get("confirm_frames", 3)
    dwell_count = 0
    confirm_count = 0
    fired = False

    logger.info("くくり罠システム起動 mode=%s (終了: Ctrl+C%s)", mode, " / Qキー" if not headless else "")
    notify(cfg, f"くくり罠システム起動 (mode={mode})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("フレーム取得失敗。1秒後に再試行")
                time.sleep(1)
                continue

            h, w = frame.shape[:2]
            approach_px = rect_to_px(snare["approach_zone"], w, h)
            fire_px = rect_to_px(snare["fire_zone"], w, h)

            detections = detector.detect(frame)
            targets = [d for d in detections if d.class_name in target_names]
            forbidden = [d for d in detections if d.class_name in forbidden_names]

            # ---- 最優先: 禁止対象(人・クマ・イヌ・カモシカ)チェック ----
            if forbidden:
                names = ", ".join(sorted({d.class_name for d in forbidden}))
                if latch.is_unlocked():
                    latch.relock(f"禁止対象出現: {names}")
                    notify(cfg, f"⚠ 禁止対象を検知し再ロックしました: {names}",
                           detector.draw_detections(frame, detections))
                confirm_count = 0
                dwell_count = 0

            elif not fired:
                # 接近ゾーン内の対象
                in_approach = [d for d in targets
                               if point_in_rect(foot_point(d.bbox), approach_px)]

                if in_approach:
                    confirm_count += 1
                else:
                    confirm_count = 0
                    dwell_count = 0

                if mode == "interlock":
                    # ---- 案A: 確認できている間だけ解錠を延長 ----
                    if confirm_count >= confirm_required:
                        latch.unlock(unlock_grace)
                    elif confirm_count == 0 and latch.is_unlocked():
                        # 猶予はウォッチドッグに任せる(急な再ロックで機構が暴れないよう自然失効)
                        pass

                elif mode == "direct":
                    # ---- 案B: 足先が発火ゾーンに滞在したら発火 ----
                    feet_on_plate = [d for d in in_approach
                                     if point_in_rect(foot_point(d.bbox), fire_px)]
                    if feet_on_plate and confirm_count >= confirm_required:
                        dwell_count += 1
                        logger.info("発火ゾーン滞在 %d/%d (conf=%.2f)",
                                    dwell_count, dwell_required,
                                    max(d.confidence for d in feet_on_plate))
                        if dwell_count >= dwell_required:
                            logger.warning("!! 発火条件成立 !! 対象=%s",
                                           feet_on_plate[0].class_name)
                            latch.fire_direct(snare.get("fire_pulse_seconds", 1.5))
                            fired = True
                            notify(cfg, "🐗 くくり罠を作動させました。至急見回りしてください。",
                                   detector.draw_detections(frame, detections))
                    else:
                        dwell_count = 0

            # ---- 表示 ----
            if not headless:
                view = detector.draw_detections(frame, detections)
                cv2.rectangle(view, approach_px[:2], approach_px[2:], (255, 200, 0), 2)
                cv2.rectangle(view, fire_px[:2], fire_px[2:], (0, 0, 255), 2)
                if fired:
                    status, color = "FIRED - 要見回り", (0, 0, 255)
                elif forbidden:
                    status, color = "LOCKED (禁止対象)", (0, 0, 255)
                elif latch.is_unlocked():
                    status, color = "UNLOCKED (発火可能)", (0, 165, 255)
                else:
                    status, color = "LOCKED (監視中)", (0, 255, 0)
                cv2.putText(view, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.imshow("Snare Trap", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Ctrl+C で終了")
    finally:
        latch.cleanup()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        logger.info("システム停止(ロック状態で終了)")


def main() -> None:
    parser = argparse.ArgumentParser(description="くくり罠 自動作動システム")
    parser.add_argument("--config", default="config_snare.yaml")
    parser.add_argument("--headless", action="store_true", help="画面なしで動作")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    run(cfg, headless=args.headless)


if __name__ == "__main__":
    main()
