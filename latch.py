"""
くくり罠 電磁ロック(ソレノイド/サーボ)制御モジュール

フェイルセーフ設計:
- ソレノイドは「バネ押し出し・通電引込(ノーマリーロック)」型を前提とする。
- 消磁 = ロック(発火不能)。停電・プロセス死・GPIO異常はすべて安全側に倒れる。
- 解錠には必ずウォッチドッグ(最大解錠時間)を付け、更新が途絶えたら自動再ロック。

Raspberry Pi 以外(開発PC)では自動的にモックとして動作しログ出力のみ行う。
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


class SolenoidLatch:
    """GPIO 経由のソレノイドロック制御(ノーマリーロック型)"""

    def __init__(self, pin: int, max_unlock_seconds: float = 30.0, active_high: bool = True):
        """
        Args:
            pin:                BCM ピン番号(MOSFETゲート/リレー入力)
            max_unlock_seconds: ウォッチドッグ。extend されない限りこの時間で必ず再ロック
            active_high:        True なら HIGH=通電(解錠)
        """
        self._pin = pin
        self._max_unlock = max_unlock_seconds
        self._active_high = active_high
        self._unlocked_until = 0.0
        self._lock = threading.Lock()
        self._fired = False

        if _HAS_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT,
                       initial=GPIO.LOW if active_high else GPIO.HIGH)
            # ウォッチドッグスレッド: 期限切れを監視して強制再ロック
            t = threading.Thread(target=self._watchdog, daemon=True)
            t.start()
        else:
            logger.warning("RPi.GPIO が無いためモックモードで動作します(実際の解錠は行われません)")

        logger.info("ソレノイドロック初期化: pin=%s 状態=LOCKED", pin)

    # ---------- 状態遷移 ----------

    def unlock(self, seconds: float) -> None:
        """指定秒数だけ解錠する。連続で呼べば解錠が延長される。"""
        seconds = min(seconds, self._max_unlock)
        with self._lock:
            self._unlocked_until = time.monotonic() + seconds
            self._write(True)
        logger.info("解錠(発火可能): %.1f 秒間", seconds)

    def relock(self, reason: str = "") -> None:
        """即時再ロック(消磁)。禁止対象出現・対象消失・異常時に呼ぶ。"""
        with self._lock:
            self._unlocked_until = 0.0
            self._write(False)
        logger.info("再ロック(発火不能): %s", reason or "manual")

    def is_unlocked(self) -> bool:
        with self._lock:
            return time.monotonic() < self._unlocked_until

    def fire_direct(self, pulse_seconds: float = 1.5) -> None:
        """
        案B(直接発火)用: ソレノイドをパルス通電してシアを解放する。
        機構が『通電=シア解放=バネ発射』になっている前提。
        """
        logger.warning("!! 直接発火 !! パルス %.1f 秒", pulse_seconds)
        self._write(True)
        time.sleep(pulse_seconds)
        self._write(False)
        self._fired = True

    @property
    def fired(self) -> bool:
        return self._fired

    # ---------- 内部 ----------

    def _write(self, energize: bool) -> None:
        if _HAS_GPIO:
            level = energize if self._active_high else not energize
            GPIO.output(self._pin, GPIO.HIGH if level else GPIO.LOW)
        else:
            logger.debug("[mock] GPIO%d %s", self._pin, "通電" if energize else "消磁")

    def _watchdog(self) -> None:
        """解錠期限を100msごとに監視し、期限切れなら強制再ロック"""
        was_unlocked = False
        while True:
            time.sleep(0.1)
            with self._lock:
                now_unlocked = time.monotonic() < self._unlocked_until
                if was_unlocked and not now_unlocked:
                    self._write(False)
                    logger.info("再ロック(ウォッチドッグ期限切れ)")
                was_unlocked = now_unlocked

    def cleanup(self) -> None:
        self.relock("shutdown")
        if _HAS_GPIO:
            GPIO.cleanup(self._pin)
