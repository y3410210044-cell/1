"""
SwitchBot API v1.1 クライアント
HMAC-SHA256 署名による認証でデバイスを操作する
"""

import hashlib
import hmac
import logging
import time
import uuid
import base64
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SWITCHBOT_API_BASE = "https://api.switch-bot.com/v1.1"


class SwitchBotClient:
    """SwitchBot API v1.1 クライアント"""

    def __init__(self, token: str, secret: str, timeout: int = 10):
        """
        Args:
            token:   SwitchBot アプリで取得した API トークン
            secret:  SwitchBot アプリで取得した クライアントシークレット
            timeout: HTTP タイムアウト (秒)
        """
        self._token = token
        self._secret = secret
        self._timeout = timeout

    def _build_headers(self) -> dict:
        """HMAC-SHA256 署名ヘッダーを生成する"""
        nonce = str(uuid.uuid4())
        timestamp = str(int(time.time() * 1000))

        message = self._token + timestamp + nonce
        sign = base64.b64encode(
            hmac.new(
                self._secret.encode("utf-8"),
                message.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        return {
            "Authorization": self._token,
            "Content-Type": "application/json",
            "sign": sign,
            "nonce": nonce,
            "t": timestamp,
        }

    def send_command(
        self,
        device_id: str,
        command: str,
        parameter: str = "default",
        command_type: str = "command",
    ) -> bool:
        """
        デバイスにコマンドを送信する

        Args:
            device_id:    対象デバイスの ID
            command:      コマンド名 ("press", "turnOn", "turnOff" など)
            parameter:    コマンドパラメータ (通常は "default")
            command_type: コマンドタイプ ("command" または "customize")

        Returns:
            True: 成功 / False: 失敗
        """
        url = f"{SWITCHBOT_API_BASE}/devices/{device_id}/commands"
        payload = {
            "command": command,
            "parameter": parameter,
            "commandType": command_type,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._build_headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

            status_code = data.get("statusCode")
            if status_code == 100:
                logger.info(
                    "SwitchBot コマンド成功: device=%s command=%s",
                    device_id,
                    command,
                )
                return True
            else:
                logger.error(
                    "SwitchBot API エラー: statusCode=%s message=%s",
                    status_code,
                    data.get("message"),
                )
                return False

        except requests.exceptions.Timeout:
            logger.error("SwitchBot API タイムアウト: device=%s", device_id)
            return False
        except requests.exceptions.RequestException as exc:
            logger.error("SwitchBot API 通信エラー: %s", exc)
            return False

    def get_devices(self) -> Optional[list]:
        """登録済みデバイス一覧を取得する (動作確認用)"""
        url = f"{SWITCHBOT_API_BASE}/devices"
        try:
            response = requests.get(
                url,
                headers=self._build_headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("statusCode") == 100:
                return data["body"].get("deviceList", [])
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("デバイス一覧取得エラー: %s", exc)
            return None
