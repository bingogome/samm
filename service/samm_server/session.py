from dataclasses import dataclass
from threading import Lock
import time


@dataclass(frozen=True)
class SessionResult:
    status_code: int
    payload: dict


class SessionTracker:
    def __init__(self, timeout_seconds=600, now=None):
        self.timeout_seconds = timeout_seconds
        self.now = now or time.monotonic
        self.sessions = {}
        self.latest_autosaves = {}
        self.started_at = self.now()
        self.lock = Lock()

    def touch(self, payload):
        if not isinstance(payload, dict):
            return SessionResult(400, {"error": "payload must be an object"})
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return SessionResult(400, {"error": "session_id must be a non-empty string"})

        with self.lock:
            autosave = payload.get("autosave", {})
            self.sessions[session_id] = {
                "last_seen": self.now(),
                "autosave": autosave,
            }
            if autosave:
                self.latest_autosaves[session_id] = autosave
            self.prune_locked()
            return SessionResult(200, self.status_locked(session_id))

    def expired(self):
        with self.lock:
            self.prune_locked()
            return self.idle_seconds_locked() >= self.timeout_seconds

    def status(self):
        with self.lock:
            self.prune_locked()
            return self.status_locked(None)

    def autosaves(self):
        with self.lock:
            return dict(self.latest_autosaves)

    def prune_locked(self):
        now = self.now()
        self.sessions = {
            session_id: session
            for session_id, session in self.sessions.items()
            if now - session["last_seen"] < self.timeout_seconds
        }

    def idle_seconds_locked(self):
        if self.sessions:
            return 0
        return self.now() - self.started_at

    def status_locked(self, session_id):
        return {
            "status": "active",
            "session_id": session_id,
            "active_sessions": len(self.sessions),
            "timeout_seconds": self.timeout_seconds,
        }
