"""
Axiom Telemetry & Observability Engine - Services Module.
Provides low-overhead in-memory ring-buffered event logging, live user presence tracking,
IP/Key blacklisting, global broadcast alerts, and dynamic request throttling.
Designed specifically for resource-constrained environments (Render Free/Starter tier)
with zero disk bloat.
"""

import collections
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("academic_assistant.telemetry")


class TelemetryManager:
    """
    Thread-safe in-memory telemetry and moderation manager.
    Keeps a maximum 500 recent events ring buffer (deque) and tracks active user presence.
    Persists moderation state (banned IPs, banned keys, broadcast banners, limits)
    in a lightweight JSON file.
    """

    def __init__(self, storage_path: Optional[Path] = None, max_events: int = 500):
        self.storage_path = storage_path or (Path("data") / "admin_state.json")
        self.max_events = max_events
        self.events: collections.deque = collections.deque(maxlen=max_events)
        self.active_users: Dict[str, Dict[str, Any]] = {}
        self.banned_ips: Set[str] = set()
        self.banned_keys: Set[str] = set()
        self.broadcast: Dict[str, Any] = {
            "message": "",
            "active": False,
            "updated_at": ""
        }
        self.limits: Dict[str, int] = {
            "global_hourly_limit": 0,
            "key_hourly_limit": 0
        }
        self.start_time: float = time.time()
        self._lock = threading.Lock()
        self._load_state()

    def _load_state(self):
        """Loads persistent moderation and broadcast settings from disk."""
        with self._lock:
            if self.storage_path.exists():
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.banned_ips = set(data.get("banned_ips", []))
                    self.banned_keys = set(data.get("banned_keys", []))
                    self.broadcast = data.get("broadcast", {
                        "message": "",
                        "active": False,
                        "updated_at": ""
                    })
                    self.limits = data.get("limits", {
                        "global_hourly_limit": 0,
                        "key_hourly_limit": 0
                    })
                    logger.info(
                        "Loaded admin state: %d banned IPs, %d banned keys, broadcast active=%s",
                        len(self.banned_ips),
                        len(self.banned_keys),
                        self.broadcast.get("active", False)
                    )
                except Exception as e:
                    logger.error("Failed to load admin state from %s: %s", self.storage_path, str(e))

    def _save_state(self):
        """Atomically persists moderation and broadcast settings to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "banned_ips": sorted(list(self.banned_ips)),
                "banned_keys": sorted(list(self.banned_keys)),
                "broadcast": self.broadcast,
                "limits": self.limits,
                "saved_at": datetime.now(timezone.utc).isoformat()
            }
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.storage_path)
        except Exception as e:
            logger.error("Failed to persist admin state: %s", str(e))

    @staticmethod
    def classify_action_type(path: str, method: str = "GET") -> str:
        """Classifies a URL route into a concise, standardized action tag."""
        p = path.lower()
        if "/api/research/stream" in p:
            return "SEARCH_PAPERS"
        if "/api/search/entity" in p:
            return "ENTITY_SEARCH"
        if "/api/paper/diagram" in p or "/api/diagram" in p:
            return "EXTRACT_ARCHITECTURE"
        if "/api/paper/compare-matrix" in p or "/api/papers/compare" in p:
            return "GENERATE_MATRIX"
        if "/api/outreach/generate" in p:
            return "OUTREACH_GEN"
        if "/api/paper/thesis-proposal" in p:
            return "THESIS_PROPOSAL"
        if "/api/export/" in p:
            return "EXPORT_DOC"
        if "/api/health" in p or "/health" in p:
            return "HEALTH_PING"
        if "/api/broadcast" in p:
            return "BROADCAST_CHECK"
        if "/admin" in p:
            return "ADMIN_ACCESS"
        if p == "/" or "/static" in p:
            return "STATIC_VIEW"
        clean_name = p.strip("/").replace("/", "_").replace("-", "_").upper()
        return clean_name or f"{method.upper()}_REQUEST"

    def record_event(
        self,
        client_ip: str,
        access_key: Optional[str],
        action_type: str,
        query_preview: str = "",
        path: str = "/",
        method: str = "GET",
        status_code: int = 200,
        latency_ms: float = 0.0,
        user_agent: str = ""
    ) -> Dict[str, Any]:
        """
        Appends a telemetry record to the ring buffer and updates live user presence.
        Truncates query preview to 120 characters.
        """
        now = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        clean_ip = (client_ip or "127.0.0.1").strip()
        clean_key = (access_key or "ANONYMOUS").strip()
        safe_query = (query_preview or "").strip()
        if len(safe_query) > 120:
            safe_query = safe_query[:117] + "..."

        # Simplify user agent
        clean_ua = (user_agent or "Unknown Client").strip()
        if len(clean_ua) > 100:
            clean_ua = clean_ua[:97] + "..."

        event_record = {
            "id": f"evt-{int(now * 1000)}-{len(self.events) + 1}",
            "timestamp": now_iso,
            "unix_time": now,
            "client_ip": clean_ip,
            "access_key": clean_key,
            "action_type": action_type,
            "query_preview": safe_query,
            "path": path,
            "method": method.upper(),
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "user_agent": clean_ua
        }

        user_identifier = f"{clean_ip}::{clean_key}"

        with self._lock:
            self.events.append(event_record)

            # Update live presence index
            if user_identifier not in self.active_users:
                self.active_users[user_identifier] = {
                    "client_ip": clean_ip,
                    "access_key": clean_key,
                    "first_seen": now_iso,
                    "last_seen": now_iso,
                    "last_seen_unix": now,
                    "total_requests": 1,
                    "last_action": action_type,
                    "last_query": safe_query,
                    "user_agent": clean_ua
                }
            else:
                user = self.active_users[user_identifier]
                user["last_seen"] = now_iso
                user["last_seen_unix"] = now
                user["total_requests"] += 1
                user["last_action"] = action_type
                if safe_query:
                    user["last_query"] = safe_query
                user["user_agent"] = clean_ua

        return event_record

    def get_online_users(self, window_seconds: float = 300.0) -> List[Dict[str, Any]]:
        """
        Returns active users who sent a request within the given window (default: 5 minutes).
        Marks users as 'online' (active within 5m) or 'idle' (active within 30m).
        """
        now = time.time()
        results: List[Dict[str, Any]] = []

        with self._lock:
            for user_id, data in self.active_users.items():
                diff = now - data.get("last_seen_unix", 0)
                is_online = diff <= window_seconds
                is_idle = window_seconds < diff <= 1800.0  # within 30 minutes

                if is_online or is_idle:
                    results.append({
                        "client_ip": data["client_ip"],
                        "access_key": data["access_key"],
                        "status": "online" if is_online else "idle",
                        "status_label": "🟢 Online" if is_online else "⚪ Idle",
                        "last_seen": data["last_seen"],
                        "seconds_ago": int(diff),
                        "total_requests": data["total_requests"],
                        "last_action": data["last_action"],
                        "last_query": data["last_query"],
                        "user_agent": data["user_agent"],
                        "is_ip_banned": data["client_ip"] in self.banned_ips,
                        "is_key_banned": data["access_key"] in self.banned_keys and data["access_key"] != "ANONYMOUS"
                    })

        # Sort with online users first, then by recency
        results.sort(key=lambda u: (u["status"] != "online", u["seconds_ago"]))
        return results

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns the most recent events in reverse-chronological order (newest first)."""
        with self._lock:
            event_slice = list(self.events)
        event_slice.reverse()
        return event_slice[:limit]

    def is_blocked(self, client_ip: str, access_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Checks whether the client IP or access key is blacklisted.
        Returns (is_blocked, reason).
        """
        clean_ip = (client_ip or "").strip()
        clean_key = (access_key or "").strip().upper()

        with self._lock:
            if clean_ip in self.banned_ips:
                return True, f"IP address {clean_ip} has been blacklisted by administrator."
            if clean_key and clean_key in self.banned_keys and clean_key != "ANONYMOUS":
                return True, f"Access key {clean_key} has been revoked by administrator."

        return False, ""

    def toggle_block(
        self,
        target_type: str,
        target_value: str,
        block: bool = True
    ) -> Dict[str, Any]:
        """
        Blocks or unblocks an IP address or access key.
        target_type: 'ip' or 'key'
        """
        clean_val = (target_value or "").strip()
        t_type = (target_type or "").lower().strip()

        if not clean_val:
            return {"status": "error", "message": "Target value cannot be empty."}

        with self._lock:
            if t_type == "ip":
                if block:
                    self.banned_ips.add(clean_val)
                    action_msg = f"IP {clean_val} blacklisted successfully."
                else:
                    self.banned_ips.discard(clean_val)
                    action_msg = f"IP {clean_val} unbanned successfully."
            elif t_type in ("key", "access_key", "token"):
                clean_key = clean_val.upper()
                if block:
                    self.banned_keys.add(clean_key)
                    action_msg = f"Access key {clean_key} revoked successfully."
                else:
                    self.banned_keys.discard(clean_key)
                    action_msg = f"Access key {clean_key} restored successfully."
            else:
                return {"status": "error", "message": f"Invalid target_type: {target_type}. Must be 'ip' or 'key'."}

            self._save_state()

        logger.warning("Admin moderation action: %s", action_msg)
        return {
            "status": "success",
            "message": action_msg,
            "banned_ips": sorted(list(self.banned_ips)),
            "banned_keys": sorted(list(self.banned_keys))
        }

    def set_broadcast(self, message: str, active: bool = True) -> Dict[str, Any]:
        """Sets or toggles the system-wide announcement broadcast banner."""
        with self._lock:
            self.broadcast = {
                "message": (message or "").strip(),
                "active": bool(active and (message or "").strip()),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            self._save_state()

        logger.info("Admin broadcast updated: active=%s, message=%s", self.broadcast["active"], self.broadcast["message"])
        return {
            "status": "success",
            "broadcast": self.broadcast
        }

    def get_broadcast(self) -> Dict[str, Any]:
        """Returns the current broadcast configuration."""
        with self._lock:
            return dict(self.broadcast)

    def set_limits(self, global_hourly: int = 0, key_hourly: int = 0) -> Dict[str, Any]:
        """Sets dynamic rate limits."""
        with self._lock:
            self.limits = {
                "global_hourly_limit": max(0, int(global_hourly)),
                "key_hourly_limit": max(0, int(key_hourly))
            }
            self._save_state()
        return {"status": "success", "limits": self.limits}

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Returns comprehensive metrics for the admin dashboard."""
        now = time.time()
        uptime_seconds = int(now - self.start_time)
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        seconds = uptime_seconds % 60
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        online_users = self.get_online_users(window_seconds=300.0)
        recent_events = self.get_recent_events(limit=100)

        with self._lock:
            total_events_captured = len(self.events)
            banned_ips_list = sorted(list(self.banned_ips))
            banned_keys_list = sorted(list(self.banned_keys))
            broadcast_data = dict(self.broadcast)
            limits_data = dict(self.limits)

        # Calculate action counts in recent events
        action_distribution: Dict[str, int] = {}
        for ev in recent_events:
            act = ev.get("action_type", "OTHER")
            action_distribution[act] = action_distribution.get(act, 0) + 1

        return {
            "status": "online",
            "server_uptime": uptime_str,
            "uptime_seconds": uptime_seconds,
            "total_events_in_buffer": total_events_captured,
            "max_buffer_capacity": self.max_events,
            "online_users_count": len([u for u in online_users if u["status"] == "online"]),
            "idle_users_count": len([u for u in online_users if u["status"] == "idle"]),
            "total_tracked_sessions": len(online_users),
            "banned_ips_count": len(banned_ips_list),
            "banned_keys_count": len(banned_keys_list),
            "banned_ips": banned_ips_list,
            "banned_keys": banned_keys_list,
            "broadcast": broadcast_data,
            "limits": limits_data,
            "action_distribution": action_distribution,
            "online_users": online_users,
            "recent_events": recent_events,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    get_summary_stats = get_metrics_summary


# Global singleton instance of TelemetryManager
telemetry_manager = TelemetryManager()

