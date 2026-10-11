"""
Invite-only access store: 20-code whitelist, per-user profiles, credit ledger.

Authoritative local JSON store (data/access_store.json) for codes, onboarding
profiles, and credits. Supabase Auth supplies identity only; all invite/credit
state lives here so behavior is identical with or without Supabase tables.
Thread-safe, atomic writes, zero impact on search pipelines.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("academic_assistant.access")

# Part 1: whitelist of 20 structured high-entropy invite tokens.
ACCESS_CODES: List[str] = [
    "AXIOM-RESEARCH-9472",
    "SWMP-SYSTEMS-3819",
    "KERNEL-SCHOLAR-5184",
    "DISTRIBUTED-SYNC-7291",
    "LATENCY-ZERO-6435",
    "CONSENSUS-CORE-8203",
    "VECTOR-SYNTHESIS-1956",
    "FABRIC-SECURE-4078",
    "NEURAL-SCALE-8314",
    "PIPELINE-FLOW-5721",
    "STORAGE-TIER-9163",
    "EPBF-TRACING-2847",
    "CXL-MEMORY-6039",
    "INFERENCE-ACCEL-4158",
    "QUANTUM-GATE-7392",
    "ACCELERATOR-FABRIC-3281",
    "SERVERLESS-WARM-5920",
    "COMPUTE-MESH-8471",
    "ZERO-COPY-1635",
    "OPTIMAL-STRAGGLER-9512",
]

DEFAULT_CREDITS = 25
ADMIN_PASSWORD_FALLBACK = "AxiomAdmin#2026!Secure"


def get_admin_password() -> str:
    """Admin password from ADMIN_PASSWORD env, falling back to the provisioned default."""
    return os.getenv("ADMIN_PASSWORD", "").strip() or ADMIN_PASSWORD_FALLBACK


def verify_admin_password(candidate: Optional[str]) -> bool:
    """Constant-time-ish comparison against the configured admin password."""
    if not candidate:
        return False
    expected = get_admin_password()
    c = candidate.strip()
    if len(c) != len(expected):
        return False
    match = True
    for a, b in zip(c, expected):
        if a != b:
            match = False
    return match


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AccessStore:
    """
    Thread-safe JSON store:
      codes: {CODE: {"active": bool, "bound_user_id": str|None, "bound_email": str|None,
                     "bound_at": iso|None, "bound_ip": str|None}}
      users: {supabase_user_id: {"email":..., "full_name":..., "code":..., "credits": int,
                                 "ip":..., "created_at":..., "updated_at":..., "last_active":...}}
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Path("data") / "access_store.json")
        self._lock = threading.Lock()
        self.codes: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self._load_or_initialize()

    # -- persistence ------------------------------------------------------
    def _load_or_initialize(self):
        with self._lock:
            loaded_codes: Dict[str, Dict[str, Any]] = {}
            loaded_users: Dict[str, Dict[str, Any]] = {}
            if self.storage_path.exists():
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    loaded_codes = data.get("codes", {}) or {}
                    loaded_users = data.get("users", {}) or {}
                except Exception as e:
                    logger.error("Failed to load access store from %s: %s", self.storage_path, str(e))
            for code in ACCESS_CODES:
                if code not in loaded_codes:
                    loaded_codes[code] = {
                        "active": True,
                        "bound_user_id": None,
                        "bound_email": None,
                        "bound_at": None,
                        "bound_ip": None,
                    }
            # Drop stale codes no longer on the whitelist (keeps registry tight).
            self.codes = {c: loaded_codes[c] for c in ACCESS_CODES if c in loaded_codes}
            # Guarantee all whitelist codes present even if file predates them.
            for code in ACCESS_CODES:
                self.codes.setdefault(code, {
                    "active": True, "bound_user_id": None, "bound_email": None,
                    "bound_at": None, "bound_ip": None,
                })
            self.users = loaded_users
            self._save_unlocked()

    def _save_unlocked(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"codes": self.codes, "users": self.users}, f, indent=2)
            os.replace(temp_path, self.storage_path)
        except Exception as e:
            logger.error("Failed to persist access store: %s", str(e))

    # -- codes ------------------------------------------------------------
    @staticmethod
    def normalize_code(code: Optional[str]) -> str:
        return (code or "").strip().upper()

    def is_code_active(self, code: str) -> bool:
        with self._lock:
            entry = self.codes.get(self.normalize_code(code))
            return bool(entry and entry.get("active", False))

    def deactivate_code(self, code: str) -> Tuple[bool, str]:
        clean = self.normalize_code(code)
        with self._lock:
            entry = self.codes.get(clean)
            if entry is None:
                return False, f"Code {clean} is not on the whitelist."
            entry["active"] = False
            entry["deactivated_at"] = _utcnow_iso()
            self._save_unlocked()
        logger.warning("Admin deactivated access code %s", clean)
        return True, f"Code {clean} deactivated."

    def list_codes(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"code": c, **{k: v for k, v in info.items()}}
                for c, info in self.codes.items()
            ]

    # -- onboarding / profiles --------------------------------------------
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self.users.get(user_id)
            return dict(data) if data else None

    def complete_profile(
        self,
        user_id: str,
        email: Optional[str],
        full_name: str,
        access_code: str,
        client_ip: str,
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        """
        Validate code, bind it to the user, grant starting credits on first
        completion, record IP + timestamp. Returns (ok, message, http_status, user).
        """
        name = (full_name or "").strip()
        if not name:
            return False, "Full name is required.", 400, None
        clean_code = self.normalize_code(access_code)
        now = _utcnow_iso()
        with self._lock:
            entry = self.codes.get(clean_code)
            if entry is None or not entry.get("active", False):
                return False, "Invalid or deactivated access code.", 400, None
            # One code per user: rebinding to a different code is allowed but
            # never double-grants credits.
            existing = self.users.get(user_id)
            first_completion = existing is None or not existing.get("code")
            user = {
                "email": email,
                "full_name": name,
                "code": clean_code,
                "credits": (existing.get("credits", DEFAULT_CREDITS) if existing else DEFAULT_CREDITS),
                "ip": client_ip,
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
                "last_active": now,
            }
            self.users[user_id] = user
            entry["bound_user_id"] = user_id
            entry["bound_email"] = email
            entry["bound_at"] = now
            entry["bound_ip"] = client_ip
            self._save_unlocked()
        logger.info(
            "Onboarding %s for user %s (code %s, first=%s)",
            "completed" if first_completion else "updated", user_id, clean_code, first_completion,
        )
        return True, "Profile completed. Application unlocked.", 200, dict(user)

    def touch_user(self, user_id: str, client_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                return None
            user["last_active"] = _utcnow_iso()
            if client_ip:
                user["ip"] = client_ip
            self._save_unlocked()
            return dict(user)

    # -- credits ------------------------------------------------------------
    def deduct_credit(self, user_id: str) -> Tuple[bool, str, int]:
        """Deduct 1 credit. Returns (ok, message, http_status). 402 when empty."""
        with self._lock:
            user = self.users.get(user_id)
            if not user or not user.get("code"):
                return False, "Profile setup required.", 403
            code = self.normalize_code(user.get("code", ""))
            entry = self.codes.get(code)
            if entry is None or not entry.get("active", False):
                return False, "Invalid or deactivated access code.", 403
            balance = int(user.get("credits", 0) or 0)
            if balance <= 0:
                return False, "Out of credits — contact the administrator for a top-up.", 402
            user["credits"] = balance - 1
            user["last_active"] = _utcnow_iso()
            self._save_unlocked()
            return True, "ok", 200

    def adjust_credits(
        self,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        code: Optional[str] = None,
        delta: int = 0,
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        """Admin credit adjustment (+N/-N, floor 0). Lookup by id, email, or code."""
        target_id: Optional[str] = None
        with self._lock:
            if user_id and user_id in self.users:
                target_id = user_id
            elif email:
                em = email.strip().lower()
                for uid, u in self.users.items():
                    if str(u.get("email") or "").strip().lower() == em:
                        target_id = uid
                        break
            elif code:
                clean = self.normalize_code(code)
                for uid, u in self.users.items():
                    if self.normalize_code(u.get("code", "")) == clean:
                        target_id = uid
                        break
                if target_id is None and clean in self.codes:
                    bound = self.codes[clean].get("bound_user_id")
                    if bound and bound in self.users:
                        target_id = bound
            if target_id is None:
                return False, "No onboarded user found for identifier.", 404, None
            user = self.users[target_id]
            try:
                new_balance = max(0, int(user.get("credits", 0) or 0) + int(delta))
            except (TypeError, ValueError):
                return False, "Delta must be an integer.", 400, None
            user["credits"] = new_balance
            user["updated_at"] = _utcnow_iso()
            self._save_unlocked()
            logger.warning("Admin adjusted credits for %s by %+d -> %d", target_id, int(delta), new_balance)
            return True, "Credits updated.", 200, dict(user)

    def set_credits(
        self, user_id: str, amount: int
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        """Admin absolute balance set (floor 0). Lookup strictly by user_id."""
        with self._lock:
            user = self.users.get(user_id)
            if user is None:
                return False, "No onboarded user found for identifier.", 404, None
            try:
                new_balance = max(0, int(amount))
            except (TypeError, ValueError):
                return False, "Amount must be an integer.", 400, None
            user["credits"] = new_balance
            user["updated_at"] = _utcnow_iso()
            self._save_unlocked()
            logger.warning("Admin set credits for %s -> %d", user_id, new_balance)
            return True, "Credits updated.", 200, dict(user)

    # -- admin activity -------------------------------------------------------
    def build_activity_rows(
        self,
        events_24h: List[Dict[str, Any]],
        online_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Merge last-24h telemetry events with profile/credit state.
        Row: user_email, user_name, client_ip, access_code, credits, status,
             last_active, last_action, requests_24h.
        """
        with self._lock:
            users_snapshot = {uid: dict(u) for uid, u in self.users.items()}
            codes_snapshot = {c: dict(info) for c, info in self.codes.items()}
        code_to_user: Dict[str, Dict[str, Any]] = {}
        for uid, u in users_snapshot.items():
            c = self.normalize_code(u.get("code", ""))
            if c:
                code_to_user[c] = {"user_id": uid, **u}

        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for ev in events_24h:
            ip = ev.get("client_ip", "")
            key = ev.get("access_key", "")
            gkey = (ip, key)
            g = grouped.get(gkey)
            if g is None:
                grouped[gkey] = {
                    "client_ip": ip, "access_code": key,
                    "last_active": ev.get("timestamp", ""),
                    "last_action": ev.get("action_type", ""),
                    "requests_24h": 1,
                }
            else:
                g["requests_24h"] += 1
                if str(ev.get("timestamp", "")) > str(g["last_active"]):
                    g["last_active"] = ev.get("timestamp", "")
                    g["last_action"] = ev.get("action_type", "")

        rows: List[Dict[str, Any]] = []
        for (ip, key), g in grouped.items():
            profile = code_to_user.get(str(key).upper())
            code_entry = codes_snapshot.get(str(key).upper())
            if profile:
                status = "Active" if (code_entry and code_entry.get("active", False)) else "Revoked"
                # Fresher heartbeat from profile touches wins for last_active display.
                last_active = max(str(g["last_active"]), str(profile.get("last_active", "")))
                rows.append({
                    "user_id": profile.get("user_id", ""),
                    "user_email": profile.get("email") or "—",
                    "user_name": profile.get("full_name") or "—",
                    "client_ip": profile.get("ip") or ip,
                    "access_code": key,
                    "credits": int(profile.get("credits", 0) or 0),
                    "status": status,
                    "last_active": last_active,
                    "last_action": g["last_action"],
                    "requests_24h": g["requests_24h"],
                })
            else:
                rows.append({
                    "user_id": "",
                    "user_email": "—",
                    "user_name": "Guest / legacy",
                    "client_ip": ip,
                    "access_code": key,
                    "credits": -1,
                    "status": "Revoked" if (code_entry and not code_entry.get("active", False)) else "Unknown",
                    "last_active": g["last_active"],
                    "last_action": g["last_action"],
                    "requests_24h": g["requests_24h"],
                })
        rows.sort(key=lambda r: str(r["last_active"]), reverse=True)
        return rows


access_store = AccessStore()
