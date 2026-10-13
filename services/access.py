"""
Invite-only access store: invite-code whitelist, per-user profiles, credit ledger.

Authoritative local JSON store (data/access_store.json) for codes, onboarding
profiles, and credits. Supabase Auth supplies identity only; all invite/credit
state lives here so behavior is identical with or without Supabase tables.
Thread-safe, atomic writes, zero impact on search pipelines.

Seed codes come ONLY from the INVITE_CODES environment variable
(comma-separated). When it is empty, nothing is seeded and nothing stored is
purged, so existing codes, users and credits are never destroyed.
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


def _mask_code(value: Optional[str]) -> str:
    """Short public fingerprint for logs: first 4 chars + '...'. Never the raw value."""
    text = (value or "").strip()
    if len(text) <= 4:
        return "****"
    return text[:4] + "..."


def _load_invite_codes() -> List[str]:
    """
    Optional seed list from INVITE_CODES (comma-separated, set in the hosting
    dashboard). Empty/missing means: seed nothing; stored codes are kept as-is.
    """
    raw = os.getenv("INVITE_CODES") or ""
    codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
    return codes


# Part 1: whitelist of invite tokens, seeded ONLY from INVITE_CODES.
# Empty env -> no seeding and no purging (existing stored codes are preserved).
ACCESS_CODES: List[str] = _load_invite_codes()

DEFAULT_CREDITS = 25
GUEST_TRIAL_CREDITS = 10


def get_admin_password() -> str:
    """Admin password from ADMIN_PASSWORD env only. Empty string = admin disabled."""
    return (os.getenv("ADMIN_PASSWORD") or "").strip()


def verify_admin_password(candidate: Optional[str]) -> bool:
    """Constant-time-ish comparison. Always False when no ADMIN_PASSWORD is configured."""
    if not candidate:
        return False
    expected = get_admin_password()
    if not expected:
        return False
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
      codes: {CODE: {"active": bool, "balance": int|None, "bound_user_id":...,
                     "bound_email":..., "bound_at":..., "bound_ip":...}}
      users: {supabase_user_id: {"email":..., "full_name":..., "code":...,
                                 "ip":..., "created_at":..., "updated_at":...,
                                 "last_active":...}}
      guest_ips: {ip: {"balance": int, "total_consumed": int,
                       "created_at":..., "last_active":...}}
    Balances live on codes (seeded once at first redemption) and on guest IP
    rows (seeded once at first guest use). User records carry no credits.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Path("data") / "access_store.json")
        self._lock = threading.Lock()
        self.codes: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.guest_ips: Dict[str, Dict[str, Any]] = {}
        self._load_or_initialize()

    # -- persistence ------------------------------------------------------
    def _load_or_initialize(self):
        with self._lock:
            loaded_codes: Dict[str, Dict[str, Any]] = {}
            loaded_users: Dict[str, Dict[str, Any]] = {}
            loaded_guests: Dict[str, Dict[str, Any]] = {}
            if self.storage_path.exists():
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    loaded_codes = data.get("codes", {}) or {}
                    loaded_users = data.get("users", {}) or {}
                    loaded_guests = data.get("guest_ips", {}) or {}
                except Exception as e:
                    logger.error("Failed to load access store from %s: %s", self.storage_path, str(e))
            if not ACCESS_CODES:
                # Fail closed with zero destruction: no seed list configured, so
                # keep every stored code/user exactly as persisted. Add nothing,
                # delete nothing (purge skipped entirely).
                logger.warning(
                    "INVITE_CODES is empty: invite seeding and whitelist purge "
                    "disabled; existing stored codes preserved."
                )
                self.codes = loaded_codes
                self.users = loaded_users
            else:
                for code in ACCESS_CODES:
                    if code not in loaded_codes:
                        loaded_codes[code] = {
                            "active": True,
                            "balance": None,  # None = never redeemed; seeded once on first redemption
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
                        "active": True, "balance": None, "bound_user_id": None,
                        "bound_email": None, "bound_at": None, "bound_ip": None,
                    })
                self.users = loaded_users
            # Keep only well-formed guest rows (balance/total_consumed ints).
            self.guest_ips = {}
            for ip, g in loaded_guests.items():
                if not isinstance(g, dict):
                    continue
                try:
                    self.guest_ips[str(ip)] = {
                        "balance": max(0, int(g.get("balance", 0) or 0)),
                        "total_consumed": max(0, int(g.get("total_consumed", 0) or 0)),
                        "created_at": g.get("created_at") or _utcnow_iso(),
                        "last_active": g.get("last_active") or _utcnow_iso(),
                    }
                except (TypeError, ValueError):
                    continue
            # Migration: balances used to live on user records. Adopt each bound
            # user's balance onto its code (once), then drop the user-level field
            # so the code entry is the single source of truth going forward.
            for uid, u in self.users.items():
                if not isinstance(u, dict):
                    continue
                if "credits" in u:
                    legacy = u.pop("credits")
                    c = self.normalize_code(u.get("code", ""))
                    if c in self.codes and self.codes[c].get("balance") is None:
                        try:
                            self.codes[c]["balance"] = max(0, int(legacy or 0))
                        except (TypeError, ValueError):
                            pass
            self._save_unlocked()

    def _code_balance_unlocked(self, clean_code: str) -> Optional[int]:
        """Raw code balance; None = never redeemed (no seed applied yet)."""
        entry = self.codes.get(clean_code)
        if entry is None:
            return None
        bal = entry.get("balance")
        if bal is None:
            return None
        try:
            return max(0, int(bal))
        except (TypeError, ValueError):
            return 0

    def code_balance(self, code: str) -> Optional[int]:
        """Public balance read; None only when the code was never redeemed."""
        with self._lock:
            return self._code_balance_unlocked(self.normalize_code(code))

    def user_credits(self, user_id: str) -> Optional[int]:
        """Balance for a user's bound code; None when no bound/redeemed code."""
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                return None
            return self._code_balance_unlocked(self.normalize_code(user.get("code", "")))

    def _save_unlocked(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"codes": self.codes, "users": self.users, "guest_ips": self.guest_ips},
                    f, indent=2,
                )
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
        logger.warning("Admin deactivated access code %s", _mask_code(clean))
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
        Validate code, bind it to the user, record IP + timestamp. Credits are
        strictly code-tied: the 25-credit seed applies ONLY on a code's first
        redemption (balance is None). Re-completions, rebinds, and reloads
        never touch the existing balance.
        Returns (ok, message, http_status, user-with-credits).
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
            first_redemption = entry.get("balance") is None
            if first_redemption:
                entry["balance"] = DEFAULT_CREDITS
            existing = self.users.get(user_id)
            user = {
                "email": email,
                "full_name": name,
                "code": clean_code,
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
            balance = self._code_balance_unlocked(clean_code) or 0
        logger.info(
            "Onboarding %s for user %s (code %s, first=%s, balance=%d)",
            "completed" if first_redemption else "updated", user_id, _mask_code(clean_code),
            first_redemption, balance,
        )
        result = dict(user)
        result["credits"] = balance
        return True, "Profile completed. Application unlocked.", 200, result

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
        """
        Deduct 1 credit from the user's BOUND CODE balance (atomic, permanent).
        Only consumption decreases it; only admins raise it. Zero stays zero.
        """
        with self._lock:
            user = self.users.get(user_id)
            if not user or not user.get("code"):
                return False, "Profile setup required.", 403
            code = self.normalize_code(user.get("code", ""))
            entry = self.codes.get(code)
            if entry is None or not entry.get("active", False):
                return False, "Invalid or deactivated access code.", 403
            balance = self._code_balance_unlocked(code)
            if balance is None:
                # Defensive: active code with no seeded balance (should not happen
                # since completion seeds) — seed once rather than blocking.
                balance = DEFAULT_CREDITS
            if balance <= 0:
                return False, "Out of credits — contact the administrator for a top-up.", 402
            entry["balance"] = balance - 1
            user["last_active"] = _utcnow_iso()
            self._save_unlocked()
            return True, "ok", 200

    def _resolve_code_unlocked(
        self,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        code: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve any identifier to a whitelist code (or None). Caller holds the lock."""
        if code:
            clean = self.normalize_code(code)
            return clean if clean in self.codes else None
        if user_id and user_id in self.users:
            clean = self.normalize_code(self.users[user_id].get("code", ""))
            return clean if clean in self.codes else None
        if email:
            em = email.strip().lower()
            for uid, u in self.users.items():
                if str(u.get("email") or "").strip().lower() == em:
                    clean = self.normalize_code(u.get("code", ""))
                    return clean if clean in self.codes else None
        return None

    @staticmethod
    def _with_credits(user: Dict[str, Any], balance: int) -> Dict[str, Any]:
        """Response-shape copy carrying the code balance (kept out of storage)."""
        out = dict(user)
        out["credits"] = balance
        return out

    # -- guest IP trial (10 free credits, no account, server-side only) -----
    @staticmethod
    def normalize_ip(ip: Optional[str]) -> str:
        return ((ip or "").strip() or "127.0.0.1")

    def guest_status(self, ip: str, create: bool = False) -> Dict[str, Any]:
        """
        Read-only trial state for an IP. New IPs report the full trial
        balance WITHOUT writing a row unless create=True (pure reads must
        never provision ledger rows).
        """
        clean = self.normalize_ip(ip)
        now = _utcnow_iso()
        with self._lock:
            rec = self.guest_ips.get(clean)
            if rec is None:
                if not create:
                    return {
                        "balance": GUEST_TRIAL_CREDITS,
                        "total_consumed": 0,
                        "provisioned": False,
                        "created_at": None,
                        "last_active": None,
                    }
                rec = {
                    "balance": GUEST_TRIAL_CREDITS,
                    "total_consumed": 0,
                    "created_at": now,
                    "last_active": now,
                }
                self.guest_ips[clean] = rec
                self._save_unlocked()
            return {
                "balance": int(rec.get("balance", 0) or 0),
                "total_consumed": int(rec.get("total_consumed", 0) or 0),
                "provisioned": True,
                "created_at": rec.get("created_at"),
                "last_active": rec.get("last_active"),
            }

    def deduct_guest(self, ip: str) -> Tuple[bool, str, int]:
        """
        Atomically provision (first use: 10 credits) and deduct 1 trial
        credit for an IP. Exhausted IPs get (False, msg, 402) with a
        TRIAL_EXHAUSTED machine code at the endpoint layer — they can never
        reclaim free credits; only sign-in/code paths unlock further use.
        """
        clean = self.normalize_ip(ip)
        now = _utcnow_iso()
        with self._lock:
            rec = self.guest_ips.get(clean)
            if rec is None:
                rec = {
                    "balance": GUEST_TRIAL_CREDITS,
                    "total_consumed": 0,
                    "created_at": now,
                    "last_active": now,
                }
                self.guest_ips[clean] = rec
            balance = int(rec.get("balance", 0) or 0)
            if balance <= 0:
                rec["last_active"] = now
                self._save_unlocked()
                return (
                    False,
                    "You've used your 10 free trial credits on this network. "
                    "Sign in or enter an Access Code to continue.",
                    402,
                )
            rec["balance"] = balance - 1
            rec["total_consumed"] = int(rec.get("total_consumed", 0) or 0) + 1
            rec["last_active"] = now
            self._save_unlocked()
            return True, "ok", 200

    def adjust_credits(
        self,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        code: Optional[str] = None,
        delta: int = 0,
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        """
        Admin credit adjustment (+N/-N, floor 0) applied to the CODE balance.
        Lookup by id, email, or code. Zero stays zero unless admin raises it.
        """
        with self._lock:
            clean = self._resolve_code_unlocked(user_id=user_id, email=email, code=code)
            if clean is None:
                # Fall back to legacy bound-user lookup for identifiers whose
                # code row is missing (defensive; normally unreachable).
                target_id = None
                if user_id and user_id in self.users:
                    target_id = user_id
                if target_id is None:
                    return False, "No onboarded user found for identifier.", 404, None
                clean = self.normalize_code(self.users[target_id].get("code", ""))
                if clean not in self.codes:
                    return False, "No onboarded user found for identifier.", 404, None
            try:
                current = self._code_balance_unlocked(clean)
                current = current if current is not None else 0
                new_balance = max(0, current + int(delta))
            except (TypeError, ValueError):
                return False, "Delta must be an integer.", 400, None
            self.codes[clean]["balance"] = new_balance
            self.codes[clean]["updated_at"] = _utcnow_iso()
            self._save_unlocked()
            logger.warning("Admin adjusted credits for code %s by %+d -> %d", _mask_code(clean), int(delta), new_balance)
            bound_uid = self.codes[clean].get("bound_user_id")
            user_view = dict(self.users.get(bound_uid, {})) if bound_uid else {}
            user_view.setdefault("code", clean)
            return True, "Credits updated.", 200, self._with_credits(user_view, new_balance)

    def set_credits(
        self, user_id: str, amount: int
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        """Admin absolute balance set on the CODE balance (floor 0). Lookup by user_id."""
        with self._lock:
            user = self.users.get(user_id)
            if user is None:
                return False, "No onboarded user found for identifier.", 404, None
            clean = self.normalize_code(user.get("code", ""))
            if clean not in self.codes:
                return False, "No onboarded user found for identifier.", 404, None
            try:
                new_balance = max(0, int(amount))
            except (TypeError, ValueError):
                return False, "Amount must be an integer.", 400, None
            self.codes[clean]["balance"] = new_balance
            self.codes[clean]["updated_at"] = _utcnow_iso()
            self._save_unlocked()
            logger.warning("Admin set credits for code %s -> %d", _mask_code(clean), new_balance)
            return True, "Credits updated.", 200, self._with_credits(dict(user), new_balance)

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
            # Credits are strictly code-tied: read the code ledger, never the profile.
            code_credits = -1
            if code_entry is not None:
                bal = code_entry.get("balance")
                try:
                    code_credits = max(0, int(bal)) if bal is not None else 0
                except (TypeError, ValueError):
                    code_credits = 0
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
                    "credits": code_credits,
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


def code_for_bearer(raw_token: Optional[str]) -> Optional[str]:
    """
    Resolve a Supabase JWT to the bearer's redeemed access code WITHOUT any
    network call: base64-decode the payload, read `sub`, look up the profile.
    Display/attribution use only — never for authorization (endpoints still
    verify via supabase.auth.get_user). Returns None for non-JWT strings.
    """
    if not raw_token:
        return None
    token = raw_token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token.startswith("eyJ") or token.count(".") < 2:
        return None
    try:
        import base64
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        sub = str(payload.get("sub") or "").strip()
        if not sub:
            return None
    except Exception:
        return None
    with access_store._lock:
        user = access_store.users.get(sub)
        if not user:
            return None
        code = AccessStore.normalize_code(user.get("code", ""))
        return code or None


access_store = AccessStore()
