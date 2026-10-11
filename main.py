"""
Academic Research Assistant - FastAPI Main Application.
Provides concurrent academic search, streaming LLM synthesis via OrcaRouter/AI,
and a modern responsive single-page web interface.
"""

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

from services.search import (
    Paper,
    clean_academic_query,
    deduplicate_and_rank,
    discover_paper_code_urls,
    expand_semantic_concepts,
    fetch_arxiv_papers,
    fetch_entity_papers,
    fetch_openalex_papers,
    fetch_semantic_scholar_papers,
    search_academic_papers,
    search_cache,
    get_shared_http_client,
    set_shared_http_client,
    close_shared_http_client,
)
from services.llm import (
    ResearchLLMService,
    synthesis_cache,
    llm_semaphore,
    OutreachEmailRequest,
    generate_outreach_email,
    ComparePapersRequest,
    generate_compare_matrix,
)
from services.telemetry import telemetry_manager
from services.access import (
    access_store,
    verify_admin_password,
)
from services.db import (
    get_current_user,
    get_supabase_client,
    get_user_client,
    is_configured as supabase_configured,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("academic_assistant.main")

# Load environment variables
load_dotenv()

# App Configuration
ORCAROUTER_API_KEY = os.getenv("ORCAROUTER_API_KEY", "").strip()
ORCAROUTER_BASE_URL = os.getenv("ORCAROUTER_BASE_URL", "https://www.orcarouter.ai/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-v4-flash-free").strip()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip() or "AxiomAdmin#2026!Secure"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Initializes a single shared httpx.AsyncClient with persistent keep-alive connection pooling
    (20 keep-alive connections, 50 maximum concurrent connections) across all services.
    Ensures safe resource deallocation and connection drainage on server shutdown.
    """
    pool_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        timeout=30.0,
        follow_redirects=True,
    )
    set_shared_http_client(pool_client)
    logger.info("Axiom Connection Pool initialized: max_keepalive=20, max_connections=50")
    try:
        yield
    finally:
        await close_shared_http_client()
        logger.info("Axiom Connection Pool closed successfully")


# Initialize FastAPI App with Lifespan
app = FastAPI(
    title="Academic Research Assistant",
    description="Asynchronous Academic Research Assistant with ArXiv, Semantic Scholar, and OrcaRouter AI Synthesis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Templates
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ============================================================================
# Axiom Access Registry & IP-Binding Security Gate
# ============================================================================

class AccessRegistryManager:
    MASTER_CODE = "AXIOM-ROOT-V4HEWT"
    MASTER_ADMIN_KEYS: List[str] = ["AXIOM-MASTER-RESEARCH-2026", "AXIOM-ROOT-V4HEWT"]
    RESEARCHER_CODES = [
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
    EXPIRATION_DATE = "2027-12-31"

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self._load_or_initialize()

    def _load_or_initialize(self):
        with self._lock:
            loaded: Dict[str, Dict[str, Any]] = {}
            if self.storage_path.exists():
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                except Exception as e:
                    logger.error("Failed to load access registry from %s: %s", self.storage_path, str(e))

            # Master codes: unrestricted access, no expiration, multi-IP
            for m_code in self.MASTER_ADMIN_KEYS:
                if m_code not in loaded:
                    loaded[m_code] = {
                        "type": "master",
                        "bound_ip": None,
                        "expires_at": None,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

            # 20 invite codes: bound to single IP on first use. Legacy codes not on
            # the current whitelist are purged so retired tokens stop working.
            for code in list(loaded.keys()):
                info = loaded[code]
                if isinstance(info, dict) and info.get("type") == "researcher" and code not in self.RESEARCHER_CODES:
                    del loaded[code]
            for code in self.RESEARCHER_CODES:
                if code not in loaded:
                    loaded[code] = {
                        "type": "researcher",
                        "bound_ip": None,
                        "expires_at": self.EXPIRATION_DATE,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

            self.tokens = loaded
            self._save_unlocked()

    def _save_unlocked(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.tokens, f, indent=2)
            os.replace(temp_path, self.storage_path)
        except Exception as e:
            logger.error("Failed to persist access registry: %s", str(e))

    def validate_code(
        self,
        code: str,
        client_ip: str,
        bind_if_unbound: bool = True
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        clean_code = (code or "").strip().upper()
        with self._lock:
            if not clean_code or clean_code not in self.tokens:
                return False, "Invalid access code. Please check your credentials.", 401, None

            token_data = self.tokens[clean_code]
            token_type = token_data.get("type", "researcher")

            # 1. Master code: unrestricted access
            if token_type == "master":
                return True, "Master unrestricted access verified.", 200, token_data

            # 2. Researcher code: check expiration date (<= 2026-10-07)
            expires_at_str = token_data.get("expires_at")
            if expires_at_str:
                try:
                    expiry_date = datetime.strptime(expires_at_str, "%Y-%m-%d").date()
                    current_date = datetime.now(timezone.utc).date()
                    if current_date > expiry_date:
                        return False, "Code has expired.", 403, token_data
                except Exception as e:
                    logger.error("Error parsing expiration for code %s: %s", clean_code, str(e))

            # 3. Researcher code: check IP binding
            bound_ip = token_data.get("bound_ip")
            if bound_ip is None:
                if bind_if_unbound:
                    token_data["bound_ip"] = client_ip
                    token_data["bound_at"] = datetime.now(timezone.utc).isoformat()
                    self._save_unlocked()
                    logger.info("Bound researcher code %s permanently to client IP %s", clean_code, client_ip)
                return True, "Cohort researcher access verified.", 200, token_data

            if bound_ip != client_ip:
                logger.warning("Code %s rejected: bound to %s, received from %s", clean_code, bound_ip, client_ip)
                return False, "Code is locked to another machine/IP.", 403, token_data

            return True, "Cohort researcher access verified.", 200, token_data


class IPRateLimiter:
    """
    Tracks failed verification attempts per client IP.
    Blocks client IP for 15 minutes (900s) after 5 failed attempts.
    """
    def __init__(self, max_failures: int = 5, lockout_seconds: int = 900):
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self.failures: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def is_locked(self, ip: str) -> Tuple[bool, int]:
        now = time.time()
        with self._lock:
            attempts = self.failures.get(ip, [])
            valid_attempts = [t for t in attempts if (now - t) < self.lockout_seconds]
            self.failures[ip] = valid_attempts

            if len(valid_attempts) >= self.max_failures:
                remaining = int(self.lockout_seconds - (now - valid_attempts[-1]))
                return True, max(1, remaining)
            return False, 0

    def record_failure(self, ip: str):
        now = time.time()
        with self._lock:
            if ip not in self.failures:
                self.failures[ip] = []
            self.failures[ip].append(now)

    def reset(self, ip: str):
        with self._lock:
            self.failures.pop(ip, None)


access_registry = AccessRegistryManager(DATA_DIR / "access_registry.json")
ip_rate_limiter = IPRateLimiter(max_failures=5, lockout_seconds=900)


def get_client_ip(request: Request) -> str:
    """Extract original client IP from X-Forwarded-For, X-Real-IP, or direct connection."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    x_real = request.headers.get("x-real-ip")
    if x_real:
        return x_real.strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


async def verify_axiom_access(
    request: Request,
    x_axiom_key: Optional[str] = Header(None, alias="X-Axiom-Key"),
    access_code: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
) -> str:
    """
    FastAPI dependency that enforces Cohort Gate access verification.
    Accepts X-Axiom-Key header or query parameter (access_code / token) for EventSource SSE.
    """
    key = x_axiom_key or access_code or token
    if not key:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            key = auth_header[7:].strip()

    # Fallback for automated test suites if AXIOM_TEST_KEY environment variable is provided
    if not key and os.getenv("AXIOM_TEST_KEY"):
        key = os.getenv("AXIOM_TEST_KEY")

    if not key:
        raise HTTPException(
            status_code=401,
            detail="Access token required. Please authenticate via the Cohort Gate."
        )

    client_ip = get_client_ip(request)
    is_valid, msg, status_code, _ = access_registry.validate_code(
        code=key,
        client_ip=client_ip,
        bind_if_unbound=True
    )
    if not is_valid:
        raise HTTPException(status_code=status_code, detail=msg)

    return key
 
 
MASTER_ADMIN_KEYS: set[str] = {"AXIOM-MASTER-RESEARCH-2026", "AXIOM-ROOT-V4HEWT"}


def extract_request_access_key(request: Request) -> Optional[str]:
    """
    Extracts client access key or passkey from headers, query params, cookies, or Bearer auth.
    Used by telemetry interceptor and admin verification.
    """
    key = (
        request.headers.get("x-admin-key")
        or request.headers.get("x-axiom-key")
    )
    if not key:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            key = auth_header[7:].strip()
    if not key:
        key = (
            request.query_params.get("access_code")
            or request.query_params.get("token")
            or request.query_params.get("key")
            or request.query_params.get("admin_key")
        )
    if not key:
        key = (
            request.cookies.get("axiom_access_token")
            or request.cookies.get("axiom_admin_session")
        )
    return key.strip() if key else None


def verify_master_admin(request: Request) -> str:
    """
    Validates that the incoming request has Master Admin privileges.
    Accepts the ADMIN_PASSWORD, legacy master keys, X-Admin-Key / X-Axiom-Key
    headers, ?key= query parameter, or axiom_admin_session cookie.
    Returns HTTP 404 Not Found on failure (instead of 401/403) to ensure security through obscurity
    against scanners and scrapers.
    """
    raw_key = extract_request_access_key(request)
    if not raw_key:
        raise HTTPException(status_code=404, detail="Not Found")

    if verify_admin_password(raw_key):
        return "ADMIN_PASSWORD_SESSION"

    clean_key = raw_key.strip().upper()
    if clean_key in MASTER_ADMIN_KEYS or clean_key == access_registry.MASTER_CODE:
        return clean_key

    # Check registered tokens for master role
    with access_registry._lock:
        token_info = access_registry.tokens.get(clean_key)
        if token_info and token_info.get("type") == "master":
            return clean_key

    raise HTTPException(status_code=404, detail="Not Found")


@app.middleware("http")
async def telemetry_and_moderation_middleware(request: Request, call_next):
    """
    Asynchronous telemetry interceptor and moderation security middleware.
    1. Moderation: Evaluates client IP and Access Key against the blacklist, returning 403 if revoked.
    2. Telemetry: Records timestamp, client IP, key, latency, status code, and preview in in-memory ring buffer.
    Never consumes streaming request bodies to preserve SSE and file transfers.
    """
    path = request.url.path
    if path.startswith("/static/"):
        return await call_next(request)

    client_ip = get_client_ip(request)
    raw_key = extract_request_access_key(request)
    # Never log raw Supabase JWTs in the ring buffer (length + eyJ prefix heuristic).
    if raw_key and len(raw_key) > 64 and raw_key.startswith("eyJ"):
        raw_key = "SUPABASE_USER"
    access_key = raw_key if raw_key else "UNAUTHENTICATED / VISITOR"

    # 1. Moderation Blacklist Check (check IP and raw_key if present)
    is_blocked, reason = telemetry_manager.is_blocked(client_ip, raw_key)
    if is_blocked:
        # Allow Master Admin override for admin routes
        is_admin_override = False
        if path.startswith("/admin") or path.startswith("/api/admin"):
            clean_k = (raw_key or "").strip().upper()
            if clean_k in MASTER_ADMIN_KEYS:
                is_admin_override = True

        if not is_admin_override:
            action_type = telemetry_manager.classify_action_type(path, request.method)
            telemetry_manager.record_event(
                client_ip=client_ip,
                access_key=access_key,
                action_type="BLOCKED_" + action_type,
                query_preview="Access revoked: " + str(reason),
                path=path,
                method=request.method,
                status_code=403,
                latency_ms=0.0,
                user_agent=request.headers.get("user-agent", "")
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Access revoked by administrator", "detail": reason}
            )

    # 2. Request execution with telemetry timing
    start_time = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        latency_ms = (time.time() - start_time) * 1000.0
        action_type = telemetry_manager.classify_action_type(path, request.method)
        query_preview = (
            request.query_params.get("query")
            or request.query_params.get("inquiry")
            or request.query_params.get("name")
            or request.query_params.get("author")
            or request.query_params.get("target_professor")
            or request.query_params.get("q")
            or ""
        )
        telemetry_manager.record_event(
            client_ip=client_ip,
            access_key=access_key,
            action_type=action_type,
            query_preview=query_preview,
            path=path,
            method=request.method,
            status_code=status_code,
            latency_ms=latency_ms,
            user_agent=request.headers.get("user-agent", "")
        )


runtime_config: Dict[str, str] = {
    "orcarouter_api_key": ORCAROUTER_API_KEY,
    "orcarouter_base_url": ORCAROUTER_BASE_URL,
    "model_name": MODEL_NAME,
    "semantic_scholar_api_key": SEMANTIC_SCHOLAR_API_KEY,
}

# LLM Service instance
llm_service = ResearchLLMService(
    api_key=runtime_config["orcarouter_api_key"],
    base_url=runtime_config["orcarouter_base_url"],
    model_name=runtime_config["model_name"]
)


class SearchRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Research query or topic")
    inquiry: Optional[str] = Field(default=None, description="Research query or topic (alias)")
    author: Optional[str] = Field(default=None, description="Filter by author name")
    institution: Optional[str] = Field(default=None, description="Filter by institution or university name")
    categories: Optional[List[str]] = Field(default=None, description="Filter by subject categories")
    top_k: int = Field(default=10, ge=1, le=100)
    limit: Optional[int] = Field(default=None, ge=1, le=100, description="Target paper count (10, 25, 50, 100)")
    semantic_scholar_key: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_query_inquiry(cls, data: Any) -> Any:
        if isinstance(data, dict):
            q = data.get("query")
            inq = data.get("inquiry")
            val = q or inq or ""
            data["query"] = val
            data["inquiry"] = val
        return data


class SynthesizeRequest(BaseModel):
    inquiry: str = Field(..., description="Research query or topic")
    papers: Optional[List[Paper]] = Field(default=None, description="Pre-fetched papers if available")
    orcarouter_key: Optional[str] = None


class SinglePaperAnalyzeRequest(BaseModel):
    paper: Paper = Field(..., description="Paper metadata to analyze")
    inquiry: Optional[str] = Field(default="", description="Original user inquiry or research context")
    depth: Optional[str] = Field(default="balanced", description="Analysis depth: quick, balanced, or deep")
    orcarouter_key: Optional[str] = Field(default=None, description="Optional override OrcaRouter API key")


SinglePaperSynthesizeRequest = SinglePaperAnalyzeRequest


class PaperDiagramRequest(BaseModel):
    title: str = Field(..., description="Paper title to model")
    abstract: Optional[str] = Field(default="", description="Paper abstract")
    tldr: Optional[str] = Field(default="", description="Paper TLDR summary")
    orcarouter_key: Optional[str] = Field(default=None, description="Optional override OrcaRouter API key")


class ThesisProposalRequest(BaseModel):
    paper: Paper = Field(..., description="Target academic paper to generate thesis draft from")
    inquiry: Optional[str] = Field(default="", description="Research focus or user context")
    orcarouter_key: Optional[str] = Field(default=None, description="Optional override OrcaRouter API key")


class PapersCompareRequest(BaseModel):
    papers: List[Paper] = Field(..., min_length=2, max_length=5, description="2 to 5 papers to compare")
    inquiry: Optional[str] = Field(default="", description="Research context for comparative analysis")
    orcarouter_key: Optional[str] = Field(default=None, description="Optional override OrcaRouter API key")


class NotebookLMExportRequest(BaseModel):
    papers: List[Paper] = Field(default_factory=list, description="Papers to bundle for NotebookLM")
    inquiry: Optional[str] = Field(default="Academic Literature Analysis", description="Research inquiry title")
    notes: Optional[str] = Field(default="", description="Optional synthesis or analysis notes to bundle")


class LiteratureReviewExportRequest(BaseModel):
    inquiry: str = Field(default="Academic Literature Review", description="Literature review title")
    papers: List[Paper] = Field(default_factory=list, description="Surveyed papers to include")
    synthesis_content: Optional[str] = Field(default="", description="Markdown synthesis content")


class AuthVerifyRequest(BaseModel):
    access_code: str = Field(..., description="Axiom Cohort Access Code")


class ConfigUpdateRequest(BaseModel):
    orcarouter_api_key: Optional[str] = None
    orcarouter_base_url: Optional[str] = None
    model_name: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None


class ModerationBlockRequest(BaseModel):
    target_type: str = Field(..., description="'ip' or 'key'")
    target_value: str = Field(..., description="IP address or Access Key to block/unblock")
    action: str = Field(default="block", description="'block' or 'unblock'")


class BroadcastRequest(BaseModel):
    message: str = Field(default="", description="Alert announcement text")
    active: bool = Field(default=True, description="Whether alert banner is active")


class AdminLimitsRequest(BaseModel):
    global_hourly_limit: Optional[int] = Field(default=None)
    key_hourly_limit: Optional[int] = Field(default=None)


# ============================================================================
# Invite-Only Access Gate, Onboarding, Credits & Admin Code Management
# Supabase Auth supplies identity; invite codes / profiles / credits live in
# services/access.py. Metered routes use resolve_access (Supabase path first,
# legacy Axiom cohort gate as fallback) instead of verify_axiom_access.
# Machine-readable outcomes travel in the X-Access-Code response header so
# existing string `detail` parsing on clients never breaks.
# ============================================================================

class OnboardingCompleteRequest(BaseModel):
    full_name: str = Field(..., max_length=120)
    access_code: str = Field(..., max_length=64)


class CreditAdjustRequest(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    code: Optional[str] = None
    delta: int = 0


class CodeDeactivateRequest(BaseModel):
    code: str


def _access_denied(status_code: int, machine_code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=message,
        headers={"X-Access-Code": machine_code},
    )


async def resolve_access(
    request: Request,
    x_axiom_key: Optional[str] = Header(None, alias="X-Axiom-Key"),
    access_code: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
    sb_token: Optional[str] = Query(None),
) -> str:
    """
    Unified gate for metered endpoints (search + AI generation, 1 credit each).
    1. Supabase invite path: valid Bearer/sb_token + completed profile + active
       code + credits>0 -> deduct 1 credit, allow (no cohort code needed).
    2. Legacy path: existing Axiom cohort gate, unchanged, no credit burn.
    """
    sb_user = await get_current_user(request.headers.get("authorization"), sb_token)
    if sb_user is not None:
        profile = access_store.get_user(sb_user["id"])
        if not profile or not profile.get("code"):
            raise _access_denied(403, "ONBOARDING_REQUIRED", "Profile setup required. Complete onboarding to unlock.")
        access_store.touch_user(sb_user["id"], get_client_ip(request))
        ok, msg, status = access_store.deduct_credit(sb_user["id"])
        if not ok:
            machine = "CODE_REVOKED" if status == 403 else "CREDITS_EXHAUSTED"
            raise _access_denied(status, machine, msg)
        return f"sb:{sb_user['id']}"

    # Legacy cohort path (unchanged behavior).
    key = x_axiom_key or access_code or token
    if not key:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            key = auth_header[7:].strip()
    if not key and os.getenv("AXIOM_TEST_KEY"):
        key = os.getenv("AXIOM_TEST_KEY")
    if not key:
        raise HTTPException(status_code=401, detail="Access token required. Please authenticate via the Cohort Gate.")
    client_ip = get_client_ip(request)
    is_valid, msg, status_code, _ = access_registry.validate_code(
        code=key, client_ip=client_ip, bind_if_unbound=True
    )
    if not is_valid:
        raise HTTPException(status_code=status_code, detail=msg)
    return key


@app.get("/api/onboarding/status")
async def onboarding_status(request: Request):
    """Return onboarding state + credit balance for the Supabase caller (guest-safe)."""
    sb_user = await get_current_user(
        request.headers.get("authorization"), request.query_params.get("sb_token")
    )
    if sb_user is None:
        return {"completed": False, "guest": True, "credits": 0}
    profile = access_store.get_user(sb_user["id"])
    if not profile or not profile.get("code"):
        return {"completed": False, "guest": False, "credits": 0, "email": sb_user.get("email")}
    code_active = access_store.is_code_active(profile.get("code", ""))
    return {
        "completed": code_active,
        "guest": False,
        "email": sb_user.get("email"),
        "full_name": profile.get("full_name", ""),
        "code": profile.get("code", ""),
        "credits": int(profile.get("credits", 0) or 0),
        "code_active": code_active,
    }


@app.post("/api/onboarding/complete")
async def onboarding_complete(payload: OnboardingCompleteRequest, request: Request):
    """
    Mandatory gated step: full name + invite code. Validates against the active
    whitelist, binds the code, grants 25 starting credits, records IP + time.
    """
    sb_user = await get_current_user(
        request.headers.get("authorization"), request.query_params.get("sb_token")
    )
    if sb_user is None:
        raise HTTPException(status_code=401, detail="Sign-in required before profile setup.")
    ok, msg, status, user = access_store.complete_profile(
        user_id=sb_user["id"],
        email=sb_user.get("email"),
        full_name=payload.full_name,
        access_code=payload.access_code,
        client_ip=get_client_ip(request),
    )
    if not ok:
        raise HTTPException(status_code=status, detail=msg)
    assert user is not None
    return {
        "status": "ok",
        "message": msg,
        "completed": True,
        "full_name": user.get("full_name", ""),
        "code": user.get("code", ""),
        "credits": int(user.get("credits", 0) or 0),
    }


@app.post("/api/admin/code/deactivate")
async def admin_deactivate_code(payload: CodeDeactivateRequest, request: Request):
    """
    Instantly revoke an invite code: whitelist flag off, cohort registry token
    dropped, telemetry key-ban added — terminating that user/code immediately.
    """
    verify_master_admin(request)
    clean = access_store.normalize_code(payload.code)
    ok, msg = access_store.deactivate_code(clean)
    with access_registry._lock:
        access_registry.tokens.pop(clean, None)
        access_registry._save_unlocked()
    telemetry_manager.toggle_block("key", clean, True)
    if not ok:
        # Still report if the key wasn't whitelisted but got banned anyway.
        return {"status": "ok", "message": msg + " Key additionally blacklisted.", "code": clean}
    return {"status": "ok", "message": msg, "code": clean}


@app.post("/api/admin/credits/adjust")
async def admin_adjust_credits(payload: CreditAdjustRequest, request: Request):
    """Real-time credit top-up/deduction by user_id, email, or invite code."""
    verify_master_admin(request)
    ok, msg, status, user = access_store.adjust_credits(
        user_id=payload.user_id, email=payload.email, code=payload.code, delta=payload.delta
    )
    if not ok:
        raise HTTPException(status_code=status, detail=msg)
    assert user is not None
    return {"status": "ok", "message": msg, "credits": int(user.get("credits", 0) or 0), "user": user}


@app.get("/api/admin/telemetry")
async def admin_telemetry_24h(request: Request):
    """
    24-hour activity log: user email/name, client IP, access code used, current
    credits, status pill (Active/Revoked/Unknown), last active, actions toolbar data.
    """
    verify_master_admin(request)
    cutoff = time.time() - 86400.0
    with telemetry_manager._lock:
        events_24h = [e for e in telemetry_manager.events if float(e.get("unix_time", 0)) >= cutoff]
    rows = access_store.build_activity_rows(events_24h)
    revoked_codes = [c["code"] for c in access_store.list_codes() if not c.get("active", True)]
    return {
        "status": "ok",
        "window_hours": 24,
        "row_count": len(rows),
        "rows": rows,
        "codes": access_store.list_codes(),
        "revoked_codes": revoked_codes,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# Supabase Persistence API (sessions history, artifacts, saved papers)
# Auth: Supabase Bearer tokens via get_current_user — fully independent of the
# Axiom cohort gate so guest/offline search pipelines are never affected.
# Writes require auth (401); reads fall back gracefully ([]) for guests.
# ============================================================================

class SessionCreateRequest(BaseModel):
    title: str = Field(default="Untitled session", max_length=200)
    search_mode: str = Field(default="topic", max_length=32)
    query_params: Optional[Dict[str, Any]] = Field(default=None)
    discovered_paper_ids: Optional[List[str]] = Field(default=None)
    session_id: Optional[str] = Field(default=None, description="If set, update this session instead of inserting")


class MessageCreateRequest(BaseModel):
    session_id: str
    role: str = Field(max_length=16)
    content: str


class ArtifactCreateRequest(BaseModel):
    session_id: Optional[str] = Field(default=None)
    artifact_type: str = Field(max_length=32)
    title: str = Field(default="", max_length=200)
    target_paper_id: Optional[str] = Field(default=None, max_length=128)
    content: str


class SavedPaperCreateRequest(BaseModel):
    paper_id: str = Field(max_length=256)
    paper_metadata: Optional[Dict[str, Any]] = Field(default=None)


ARTIFACT_TYPES = {"outreach", "thesis", "ieee_review"}


def _require_supabase():
    """Return (client_or_None, error_response_or_None)."""
    client = get_supabase_client()
    if client is None:
        return None, JSONResponse(
            status_code=503,
            content={"error": "History backend not configured (Supabase unavailable)."},
        )
    return client, None


def _require_user(user: Optional[Dict[str, Any]]):
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required. Please sign in."},
        )
    return None


@app.get("/api/history/sessions")
async def list_history_sessions(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Return the current user's sessions ordered by created_at desc (guest: [])."""
    if not user:
        return {"sessions": []}
    client, err = _require_supabase()
    if err is not None:
        return {"sessions": []}
    try:
        uclient = get_user_client(user["token"])
        res = (
            (uclient or client).table("research_sessions")
            .select("*")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .execute()
        )
        return {"sessions": res.data or []}
    except Exception as e:
        logger.error("Failed to list history sessions: %s", str(e))
        return {"sessions": []}


@app.post("/api/history/sessions")
async def create_history_session(
    payload: SessionCreateRequest,
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Create (or update, when session_id is given) a research session. Auth required."""
    auth_err = _require_user(user)
    if auth_err is not None:
        return auth_err
    client, err = _require_supabase()
    if err is not None:
        return err
    assert user is not None
    try:
        uclient = get_user_client(user["token"]) or client
        if payload.session_id:
            res = (
                uclient.table("research_sessions")
                .update({
                    "title": payload.title,
                    "search_mode": payload.search_mode,
                    "query_params": payload.query_params or {},
                    "discovered_paper_ids": payload.discovered_paper_ids or [],
                })
                .eq("id", payload.session_id)
                .eq("user_id", user["id"])
                .execute()
            )
            rows = res.data or []
            if not rows:
                raise HTTPException(status_code=404, detail="Session not found.")
            return {"status": "ok", "session_id": rows[0].get("id", payload.session_id)}
        res = (
            uclient.table("research_sessions")
            .insert({
                "user_id": user["id"],
                "title": payload.title,
                "search_mode": payload.search_mode,
                "query_params": payload.query_params or {},
                "discovered_paper_ids": payload.discovered_paper_ids or [],
            })
            .execute()
        )
        rows = res.data or []
        if not rows or not rows[0].get("id"):
            raise HTTPException(status_code=502, detail="Session insert returned no id.")
        return {"status": "ok", "session_id": rows[0]["id"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create history session: %s", str(e))
        return JSONResponse(status_code=502, content={"error": f"Failed to save session: {str(e)}"})


@app.get("/api/history/sessions/{session_id}")
async def get_history_session(
    session_id: str,
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Return session details plus related session_messages (guest/foreign: 404-safe {})."""
    if not user:
        return {"session": None, "messages": []}
    client, err = _require_supabase()
    if err is not None:
        return {"session": None, "messages": []}
    try:
        uclient = get_user_client(user["token"]) or client
        sess = (
            uclient.table("research_sessions")
            .select("*")
            .eq("id", session_id)
            .eq("user_id", user["id"])
            .execute()
        )
        rows = sess.data or []
        if not rows:
            return {"session": None, "messages": []}
        msgs = (
            uclient.table("session_messages")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return {"session": rows[0], "messages": msgs.data or []}
    except Exception as e:
        logger.error("Failed to fetch history session %s: %s", session_id, str(e))
        return {"session": None, "messages": []}


@app.post("/api/history/messages")
async def append_history_message(
    payload: MessageCreateRequest,
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Append a {session_id, role, content} message. Auth required."""
    auth_err = _require_user(user)
    if auth_err is not None:
        return auth_err
    client, err = _require_supabase()
    if err is not None:
        return err
    assert user is not None
    if payload.role not in ("user", "assistant", "system"):
        return JSONResponse(status_code=400, content={"error": "role must be user|assistant|system."})
    try:
        uclient = get_user_client(user["token"]) or client
        own = (
            uclient.table("research_sessions")
            .select("id")
            .eq("id", payload.session_id)
            .eq("user_id", user["id"])
            .execute()
        )
        if not (own.data or []):
            raise HTTPException(status_code=404, detail="Session not found.")
        res = (
            uclient.table("session_messages")
            .insert({
                "session_id": payload.session_id,
                "role": payload.role,
                "content": payload.content,
            })
            .execute()
        )
        rows = res.data or []
        return {"status": "ok", "message_id": rows[0].get("id") if rows else None}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to append history message: %s", str(e))
        return JSONResponse(status_code=502, content={"error": f"Failed to save message: {str(e)}"})


@app.get("/api/artifacts")
async def list_artifacts(
    artifact_type: Optional[str] = Query(None, description="Filter: outreach|thesis|ieee_review"),
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Fetch the current user's saved artifacts (guest: [])."""
    if not user:
        return {"artifacts": []}
    client, err = _require_supabase()
    if err is not None:
        return {"artifacts": []}
    try:
        uclient = get_user_client(user["token"]) or client
        q = (
            uclient.table("saved_artifacts")
            .select("*")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
        )
        if artifact_type:
            q = q.eq("artifact_type", artifact_type)
        res = q.execute()
        return {"artifacts": res.data or []}
    except Exception as e:
        logger.error("Failed to list artifacts: %s", str(e))
        return {"artifacts": []}


@app.post("/api/artifacts")
async def create_artifact(
    payload: ArtifactCreateRequest,
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Save an outreach/thesis/ieee_review artifact. Auth required."""
    auth_err = _require_user(user)
    if auth_err is not None:
        return auth_err
    client, err = _require_supabase()
    if err is not None:
        return err
    assert user is not None
    if payload.artifact_type not in ARTIFACT_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": "artifact_type must be one of: outreach, thesis, ieee_review."},
        )
    if not payload.content.strip():
        return JSONResponse(status_code=400, content={"error": "content cannot be empty."})
    try:
        uclient = get_user_client(user["token"]) or client
        res = (
            uclient.table("saved_artifacts")
            .insert({
                "user_id": user["id"],
                "session_id": payload.session_id,
                "artifact_type": payload.artifact_type,
                "title": payload.title,
                "target_paper_id": payload.target_paper_id,
                "content": payload.content,
            })
            .execute()
        )
        rows = res.data or []
        return {"status": "ok", "artifact_id": rows[0].get("id") if rows else None}
    except Exception as e:
        logger.error("Failed to save artifact: %s", str(e))
        return JSONResponse(status_code=502, content={"error": f"Failed to save artifact: {str(e)}"})


@app.get("/api/papers/saved")
async def list_saved_papers(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Return the current user's bookmarked papers (guest: [])."""
    if not user:
        return {"papers": []}
    client, err = _require_supabase()
    if err is not None:
        return {"papers": []}
    try:
        uclient = get_user_client(user["token"]) or client
        res = (
            uclient.table("saved_papers")
            .select("*")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .execute()
        )
        return {"papers": res.data or []}
    except Exception as e:
        logger.error("Failed to list saved papers: %s", str(e))
        return {"papers": []}


@app.post("/api/papers/saved")
async def save_paper_bookmark(
    payload: SavedPaperCreateRequest,
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Bookmark a paper (idempotent). Auth required."""
    auth_err = _require_user(user)
    if auth_err is not None:
        return auth_err
    client, err = _require_supabase()
    if err is not None:
        return err
    assert user is not None
    if not payload.paper_id.strip():
        return JSONResponse(status_code=400, content={"error": "paper_id cannot be empty."})
    try:
        uclient = get_user_client(user["token"]) or client
        existing = (
            uclient.table("saved_papers")
            .select("paper_id")
            .eq("user_id", user["id"])
            .eq("paper_id", payload.paper_id)
            .execute()
        )
        if existing.data:
            return {"status": "ok", "paper_id": payload.paper_id, "deduped": True}
        uclient.table("saved_papers").insert({
            "user_id": user["id"],
            "paper_id": payload.paper_id,
            "paper_metadata": payload.paper_metadata or {},
        }).execute()
        return {"status": "ok", "paper_id": payload.paper_id, "deduped": False}
    except Exception as e:
        logger.error("Failed to save paper bookmark: %s", str(e))
        return JSONResponse(status_code=502, content={"error": f"Failed to save paper: {str(e)}"})


@app.delete("/api/papers/saved/{paper_id}")
async def delete_paper_bookmark(
    paper_id: str,
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Remove a bookmark. Auth required."""
    auth_err = _require_user(user)
    if auth_err is not None:
        return auth_err
    client, err = _require_supabase()
    if err is not None:
        return err
    assert user is not None
    try:
        uclient = get_user_client(user["token"]) or client
        uclient.table("saved_papers").delete().eq("user_id", user["id"]).eq(
            "paper_id", paper_id
        ).execute()
        return {"status": "ok", "paper_id": paper_id}
    except Exception as e:
        logger.error("Failed to delete paper bookmark: %s", str(e))
        return JSONResponse(status_code=502, content={"error": f"Failed to remove paper: {str(e)}"})



@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the single-page application UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_name": runtime_config["model_name"],
            "has_orcarouter_key": bool(runtime_config["orcarouter_api_key"]),
            "has_ss_key": bool(runtime_config["semantic_scholar_api_key"]),
            "base_url": runtime_config["orcarouter_base_url"],
        }
    )


@app.post("/api/auth/verify")
async def verify_auth_code_endpoint(payload: AuthVerifyRequest, request: Request):
    """
    Verify client access token against the closed cohort registry.
    Handles rate limiting (15m lockout after 5 failures) and machine IP binding.
    """
    client_ip = get_client_ip(request)

    # 1. Check rate limit
    locked, remaining = ip_rate_limiter.is_locked(client_ip)
    if locked:
        return JSONResponse(
            status_code=429,
            content={
                "valid": False,
                "error": f"Too many failed verification attempts. Access locked for {remaining} seconds.",
                "remaining_seconds": remaining,
            }
        )

    # 2. Check access code
    is_valid, msg, status_code, token_data = access_registry.validate_code(
        code=payload.access_code,
        client_ip=client_ip,
        bind_if_unbound=True
    )

    if not is_valid:
        ip_rate_limiter.record_failure(client_ip)
        return JSONResponse(
            status_code=status_code,
            content={
                "valid": False,
                "error": msg,
            }
        )

    # Success: clear failed attempts
    ip_rate_limiter.reset(client_ip)
    resp: Dict[str, Any] = {
        "valid": True,
        "type": token_data.get("type", "researcher") if token_data else "researcher",
        "message": msg,
    }
    if token_data and token_data.get("expires_at"):
        resp["expires_at"] = token_data["expires_at"]

    return resp


@app.get("/health")
async def health_check():
    """Health check and configuration status."""
    return {
        "status": "healthy",
        "service": "academic-research-assistant",
        "model_name": runtime_config["model_name"],
        "orcarouter_configured": bool(runtime_config["orcarouter_api_key"]),
        "semantic_scholar_configured": bool(runtime_config["semantic_scholar_api_key"]),
        "orcarouter_base_url": runtime_config["orcarouter_base_url"],
    }


@app.get("/api/health")
async def api_health_check():
    """Lightweight server health-check endpoint that responds instantly."""
    return {"status": "online", "version": "1.0.0"}


@app.post("/api/config", dependencies=[Depends(verify_axiom_access)])
async def update_runtime_config(payload: ConfigUpdateRequest):
    """Update runtime API keys or endpoints directly from the UI without restart."""
    global llm_service
    if payload.orcarouter_api_key is not None:
        runtime_config["orcarouter_api_key"] = payload.orcarouter_api_key.strip()
    if payload.orcarouter_base_url is not None and payload.orcarouter_base_url.strip():
        runtime_config["orcarouter_base_url"] = payload.orcarouter_base_url.strip()
    if payload.model_name is not None and payload.model_name.strip():
        runtime_config["model_name"] = payload.model_name.strip()
    if payload.semantic_scholar_api_key is not None:
        runtime_config["semantic_scholar_api_key"] = payload.semantic_scholar_api_key.strip()

    # Re-initialize LLM service
    llm_service = ResearchLLMService(
        api_key=runtime_config["orcarouter_api_key"],
        base_url=runtime_config["orcarouter_base_url"],
        model_name=runtime_config["model_name"]
    )

    return {
        "status": "success",
        "model_name": runtime_config["model_name"],
        "orcarouter_configured": bool(runtime_config["orcarouter_api_key"]),
        "semantic_scholar_configured": bool(runtime_config["semantic_scholar_api_key"]),
    }


@app.post("/api/search", dependencies=[Depends(resolve_access)])
async def search_endpoint(payload: SearchRequest):
    """
    Direct asynchronous academic search endpoint.
    Fetches concurrently from ArXiv, Semantic Scholar, and OpenAlex,
    discovers code repositories, deduplicates, and ranks top papers.
    """
    ss_key = payload.semantic_scholar_key or runtime_config.get("semantic_scholar_api_key")
    effective_top_k = payload.limit or payload.top_k
    target_inquiry = payload.query or payload.inquiry or ""
    papers, meta = await search_academic_papers(
        inquiry=target_inquiry,
        semantic_scholar_api_key=ss_key,
        top_k=effective_top_k,
        author=payload.author,
        institution=payload.institution,
        categories=payload.categories
    )
    return {
        "inquiry": target_inquiry,
        "query": target_inquiry,
        "keywords": meta.get("extracted_keywords"),
        "papers": [p.model_dump() for p in papers],
        "metadata": meta
    }


@app.get("/api/search/entity", dependencies=[Depends(resolve_access)])
async def search_entity_endpoint(
    entity_type: str = Query(..., description="Entity type: 'author' or 'institution'"),
    name: str = Query(..., description="Researcher or institution name"),
    limit: int = Query(50, ge=1, le=100, description="Paper count up to 100")
):
    """
    Dedicated endpoint to query all works by a specific researcher or institution without requiring a topic.
    Queries OpenAlex works index, discovers code repositories, and returns normalized papers.
    """
    clean_type = (entity_type or "").strip().lower()
    if clean_type not in ("author", "institution"):
        raise HTTPException(status_code=400, detail="entity_type must be 'author' or 'institution'")
    clean_name = (name or "").strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="name parameter is required")

    papers = await fetch_entity_papers(entity_type=clean_type, name=clean_name, limit=limit)
    return {
        "entity_type": clean_type,
        "name": clean_name,
        "count": len(papers),
        "papers": [p.model_dump() for p in papers]
    }


@app.post("/api/synthesize", dependencies=[Depends(resolve_access)])
async def synthesize_endpoint(payload: SynthesizeRequest):
    """Non-streaming complete academic synthesis."""
    orca_key = payload.orcarouter_key or runtime_config["orcarouter_api_key"]

    papers = payload.papers
    meta = {}
    if papers is None:
        papers, meta = await search_academic_papers(
            inquiry=payload.inquiry,
            semantic_scholar_api_key=runtime_config.get("semantic_scholar_api_key"),
            top_k=6
        )

    report = await llm_service.synthesize(
        inquiry=payload.inquiry,
        papers=papers,
        custom_api_key=orca_key
    )

    return {
        "inquiry": payload.inquiry,
        "papers": [p.model_dump() for p in papers],
        "synthesis": report,
        "metadata": meta
    }


def sse_event(data: Any) -> str:
    """Format payload dictionary as a standard Server-Sent Event (SSE) message."""
    return "data: " + json.dumps(data) + "\n\n"


@app.get("/api/research/stream", dependencies=[Depends(resolve_access)])
async def research_stream_endpoint(
    query: str = Query("", description="User research question, topic, author, or institution name"),
    limit: int = Query(10, ge=1, le=100, description="Target number of papers to retrieve (10, 25, 50, 100)"),
    search_mode: str = Query("topic", description="Search mode: 'topic', 'author', 'institution', or 'custom' (combined topic+author+institution)"),
    author: Optional[str] = Query(None, description="Optional author filter (used by topic/custom modes)"),
    institution: Optional[str] = Query(None, description="Optional institution filter (used by topic/custom modes)"),
    orcarouter_key: Optional[str] = Query(None, description="Optional override OrcaRouter API key"),
    semantic_scholar_key: Optional[str] = Query(None, description="Optional override Semantic Scholar API key")
):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint.
    Emits live pipeline stages for topic exploration, dedicated entity profiles (author/institution),
    or custom combined search (topic + author + institution simultaneously):
      1. 'keywords': Query sanitized & academic terms extracted
      2. 'fetching_arxiv': ArXiv querying CS categories
      3. 'fetching_semanticscholar': Semantic Scholar querying
      4. 'fetching_openalex': OpenAlex peer-reviewed index querying
      5. 'papers_ready': Emits deduplicated & code-prioritized papers (up to 100)
      6. 'discovery_ready': Literature overview summary
      7. 'done': Pipeline complete
    Standalone 'topic'/'author'/'institution' flows are unchanged; 'custom' reuses the
    topic pipeline with author/institution filters applied via the shared query builder.
    """
    import asyncio
    import httpx

    normalized_mode = (search_mode or "topic").strip().lower()
    clean_query = (query or "").strip()
    clean_author = (author or "").strip() or None
    clean_institution = (institution or "").strip() or None

    # Custom/advanced search requires at least one of topic, author, institution.
    if normalized_mode == "custom" and not clean_query and not clean_author and not clean_institution:
        raise HTTPException(status_code=400, detail="Custom search requires at least one of: query, author, or institution.")

    effective_orca_key = (orcarouter_key or runtime_config["orcarouter_api_key"] or "").strip()
    effective_ss_key = (semantic_scholar_key or runtime_config.get("semantic_scholar_api_key") or "").strip()

    async def event_generator():
        try:
            # Check if this is an entity search (author or university/institution)
            # NOTE: 'custom' intentionally falls through to the combined topic pipeline below.
            if normalized_mode in ("author", "institution"):
                clean_mode = normalized_mode
                entity_cache_key = f"entity:{clean_mode}:{clean_query.lower()}:{limit}"
                cached_entity = search_cache.get(entity_cache_key)
                if cached_entity is not None and cached_entity[0]:
                    cached_papers, cached_meta = cached_entity
                    cached_slice = cached_papers[:limit]
                    papers_payload = [p.dict() if hasattr(p, "dict") else (p.model_dump() if hasattr(p, "model_dump") else p) for p in cached_slice]
                    yield sse_event({"type": "status", "stage": "cache_hit", "message": f"Instant cache hit: Retrieving pre-indexed {clean_mode} publications (0ms network calls)..."})
                    await asyncio.sleep(0.04)
                    yield sse_event({"type": "status", "stage": "keywords_ready", "keywords": clean_query, "message": f"Target {clean_mode}: '{clean_query}' (from cache)"})
                    await asyncio.sleep(0.04)
                    yield sse_event({
                        "type": "papers_ready",
                        "stage": "papers_ready",
                        "count": len(cached_slice),
                        "papers": papers_payload,
                        "metadata": cached_meta,
                        "message": f"Retrieved {len(cached_slice)} cached publications for {clean_query}."
                    })
                    await asyncio.sleep(0.04)
                    yield sse_event({"type": "status", "stage": "discovery_ready", "message": f"{clean_mode.capitalize()} literature discovery complete (cache hit)."})
                    entity_summary = (
                        f"### {clean_mode.capitalize()} Profile: {clean_query} (Cached)\n\n"
                        f"- **Discovered Publications**: **{len(cached_slice)}** works retrieved from cache.\n"
                        f"- **Ranking Metric**: Sorted by academic citation impact and code availability.\n"
                        f"- **Sidebar Available**: Publications are loaded in the **Discovered Papers** sidebar with affiliations and open-access links.\n\n"
                        f"> [!TIP]\n"
                        f"> Use the instant filter in the sidebar to search within these works, or select papers for deep synthesis and comparison."
                    )
                    yield sse_event({"type": "token", "content": entity_summary})
                    yield sse_event({"type": "done", "message": f"{clean_mode.capitalize()} discovery complete (cache hit)."})
                    return

                yield sse_event({"type": "status", "stage": "extract_keywords", "message": f"Initializing {clean_mode} lookup for '{clean_query}'..."})
                await asyncio.sleep(0.05)
                yield sse_event({"type": "status", "stage": "keywords_ready", "keywords": clean_query, "message": f"Target {clean_mode}: '{clean_query}'"})
                yield sse_event({"type": "status", "stage": "fetching_openalex", "message": f"Querying OpenAlex works index for {clean_mode} '{clean_query}' (up to {limit} papers)..."})

                entity_papers = await fetch_entity_papers(entity_type=clean_mode, name=clean_query, limit=limit)

                yield sse_event({"type": "status", "stage": "discovering_code", "message": "Scanning papers for open-source code repositories..."})
                await asyncio.sleep(0.04)

                entity_meta = {"entity_type": clean_mode, "name": clean_query, "final_count": len(entity_papers), "openalex_count": len(entity_papers)}
                papers_payload = [p.dict() if hasattr(p, "dict") else (p.model_dump() if hasattr(p, "model_dump") else p) for p in entity_papers]
                yield sse_event({
                    "type": "papers_ready",
                    "stage": "papers_ready",
                    "count": len(entity_papers),
                    "papers": papers_payload,
                    "metadata": entity_meta,
                    "message": f"Discovered {len(entity_papers)} works by {clean_query} ranked by citations."
                })
                await asyncio.sleep(0.04)

                yield sse_event({"type": "status", "stage": "discovery_ready", "message": f"{clean_mode.capitalize()} literature discovery complete."})

                entity_summary = (
                    f"### {clean_mode.capitalize()} Profile: {clean_query}\n\n"
                    f"- **Discovered Publications**: **{len(entity_papers)}** peer-reviewed works retrieved from OpenAlex index.\n"
                    f"- **Ranking Metric**: Sorted in descending order of citation count and code availability.\n"
                    f"- **Sidebar Available**: All publications are loaded in the **Discovered Papers** sidebar with affiliations and open-access links.\n\n"
                    f"> [!TIP]\n"
                    f"> Use the instant filter in the sidebar to search within these works, or select papers for deep synthesis and comparison."
                )
                yield sse_event({"type": "token", "content": entity_summary})
                yield sse_event({"type": "done", "message": f"{clean_mode.capitalize()} discovery complete."})
                return

            # Topic / Custom Combined Exploration Pipeline (multi-field query builder)
            # 'topic' = standalone topic; 'custom' = topic + author + institution combined.
            cache_key = f"{clean_query}|author:{clean_author or ''}|inst:{clean_institution or ''}|k:{limit}"
            cached_entry = search_cache.get(cache_key) or search_cache.get(clean_query)
            if cached_entry is not None:
                cached_papers, cached_meta = cached_entry
                selected_papers = cached_papers[:limit]
                logger.info("Stream research: cache hit for %r (%d papers)", cache_key, len(selected_papers))
                keywords = cached_meta.get("extracted_keywords") or clean_academic_query(clean_query)
                keywords_label = keywords or "(filters only — no topic keywords)"
                yield sse_event({"type": "status", "stage": "cache_hit", "message": "Instant cache hit: Retrieving pre-indexed academic literature (0ms network calls)..."})
                await asyncio.sleep(0.04)

                payload = {
                    "type": "status",
                    "stage": "keywords_ready",
                    "keywords": keywords,
                    "message": f"Target keywords: '{keywords_label}' (from cache)",
                }
                yield sse_event(payload)
                await asyncio.sleep(0.04)

                yield sse_event({
                    "type": "papers_ready",
                    "stage": "papers_ready",
                    "count": len(selected_papers),
                    "papers": [p.dict() if hasattr(p, "dict") else p.model_dump() for p in selected_papers],
                    "metadata": cached_meta,
                    "message": f"Retrieved {len(selected_papers)} cached high-relevance papers."
                })
                await asyncio.sleep(0.04)

                yield sse_event({"type": "status", "stage": "discovery_ready", "message": "Literature discovery complete (cache hit). Ready for on-demand analysis."})

                source_breakdown = f"ArXiv: {cached_meta.get('arxiv_count', 0)} | Semantic Scholar: {cached_meta.get('ss_count', 0)} | OpenAlex: {cached_meta.get('openalex_count', 0)}"
                overview_summary = (
                    f"### Literature Discovery Overview (Cached)\n\n"
                    f"- **Extracted Academic Keywords**: `{keywords_label}`\n"
                    f"- **Discovered Papers**: **{len(selected_papers)}** high-signal papers retrieved from cache ({source_breakdown}).\n"
                    f"- **Sidebar Available**: All identified papers are rendered with metadata, citations, and PDF links in the **Discovered Papers** sidebar.\n\n"
                    f"> [!TIP]\n"
                    f"> Click **'Explain Paper with AI'** on any paper card in the sidebar to generate a focused, on-demand breakdown covering its problem statement, technical approach, research gaps, and proposed thesis extensions."
                )
                yield sse_event({"type": "token", "content": overview_summary})
                yield sse_event({"type": "done", "message": "Literature discovery complete (cache hit)."})
                return

            # Stage 1: Semantic intent extraction (stage names unchanged per SSE contract)
            yield sse_event({"type": "status", "stage": "extract_keywords", "message": "Extracting semantic intent and conceptual equivalents..."})
            await asyncio.sleep(0.05)

            keywords = clean_academic_query(clean_query)
            # Custom-search hardening: author/institution-only queries yield keywords == "".
            # Keep "" for fetchers (filter-only) but show a readable label in the status event.
            keywords_label = keywords or "(filters only — no topic keywords)"
            semantic_concepts = expand_semantic_concepts(keywords) if keywords else []
            payload = {
                "type": "status",
                "stage": "keywords_ready",
                "keywords": keywords,
                "semantic_concepts": semantic_concepts,
                "message": f"Target intent: '{keywords_label}'" + (f" (+{len(semantic_concepts)} related concepts)" if semantic_concepts else ""),
            }
            yield sse_event(payload)

            # Stage 2: Concurrently query ArXiv, Semantic Scholar, and OpenAlex
            yield sse_event({"type": "status", "stage": "fetching_arxiv", "message": "Fetching ArXiv papers in CS categories (cs.DC, cs.SE, cs.AI, cs.AR)..."})
            yield sse_event({"type": "status", "stage": "fetching_semanticscholar", "message": "Querying Semantic Scholar API for citations & open-access PDFs..."})
            yield sse_event({"type": "status", "stage": "fetching_openalex", "message": "Querying OpenAlex works index for peer-reviewed & open-access literature..."})

            metadata: Dict[str, Any] = {
                "raw_inquiry": clean_query,
                "extracted_keywords": keywords,
                "semantic_intent": keywords,
                "semantic_concepts": semantic_concepts,
                "retrieval_mode": "semantic",
                "author_filter": clean_author,
                "institution_filter": clean_institution,
                "search_mode": normalized_mode,
                "arxiv_count": 0,
                "ss_count": 0,
                "openalex_count": 0,
                "arxiv_status": "pending",
                "ss_status": "pending",
                "openalex_status": "pending",
                "errors": [],
                "cache_hit": False,
            }

            fetch_quota = min(max(limit, 10), 100)
            client = get_shared_http_client()

            arxiv_task = fetch_arxiv_papers(keywords, client, max_results=fetch_quota, author=clean_author)
            ss_task = fetch_semantic_scholar_papers(
                keywords,
                client,
                api_key=effective_ss_key or None,
                limit=fetch_quota
            )
            openalex_task = fetch_openalex_papers(keywords, client, limit=fetch_quota, author=clean_author, institution=clean_institution)

            results = await asyncio.gather(arxiv_task, ss_task, openalex_task, return_exceptions=True)

            # Process ArXiv results
            arxiv_papers: List[Paper] = []
            if isinstance(results[0], Exception):
                err = f"ArXiv error: {str(results[0])}"
                metadata["errors"].append(err)
                metadata["arxiv_status"] = "failed"
            else:
                arxiv_papers = results[0]
                metadata["arxiv_count"] = len(arxiv_papers)
                metadata["arxiv_status"] = "success" if arxiv_papers else "empty"

            # Process Semantic Scholar results
            ss_papers: List[Paper] = []
            if isinstance(results[1], Exception):
                err = f"Semantic Scholar error: {str(results[1])}"
                metadata["errors"].append(err)
                metadata["ss_status"] = "failed"
            else:
                ss_papers = results[1]
                metadata["ss_count"] = len(ss_papers)
                metadata["ss_status"] = "success" if ss_papers else "empty"

            # Process OpenAlex results
            openalex_papers: List[Paper] = []
            if isinstance(results[2], Exception):
                err = f"OpenAlex error: {str(results[2])}"
                metadata["errors"].append(err)
                metadata["openalex_status"] = "failed"
            else:
                openalex_papers = results[2]
                metadata["openalex_count"] = len(openalex_papers)
                metadata["openalex_status"] = "success" if openalex_papers else "empty"

            # Stage 3: Discover code and deduplicate
            yield sse_event({"type": "status", "stage": "discovering_code", "message": "Scanning repositories and discoverable code implementations..."})
            all_candidates = arxiv_papers + ss_papers + openalex_papers
            await discover_paper_code_urls(all_candidates, client)

            yield sse_event({"type": "status", "stage": "deduplicating", "message": f"Deduplicating across {len(arxiv_papers)} ArXiv, {len(ss_papers)} Semantic Scholar, and {len(openalex_papers)} OpenAlex papers..."})
            final_papers = deduplicate_and_rank(
                arxiv_papers,
                ss_papers,
                openalex_papers,
                top_k=limit,
                author=clean_author,
                institution=clean_institution,
                query=keywords,
            )
            metadata["final_count"] = len(final_papers)

            # Store in search cache
            search_cache.set(cache_key, final_papers, metadata)

            # Stage 4: Emit papers_ready event
            yield sse_event({
                "type": "papers_ready",
                "stage": "papers_ready",
                "count": len(final_papers),
                "papers": [p.dict() if hasattr(p, "dict") else p.model_dump() for p in final_papers],
                "metadata": metadata,
                "message": f"Discovered {len(final_papers)} high-relevance papers across multi-source repositories."
            })
            await asyncio.sleep(0.05)

            # Stage 5: Search Overview Summary
            yield sse_event({"type": "status", "stage": "discovery_ready", "message": "Literature discovery complete. Ready for on-demand analysis."})

            source_breakdown = f"ArXiv: {metadata['arxiv_count']} | Semantic Scholar: {metadata['ss_count']} | OpenAlex: {metadata['openalex_count']}"
            overview_summary = (
                f"### Literature Discovery Overview\n\n"
                f"- **Extracted Academic Keywords**: `{keywords}`\n"
                f"- **Discovered Papers**: **{len(final_papers)}** high-signal papers retrieved and prioritized ({source_breakdown}).\n"
                f"- **Multi-Source Repositories**: Cross-referenced ArXiv CS, Semantic Scholar, and OpenAlex.\n"
                f"- **Sidebar Available**: All identified papers are rendered with metadata, citations, open-source code links, and PDF links in the **Discovered Papers** sidebar.\n\n"
                f"> [!TIP]\n"
                f"> Click **'Explain Paper with AI'** on any paper card in the sidebar to generate a focused, on-demand breakdown covering its problem statement, technical approach, research gaps, and proposed thesis extensions."
            )
            yield sse_event({"type": "token", "content": overview_summary})

            # Stage 6: Done
            yield sse_event({"type": "done", "message": "Literature discovery complete."})

        except Exception as e:
            logger.error("Unhandled error in research stream: %s", str(e), exc_info=True)
            yield sse_event({"type": "error", "message": f"Pipeline error: {str(e)}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/paper/synthesize", dependencies=[Depends(resolve_access)])
@app.post("/api/paper/analyze", dependencies=[Depends(resolve_access)])
async def analyze_single_paper_endpoint(payload: SinglePaperAnalyzeRequest):
    """
    On-demand single-paper LLM synthesis streaming endpoint.
    Supports multi-tier analysis depths:
      - quick: 3 concise bullet points (~100-150 words, max_tokens=250)
      - balanced: 4-section summary (~250-350 words, max_tokens=600)
      - deep: comprehensive breakdown with Mermaid.js diagram (~700+ words, max_tokens=1500)
    """
    effective_key = (payload.orcarouter_key or runtime_config["orcarouter_api_key"] or "").strip()
    depth_norm = (payload.depth or "balanced").lower()
    if depth_norm not in ("quick", "balanced", "deep"):
        depth_norm = "balanced"

    async def event_generator():
        try:
            model_display = runtime_config["model_name"]
            paper_title = payload.paper.title
            payload_msg = {
                "type": "status",
                "stage": "starting",
                "message": f"Initiating AI [{depth_norm.capitalize()}] synthesis for '{paper_title}' with {model_display}...",
            }
            yield sse_event(payload_msg)

            async for token in llm_service.stream_single_paper_analysis(
                paper=payload.paper,
                inquiry=payload.inquiry or "",
                depth=depth_norm,
                custom_api_key=effective_key
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'message': f'Paper synthesis [{depth_norm}] complete.'})}\n\n"

        except Exception as e:
            logger.error("Error in single paper analysis stream: %s", str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': f'Analysis error: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/paper/diagram", dependencies=[Depends(resolve_access)])
@app.post("/api/diagram", dependencies=[Depends(resolve_access)])
async def generate_paper_diagram_endpoint(payload: PaperDiagramRequest):
    """
    On-demand professional architecture diagram engine using PlantUML & public Kroki API.
    1. Extracts/maps distributed system architecture into PlantUML via AI.
    2. Renders into high-resolution SVG via Kroki API with automatic resilient fallback.
    """
    try:
        effective_key = (payload.orcarouter_key or runtime_config["orcarouter_api_key"] or "").strip()
        plantuml = await llm_service.generate_plantuml_architecture(
            title=payload.title,
            abstract=payload.abstract or "",
            tldr=payload.tldr or "",
            custom_api_key=effective_key
        )
        svg = await llm_service.render_plantuml_kroki(
            plantuml_code=plantuml,
            paper_title=payload.title,
            abstract=payload.abstract or "",
            tldr=payload.tldr or "",
            enable_fallback=True
        )
        return {"svg": svg, "status": "success", "plantuml": plantuml}
    except Exception as e:
        logger.error("Diagram generation failed: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=502,
            content={
                "error": f"Diagram generation failed: {str(e)}",
                "details": "Kroki diagram service could not render the architecture."
            }
        )


@app.post("/api/paper/outreach-email", dependencies=[Depends(resolve_access)])
async def generate_outreach_email_endpoint(payload: OutreachEmailRequest):
    """
    On-demand Professor Cold Outreach Email Generator.
    Returns a sub-120-word faculty outreach draft (subject, body, word_count)
    with bracketed placeholders, built from structured paper metadata.
    """
    if not payload.paper_title or not payload.paper_title.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Field 'paper_title' cannot be empty."}
        )
    if not payload.target_professor or not payload.target_professor.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Field 'target_professor' cannot be empty."}
        )
    if payload.scope_type not in ("single_paper", "holistic_lab"):
        return JSONResponse(
            status_code=400,
            content={"error": "Field 'scope_type' must be 'single_paper' or 'holistic_lab'."}
        )

    try:
        if not payload.orcarouter_key:
            payload.orcarouter_key = runtime_config["orcarouter_api_key"] or None

        result = await llm_service.generate_outreach_email(payload)
        body_text = str(result.get("body", ""))
        word_count = result.get("word_count")
        if not isinstance(word_count, int):
            word_count = len(body_text.split()) if body_text else 0
        return {
            "status": "success",
            "subject": result.get("subject", ""),
            "body": body_text,
            "word_count": word_count,
            "scope_type": payload.scope_type,
            "target_professor": payload.target_professor,
        }
    except Exception as e:
        logger.error("Outreach email generation failed: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to generate outreach email: {str(e)}",
                "status": "error"
            }
        )


@app.post("/api/thesis/draft", dependencies=[Depends(resolve_access)])
async def thesis_draft_endpoint(payload: ThesisProposalRequest):
    """
    On-demand academic thesis proposal generator endpoint.
    Streams a formal ~250-word proposal draft containing Problem Statement,
    Proposed Delta / Methodology, Evaluation Metrics, and Expected Contribution.
    """
    effective_key = (payload.orcarouter_key or runtime_config["orcarouter_api_key"] or "").strip()

    async def event_generator():
        try:
            model_display = runtime_config["model_name"]
            paper_title = payload.paper.title
            payload_msg = {
                "type": "status",
                "stage": "starting",
                "message": f"Drafting Academic Thesis Proposal for '{paper_title}' with {model_display}...",
            }
            yield sse_event(payload_msg)
            await asyncio.sleep(0.05)

            async for token in llm_service.stream_thesis_proposal(
                paper=payload.paper,
                inquiry=payload.inquiry or "",
                custom_api_key=effective_key
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'message': 'Thesis proposal draft complete.'})}\n\n"

        except Exception as e:
            logger.error("Error in thesis proposal stream: %s", str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': f'Proposal drafting error: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/papers/compare", dependencies=[Depends(resolve_access)])
async def papers_compare_endpoint(payload: PapersCompareRequest):
    """
    Multi-paper comparative systems analysis endpoint.
    Generates an analytical comparison matrix contrasting:
    Problem, Infrastructure / Cloud Stack, Scalability Bottlenecks, and Evaluation Datasets.
    """
    effective_key = (payload.orcarouter_key or runtime_config["orcarouter_api_key"] or "").strip()

    async def event_generator():
        try:
            model_display = runtime_config["model_name"]
            yield f"data: {json.dumps({'type': 'status', 'stage': 'starting', 'message': f'Synthesizing Comparative Systems Matrix across {len(payload.papers)} papers with {model_display}...'})}\n\n"
            await asyncio.sleep(0.05)

            async for token in llm_service.stream_papers_comparison(
                papers=payload.papers,
                inquiry=payload.inquiry or "",
                custom_api_key=effective_key
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'message': 'Comparative analysis matrix complete.'})}\n\n"

        except Exception as e:
            logger.error("Error in comparative matrix stream: %s", str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': f'Comparison error: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/paper/compare-matrix", dependencies=[Depends(resolve_access)])
async def paper_compare_matrix_endpoint(payload: ComparePapersRequest):
    """
    Authoritative cross-paper trade-off matrix endpoint.
    Generates a structured Markdown table comparing 2 to 4 papers across 5 required dimensions:
    1. Fundamental Problem Addressed
    2. Core Architectural Mechanism
    3. Key Assumptions & Constraints
    4. Performance / Complexity Ceiling
    5. Engineering Trade-offs & Failure Modes
    """
    if len(payload.papers) < 2 or len(payload.papers) > 4:
        raise HTTPException(
            status_code=400,
            detail="Must provide between 2 and 4 papers for comparison matrix."
        )

    for i, p in enumerate(payload.papers):
        if not isinstance(p, dict) or not (p.get("title") or "").strip():
            raise HTTPException(
                status_code=400,
                detail=f"Paper at index {i} must be a valid dictionary with a non-empty 'title'."
            )

    effective_key = (payload.orcarouter_key or runtime_config["orcarouter_api_key"] or "").strip()

    try:
        matrix_md = await llm_service.generate_compare_matrix(
            papers=payload.papers,
            custom_api_key=effective_key
        )
        return {
            "status": "success",
            "markdown": matrix_md,
            "paper_count": len(payload.papers)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Error in compare-matrix endpoint: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate trade-off matrix: {str(e)}")


@app.post("/api/export/notebooklm", dependencies=[Depends(verify_axiom_access)])
async def export_notebooklm_endpoint(payload: NotebookLMExportRequest):
    """
    Bundle discovered papers into a single clean Markdown source pack
    specifically formatted for instant drag-and-drop into Google NotebookLM.
    """
    clean_title = payload.inquiry.strip() or "Academic Systems Research"
    lines = [
        f"# NotebookLM Source Pack: {clean_title}",
        f"**Curated By**: SWMP Labs",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        f"**Total Indexed Literature Sources**: {len(payload.papers)}",
        f"",
        f"---",
        f"",
    ]
    if payload.notes:
        lines.extend([
            f"## Research Synthesis & Analytical Notes",
            payload.notes.strip(),
            f"",
            f"---",
            f"",
        ])

    for idx, p in enumerate(payload.papers, 1):
        authors_str = ", ".join(p.authors) if p.authors else "Author Unlisted"
        link = p.pdf_url or p.url or (f"https://arxiv.org/abs/{p.id.replace('arxiv:', '')}" if "arxiv" in p.id else "https://semanticscholar.org")
        year_str = f"{p.year}" if p.year else "Recent"
        citations = f" | Citations: {p.citation_count}" if p.citation_count is not None else ""
        lines.extend([
            f"## Source {idx}: {p.title}",
            f"- **Authors**: {authors_str}",
            f"- **Publication Year**: {year_str} | **Repository**: {p.source}{citations}",
            f"- **Direct Canonical Link / PDF**: {link}",
        ])
        if p.tldr:
            lines.append(f"- **Executive TLDR**: {p.tldr}")
        lines.extend([
            f"",
            f"### Abstract & Methodology",
            p.abstract or "No abstract text provided in repository index.",
            f"",
            f"---",
            f"",
        ])

    content = "\n".join(lines)
    clean_filename = re.sub(r"[^a-zA-Z0-9]", "_", payload.inquiry)[:35]
    filename = f"Axiom_NotebookLM_{clean_filename or 'Source_Pack'}.md"

    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/api/export/literature-review", dependencies=[Depends(verify_axiom_access)])
async def export_literature_review_endpoint(payload: LiteratureReviewExportRequest):
    """
    Generate and export an IEEE-styled formal literature review document
    formatted for publication or thesis integration.
    """
    clean_title = payload.inquiry.strip().title() if payload.inquiry else "Distributed Systems Computing"
    lines = [
        f"# Systematic Literature Review: {clean_title}",
        f"**Author / Literature Engine**: SWMP Labs (Open-Source Initiative)",
        f"**Date**: {time.strftime('%B %Y')}",
        f"",
        f"## Abstract",
        f"This systematic literature review surveys recent architectural advancements, algorithmic paradigms, and open research challenges in *{payload.inquiry}*. "
        f"Drawing from {len(payload.papers)} peer-reviewed and pre-print publications indexed across ArXiv and Semantic Scholar, "
        f"we categorize the state of the art, synthesize core design trade-offs, and outline high-impact thesis extensions.",
        f"",
        f"---",
        f"",
        f"## 1. Introduction & Research Scope",
        f"The acceleration of modern workloads necessitates scalable, fault-tolerant, and low-overhead computing infrastructure. "
        f"This review examines foundational literature addressing: *\"{payload.inquiry}\"*.",
        f"",
        f"---",
        f"",
    ]

    if payload.synthesis_content:
        lines.extend([
            f"## 2. In-Depth Synthesis & Systems Analysis",
            payload.synthesis_content.strip(),
            f"",
            f"---",
            f"",
        ])

    lines.extend([
        f"## 3. Taxonomy of Surveyed Literature",
        f"",
        f"| Reference | Key Contribution | Primary Infrastructure | Primary Evaluation |",
        f"| :--- | :--- | :--- | :--- |",
    ])
    for p in payload.papers:
        short_title = p.title[:38] + "..." if len(p.title) > 38 else p.title
        author = (p.authors[0] + " et al.") if p.authors else "Unknown"
        year = f"({p.year})" if p.year else ""
        tldr = (p.tldr or p.abstract[:60] + "...").replace("|", "-")
        lines.append(f"| **{author} {year}**<br>*{short_title}* | {tldr} | {p.source} | Systems Benchmarks |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Formal Bibliography (IEEE Format)",
        f"",
    ])
    for idx, p in enumerate(payload.papers, 1):
        authors_str = ", ".join(p.authors[:3]) if p.authors else "Author Unlisted"
        if p.authors and len(p.authors) > 3:
            authors_str += " et al."
        year_str = f", {p.year}" if p.year else ""
        link = p.pdf_url or p.url or f"https://arxiv.org/abs/{p.id.replace('arxiv:', '')}"
        lines.append(f"[{idx}] {authors_str}, \"{p.title},\" *{p.source}*{year_str}. [Online]. Available: {link}")

    content = "\n".join(lines)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="Literature_Review.md"'}
    )


# ============================================================================
# Admin Observability Dashboard & Telemetry API
# ============================================================================

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard_page(request: Request):
    """
    Renders the protected Admin Observability Dashboard.
    Enforces Master Admin Key verification (via header, query param, or cookie).
    Returns 404 Not Found on unauthorized access for security obscurity against scrapers.
    """
    try:
        admin_key = verify_master_admin(request)
    except HTTPException:
        raise HTTPException(status_code=404, detail="Not Found")

    response = templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={"admin_key": admin_key}
    )
    response.set_cookie(
        key="axiom_admin_session",
        value=admin_key,
        max_age=86400 * 30,
        httponly=False,
        samesite="lax",
    )
    return response


@app.get("/api/admin/metrics")
async def get_admin_metrics_endpoint(request: Request):
    """
    Returns live presence, recent telemetry events, ring buffer usage, and moderation status.
    Protected by verify_master_admin (returns 404 Not Found if unauthorized).
    """
    verify_master_admin(request)
    return telemetry_manager.get_metrics_summary()



@app.post("/api/admin/block")
async def admin_block_endpoint(payload: ModerationBlockRequest, request: Request):
    """
    Ban or unban an IP address or Access Key.
    Protected by verify_master_admin (returns 404 Not Found if unauthorized).
    """
    verify_master_admin(request)
    ban = payload.action.lower() == "block"
    result = telemetry_manager.toggle_block(
        target_type=payload.target_type,
        target_value=payload.target_value,
        block=ban
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/admin/broadcast")
async def admin_broadcast_endpoint(payload: BroadcastRequest, request: Request):
    """
    Publish or clear the global system broadcast banner.
    Protected by verify_master_admin (returns 404 Not Found if unauthorized).
    """
    verify_master_admin(request)
    result = telemetry_manager.set_broadcast(message=payload.message, active=payload.active)
    return {"status": "ok", "broadcast": result}


@app.post("/api/admin/limits")
async def admin_limits_endpoint(payload: AdminLimitsRequest, request: Request):
    """
    Configure global or per-key hourly rate limits.
    Protected by verify_master_admin (returns 404 Not Found if unauthorized).
    """
    verify_master_admin(request)
    result = telemetry_manager.set_limits(
        global_hourly_limit=payload.global_hourly_limit,
        key_hourly_limit=payload.key_hourly_limit
    )
    return {"status": "ok", "limits": result}


@app.get("/api/broadcast")
async def get_public_broadcast_endpoint():
    """
    Public unauthenticated endpoint returning current system broadcast banner.
    Polled or fetched on load by client frontend (index.html).
    """
    return telemetry_manager.get_broadcast()


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Academic Research Assistant on http://%s:%d", HOST, PORT)
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
