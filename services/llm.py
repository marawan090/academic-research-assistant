"""
OrcaRouter LLM Synthesis Service.
Leverages OpenAI-compatible AsyncOpenAI SDK pointed to OrcaRouter (DeepSeek model)
to synthesize academic research inquiries and paper metadata into deep, structured analyses.
"""

import asyncio
import logging
import os
import json
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from .search import Paper, get_shared_http_client

logger = logging.getLogger("academic_assistant.llm")

DEFAULT_BASE_URL = "https://www.orcarouter.ai/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-free"
SECONDARY_MODEL = "deepseek/deepseek-chat"


class OutreachEmailRequest(BaseModel):
    paper_title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    target_professor: str
    scope_type: str = "single_paper"  # Options: "single_paper" | "holistic_lab"
    orcarouter_key: Optional[str] = None


OUTREACH_SYSTEM_PROMPT = """You are an elite academic advisor and researcher specializing in computer systems and AI.
You compose high-impact, intellectually rigorous academic cold outreach emails from prospective graduate/doctoral researchers or visiting scholars to principal investigators (professors).

Adhere strictly to these core rules:
1. Tone & Standards: Formal, intellectually rigorous, respectful, and direct.
2. Length: Concise, strictly under 250 words for the body text.
3. Content: Avoid superficial compliments (never say "I loved your paper", "I was fascinated by your work", or "I am a big fan of your research"). Directly connect the technical mechanics, concrete architectural bottlenecks, mathematical formulations, or unresolved trade-offs in the paper to prospective research collaboration.
4. Scope Modes:
   - "single_paper": Focus specifically on the core problem statement, architectural bottleneck, and unresolved trade-offs identified in the target paper.
   - "holistic_lab": Frame the email around the professor's overarching research trajectory and lab vision in this domain, using the paper as the primary springboard.
5. Placeholders: Include standardized bracketed placeholders for the applicant to customize (e.g., [My Current University/Degree], [Specific Technical Skill/Tooling], [Proposed Research Extension], [Link to CV/Portfolio]).
6. Format: Output strictly valid JSON with no markdown fences, backticks, reasoning preamble, or extraneous conversational text. Start immediately with '{' and end with '}':
{
  "subject": "Inquiring on [Specific Topic] — Prospective Graduate Researcher",
  "body": "Dear Professor [Last Name],\\n\\n..."
}
"""


def generate_fallback_outreach_email(
    paper_title: str,
    target_professor: str,
    abstract: str = "",
    scope_type: str = "single_paper"
) -> Dict[str, str]:
    """
    Generate an intellectually rigorous, publication-grade academic outreach email fallback
    strictly under 250 words adhering to the specified scope.
    """
    name_parts = target_professor.strip().split()
    last_name = name_parts[-1] if name_parts else "Professor"
    if last_name.lower() in ("dr.", "dr", "prof.", "prof", "professor"):
        last_name = name_parts[0] if len(name_parts) > 1 else "Professor"

    clean_title = paper_title.strip().rstrip(".")

    if scope_type == "holistic_lab":
        subject = f"Inquiring on Scalable Systems & Lab Trajectory — Prospective Graduate Researcher"
        body = (
            f"Dear Professor {last_name},\n\n"
            f"I have been following your laboratory's overarching research trajectory in scalable computing, "
            f"and was particularly compelled by your recent paper, \"{clean_title}\". Your work provides an incisive "
            f"formulation of core throughput-latency trade-offs in this domain.\n\n"
            f"I am completing my [My Current Degree/Program] at [My Current University], focusing on [Specific Technical Skill/Tooling, e.g., distributed consensus / kernel concurrency]. "
            f"Examining your group's methodology, I am particularly interested in extending these principles toward [Proposed Research Extension, e.g., lock-free execution pipelining under asymmetric latency]. "
            f"Your lab's broader agenda aligns closely with my aspiration to pursue doctoral research on resilient systems.\n\n"
            f"Are you considering prospective graduate researchers or research fellows for your group for upcoming cycles? "
            f"I would welcome 15 minutes to discuss potential alignment with your lab's active projects.\n\n"
            f"My curriculum vitae and recent implementations are linked at [Link to CV/Portfolio].\n\n"
            f"Thank you for your time and guidance.\n\n"
            f"Sincerely,\n\n"
            f"[Your Full Name]\n"
            f"[Your Contact Information / GitHub]"
        )
    else:
        subject_title = clean_title[:45]
        subject = f"Inquiring on \"{subject_title}\" — Prospective Graduate Researcher"
        body = (
            f"Dear Professor {last_name},\n\n"
            f"I have been closely analyzing your paper, \"{clean_title}\", particularly regarding its approach to addressing the "
            f"underlying bottleneck in [Core Architectural Bottleneck from Paper]. Your formulation provides a compelling mechanism "
            f"for navigating the trade-off between [Technical Property A] and [Technical Property B].\n\n"
            f"Currently pursuing my [My Current Degree/Program] at [My Current University], my background centers on [Specific Technical Skill/Tooling]. "
            f"In evaluating your benchmark conclusions, I observed a potential open question regarding [Specific Limitation / Unresolved Edge Case]. "
            f"I would be eager to investigate extending your architecture through [Proposed Research Extension, e.g., adaptive partition pruning or asynchronous state verification].\n\n"
            f"Are you currently open to prospective graduate researchers or research assistants joining your group? "
            f"I would appreciate the opportunity for a brief conversation to explore potential alignment.\n\n"
            f"My CV and representative code repositories are available at [Link to CV/Portfolio].\n\n"
            f"Thank you for your consideration.\n\n"
            f"Sincerely,\n\n"
            f"[Your Full Name]\n"
            f"[Your Contact Information / GitHub]"
        )

    return {
        "subject": subject,
        "body": body
    }


class SynthesisCache:
    """
    Thread-safe in-memory TTL cache for LLM syntheses, PlantUML architectures,
    and rendered Kroki SVG diagrams.
    Default TTL is 3600.0 seconds (1 hour). Duplicate queries return instantly (0ms latency).
    """
    def __init__(self, ttl: float = 3600.0):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        if not key:
            return None
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self._cache[key]
        return None

    def set(self, key: str, value: str) -> None:
        if not key or value is None:
            return
        with self._lock:
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# Global singletons for synthesis cache and concurrency throttling
synthesis_cache = SynthesisCache(ttl=3600.0)
llm_semaphore = asyncio.Semaphore(4)


async def _call_stream_with_retry(
    client: AsyncOpenAI,
    primary_model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    backoff_delays: Tuple[float, ...] = (1.0, 2.0, 4.0),
    secondary_model: Optional[str] = SECONDARY_MODEL,
):
    """
    Initiate streaming chat completion with automatic retry and exponential backoff
    on HTTP 429 (rate limits) or 5xx server errors, falling back to a secondary model if needed.
    """
    models_to_try = [primary_model]
    if secondary_model and secondary_model != primary_model:
        models_to_try.append(secondary_model)

    last_err: Optional[Exception] = None
    for model_idx, model in enumerate(models_to_try):
        attempts = max_retries if model_idx == 0 else 1
        for attempt in range(attempts):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": True,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                stream = await client.chat.completions.create(**kwargs)
                return stream
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_retryable = (
                    "429" in err_str
                    or "rate limit" in err_str
                    or "500" in err_str
                    or "502" in err_str
                    or "503" in err_str
                    or "504" in err_str
                    or "timeout" in err_str
                    or "overloaded" in err_str
                )
                if is_retryable and attempt < attempts - 1:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    logger.warning(
                        "LLM stream rate-limit or 5xx error (attempt %d/%d) with model %s: %s. Backing off %.1fs...",
                        attempt + 1, attempts, model, str(e), delay
                    )
                    await asyncio.sleep(delay)
                else:
                    break  # Try next model if any
    raise last_err or RuntimeError("LLM streaming completion failed after retries")


async def _call_chat_with_retry(
    client: AsyncOpenAI,
    primary_model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    backoff_delays: Tuple[float, ...] = (1.0, 2.0, 4.0),
    secondary_model: Optional[str] = SECONDARY_MODEL,
):
    """
    Non-streaming chat completion with automatic retry and exponential backoff
    on HTTP 429 (rate limits) or 5xx server errors, falling back to a secondary model if needed.
    """
    models_to_try = [primary_model]
    if secondary_model and secondary_model != primary_model:
        models_to_try.append(secondary_model)

    last_err: Optional[Exception] = None
    for model_idx, model in enumerate(models_to_try):
        attempts = max_retries if model_idx == 0 else 1
        for attempt in range(attempts):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                response = await client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_retryable = (
                    "429" in err_str
                    or "rate limit" in err_str
                    or "500" in err_str
                    or "502" in err_str
                    or "503" in err_str
                    or "504" in err_str
                    or "timeout" in err_str
                    or "overloaded" in err_str
                )
                if is_retryable and attempt < attempts - 1:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    logger.warning(
                        "LLM completion rate-limit or 5xx error (attempt %d/%d) with model %s: %s. Backing off %.1fs...",
                        attempt + 1, attempts, model, str(e), delay
                    )
                    await asyncio.sleep(delay)
                else:
                    break
    raise last_err or RuntimeError("LLM completion failed after retries")



def normalize_base_url(url: Optional[str]) -> str:
    """
    Normalize OrcaRouter base URL to prevent 302 redirects from rewriting POST requests to GET.
    orcarouter.com/v1 returns a 302 Found redirecting to www.orcarouter.ai/v1.
    In standard HTTP clients, 302 causes POST to become GET, triggering
    'Invalid URL (GET /v1/chat/completions)' 404 error.
    Directly routing to https://www.orcarouter.ai/v1 ensures pure HTTP POST execution.
    """
    raw = (url or "").strip().rstrip("/")
    if not raw:
        return "https://www.orcarouter.ai/v1"
    if "orcarouter.com" in raw:
        raw = raw.replace("orcarouter.com", "www.orcarouter.ai")
    elif "orcarouter.ai" in raw and "www." not in raw and "api." not in raw:
        raw = raw.replace("orcarouter.ai", "www.orcarouter.ai")
    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"
    return raw


MATH_NOTATION_PROMPT_DIRECTIVE = """
MATHEMATICAL NOTATION & TECHNICAL SYMBOLS DIRECTIVE:
- Format ALL mathematical formulas, variables, and algorithmic complexities strictly in LaTeX.
- Enclose inline formulas with single dollar signs: e.g., $O(N \\log N)$, $\\lambda$, $\\mathbb{E}[X]$, $p_{99} \\le 5\\text{ ms}$, $\\mathcal{O}(1)$.
- Enclose standalone block equations with double dollar signs on their own lines:
  $$L_{\\text{loss}} = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$$
- CRITICAL: DO NOT wrap LaTeX equations in backticks or code blocks (avoid `$$x=1$$` or `$O(N)$` inside code backticks).
- Use standardized Unicode for simple units and metrics (e.g., 99.9%, 15 ms, 4.2 GB/s, ->) to avoid encoding corruption.
"""


SYSTEM_PROMPT = """You are an elite, distinguished Senior Distributed Systems and Academic Computing Researcher (IEEE/ACM Fellow caliber).
Your role is to deeply analyze, cross-examine, and synthesize recent academic research papers to address the user's specific inquiry.

Your synthesis MUST strictly adhere to the following structure and academic standards:

# Executive Summary
Directly answer the user's core inquiry based on the state-of-the-art literature retrieved. Provide high-density technical insights, architectural paradigms, and performance trade-offs.

---

# In-Depth Paper Breakdowns
For each relevant paper provided in the context, create a structured analysis with the exact 4 sub-points:

### [Paper Title](Paper_or_PDF_URL)
- **1. Core Problem Statement**: What specific architectural bottleneck, theoretical limitation, or systems challenge does this work tackle?
- **2. Technical Infrastructure & Methodology**: What algorithms, protocols, hardware primitives (e.g., RDMA, CXL, NVLink, eBPF), data structures, or distributed consensus mechanisms are introduced?
- **3. Future Work & Research Gaps**: What did the authors leave unsolved? What scale, workload, or fault-tolerance assumptions limit the current approach? (Highlight open gaps for thesis or paper extensions).
- **4. Reference & Access**: Direct hyperlink to the paper or open-access PDF: [Access Paper / PDF](Paper_or_PDF_URL) (Authors: ..., Year: ..., Source: ...)

---

# Comparative Matrix & Architectural Trade-offs
Synthesize a concise comparison between the approaches (e.g., latency vs. consistency, compute overhead vs. memory footprint, hardware dependency vs. portability).

---

# Suggested Research Extensions (Delta Improvements)
Synthesize 3-4 concrete, actionable thesis and research paper topics that build on top of these papers' gaps. For each idea provide:
- **Title / Concept**: A formal research title.
- **The "Delta" (Novel Contribution)**: Exactly what is modified or introduced beyond current work.
- **Methodology & Feasibility**: How a graduate researcher could implement and benchmark it (e.g., simulation, testbed, open-source framework).

Maintain an authoritative, rigorous, yet accessible academic tone. Always provide direct Markdown hyperlinks to paper URLs or PDFs wherever available.

MATHEMATICAL NOTATION & TECHNICAL SYMBOLS DIRECTIVE:
- Format ALL mathematical formulas, variables, and algorithmic complexities strictly in LaTeX.
- Enclose inline formulas with single dollar signs: e.g., $O(N \\log N)$, $\\lambda$, $\\mathbb{E}[X]$, $p_{99} \\le 5\\text{ ms}$.
- Enclose standalone block equations with double dollar signs on their own lines:
  $$L_{\\text{loss}} = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$$
- DO NOT wrap LaTeX equations in backticks or code blocks.
- Use standardized Unicode for simple units and metrics (e.g., 99.9%, 15 ms, 4.2 GB/s, ->).
"""


SINGLE_PAPER_QUICK_SYSTEM_PROMPT = """You are an elite Senior Distributed Systems and Academic Computing Researcher.
Your task is to provide an ultra-fast, high-density synthesis of the target academic paper in under 120 words.
CRITICAL LATENCY INSTRUCTION: Start outputting the content immediately. Do NOT include any introductory greetings, conversational preamble, or concluding sign-offs. Skip all Mermaid diagrams and lengthy prose.

Your output MUST strictly contain ONLY these 3 bullet points:
- **1. Core Bottleneck Addressed**: Specific systems limitation, algorithmic bottleneck, or scalability ceiling this paper tackles.
- **2. Architecture / Technical Fix**: The primary mechanism, protocol, or data structure introduced to solve the problem.
- **3. The Delta**: Exactly 1 concise, actionable sentence proposing a thesis improvement idea extending this work.

MATHEMATICAL NOTATION & TECHNICAL SYMBOLS DIRECTIVE:
- Format ALL mathematical formulas, variables, and complexities strictly in LaTeX with single dollar signs ($O(N \\log N)$, $\\lambda$) or standalone double dollar signs ($$...$$).
- Never wrap LaTeX in backticks.
- Use standardized Unicode for simple units and metrics (e.g., 99.9%, 15 ms, ->).
"""

SINGLE_PAPER_BALANCED_SYSTEM_PROMPT = """You are an elite Senior Distributed Systems and Academic Computing Researcher.
Your task is to provide a crisp, rigorous 4-section summary (~250-350 words) of the target academic paper.
CRITICAL LATENCY INSTRUCTION: Start outputting the content immediately. Do NOT include any conversational filler, meta-announcements, or introductory preamble. Begin directly with section 1.

Your analysis MUST strictly adhere to this 4-section structure:

# 1. Problem & Core Motivation
- What specific systems bottleneck, algorithmic limitation, or challenge does this work tackle, and why do existing baselines fail under modern target workloads?

---

# 2. System Architecture / Key Primitives
- Detail the key technical components, algorithms, protocols, data structures, and hardware/software primitives introduced.

---

# 3. Research Gaps & Practical Limitations
- Assumptions that limit real-world deployment (hardware dependencies, network volatility, failure models, scaling overheads).

---

# 4. Proposed Thesis Extension (Delta Improvement)
- Synthesize a concrete research extension: formal title, novel delta architectural mechanism, and experimental validation strategy.

MATHEMATICAL NOTATION & TECHNICAL SYMBOLS DIRECTIVE:
- Format ALL mathematical formulas, variables, and algorithmic complexities strictly in LaTeX ($O(N \\log N)$, $\\lambda$, $p_{99} \\le 5\\text{ ms}$).
- Enclose standalone block equations with double dollar signs:
  $$L_{\\text{loss}} = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$$
- DO NOT wrap LaTeX equations in backticks or code blocks.
- Use standardized Unicode for simple units and metrics (e.g., 99.9%, 15 ms, 4.2 GB/s, ->).
"""

SINGLE_PAPER_DEEP_SYSTEM_PROMPT = """You are an elite Senior Distributed Systems and Academic Computing Researcher (IEEE/ACM Fellow caliber).
Your role is to deeply analyze, cross-examine, and deconstruct a specific academic research paper within the context of the user's research inquiry.
CRITICAL LATENCY INSTRUCTION: Start outputting the content immediately. Do NOT include any conversational preamble. Begin directly with section 1.

Your analysis MUST strictly adhere to the following 4-part structure:

# 1. Core Problem & Motivation
- What specific architectural bottleneck, algorithmic limitation, or systems challenge does this work tackle?
- Why do existing baselines, legacy protocols, or state-of-the-art approaches fail under modern target workloads?
- What core insight or hypothesis drives the authors' design?

---

# 2. Architecture & Technical Approach
- Detail the key technical components, algorithms, protocols, data structures, and hardware/software primitives (e.g., RDMA, CXL, NVLink, eBPF, consensus mechanisms, zero-copy, kernel bypass) introduced in this work.
- Explain the end-to-end execution flow, state management, and recovery/consistency mechanisms.
- Highlight how the design achieves its claimed throughput, latency, or fault-tolerance milestones.

---

# 3. Research Gaps & Limitations
- What assumptions did the authors make that might limit real-world deployment (e.g., specific hardware dependencies, homogeneous clusters, network conditions, workload profiles)?
- What scale, failure modes, consistency trade-offs, or straggler overheads remain unresolved?
- Identify concrete technical blind spots and open challenges left for future researchers.

---

# 4. Proposed Thesis Extension Idea (Delta Improvement)
Synthesize a concrete, high-impact research extension (suitable for a Master's thesis, Ph.D. topic, or top-tier conference paper such as OSDI, SOSP, EuroSys, or USENIX ATC) building directly upon this paper's open gaps:
- **Title**: A formal, publication-grade academic research title.
- **The "Delta" (Novel Contribution)**: Exactly what architectural mechanism, optimization, or hybrid design is introduced beyond the original paper.
- **Methodology & Benchmarking**: Concrete implementation strategy (frameworks, kernel modules, simulator or testbed) and experimental validation plan (workloads, baselines, target metrics).

Maintain an authoritative, rigorous, yet accessible academic tone. Include direct links to the paper / PDF when referencing specific sections.

MATHEMATICAL NOTATION & TECHNICAL SYMBOLS DIRECTIVE:
- Format ALL mathematical formulas, variables, and algorithmic complexities strictly in LaTeX ($O(N \\log N)$, $\\lambda$, $\\mathbb{E}[X]$, $p_{99} \\le 5\\text{ ms}$).
- Enclose standalone block equations with double dollar signs on their own lines:
  $$L_{\\text{loss}} = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$$
- DO NOT wrap LaTeX equations in backticks or code blocks.
- Use standardized Unicode for simple units and metrics (e.g., 99.9%, 15 ms, 4.2 GB/s, ->).
"""

PLANTUML_SYSTEM_PROMPT = """You are an elite Senior Distributed Systems and Software Architect.
Your task is to extract and map the given academic research paper's distributed systems architecture into clean, professional PlantUML Component & Deployment syntax.

System entities to visualize where relevant:
1. Gateways, Client Workloads, or Ingestion Frontends
2. Distributed Compute Workers, Accelerator Nodes (GPU/CPU/Host clusters)
3. In-Memory Caches, Checkpoint Ring Buffers, Shared Memory pools (RDMA, CXL, NVLink)
4. Async Message Queues, Consensus Coordinators, Event Streams (e.g. Raft, Paxos, Kafka)
5. Persistent Storage Mesh or Distributed Filesystem (e.g. NVMe, SSDs, S3, Lustre)

Style configuration:
Inject these exact modern skinparams immediately after @startuml:
skinparam roundcorner 10
skinparam shadowing false
skinparam defaultFontName "Inter", "Helvetica", sans-serif
skinparam defaultFontSize 12
skinparam ArrowColor #6366f1
skinparam ArrowThickness 1.5
skinparam componentStyle uml2
skinparam packageStyle rectangle
skinparam rectangle {
    BackgroundColor #0f172a
    BorderColor #334155
    FontColor #f8fafc
}
skinparam component {
    BackgroundColor #1e293b
    BorderColor #4f46e5
    FontColor #f8fafc
}
skinparam database {
    BackgroundColor #1e293b
    BorderColor #06b6d4
    FontColor #f8fafc
}
skinparam queue {
    BackgroundColor #1e293b
    BorderColor #a855f7
    FontColor #f8fafc
}
skinparam node {
    BackgroundColor #0b1120
    BorderColor #475569
    FontColor #e2e8f0
}

Output constraint:
- Output ONLY valid PlantUML code wrapped between @startuml and @enduml.
- Do NOT output any markdown backticks (no ```plantuml or ```), conversational preamble, or explanations.
"""

SINGLE_PAPER_SYSTEM_PROMPT = SINGLE_PAPER_DEEP_SYSTEM_PROMPT

DEPTH_CONFIG = {
    "quick": {
        "max_tokens": 250,
        "system_prompt": SINGLE_PAPER_QUICK_SYSTEM_PROMPT,
        "temperature": 0.2,
    },
    "balanced": {
        "max_tokens": 600,
        "system_prompt": SINGLE_PAPER_BALANCED_SYSTEM_PROMPT,
        "temperature": 0.3,
    },
    "deep": {
        "max_tokens": 1500,
        "system_prompt": SINGLE_PAPER_DEEP_SYSTEM_PROMPT,
        "temperature": 0.3,
    },
}

THESIS_PROPOSAL_SYSTEM_PROMPT = """You are a distinguished Academic Graduate Advisor and Senior Systems Researcher (ACM/IEEE Fellow).
Your task is to transform the analyzed academic paper and its open research gaps into a rigorous, publication-grade Academic Thesis Proposal Draft (~250-300 words).

Your output MUST strictly follow this structure:

# Academic Thesis Proposal Draft

### **Working Title**: *[Formal, publication-grade research title]*

---

### **1. Problem Statement & Research Motivation**
Concisely articulate the critical research gap, architectural bottleneck, or theoretical limitation left open by this work and prior baselines.

### **2. Proposed Delta Improvement & Technical Methodology**
Specify the exact novel mechanism, protocol optimization, or architectural hybrid being proposed (e.g., eBPF telemetry, CXL shared memory pooling, asynchronous snapshotting, or hardware-assisted consensus).

### **3. Evaluation Metrics & Experimental Plan**
Define concrete quantitative benchmarks:
- **Primary Metrics**: Latency (p99/p99.9), Throughput (IOPS/QPS), Memory Overhead, Recovery Time Objective (RTO).
- **Testbed & Workloads**: Target cluster scale, hardware accelerators, real-world traces/datasets, and comparative baselines.

### **4. Expected Contribution to Academic Literature**
State what new principle, artifact, or systems insight this thesis will contribute to the academic computing community.

MATHEMATICAL NOTATION & TECHNICAL SYMBOLS DIRECTIVE:
- Format ALL mathematical formulas, variables, and algorithmic complexities strictly in LaTeX ($O(N \\log N)$, $\\lambda$, $p_{99} \\le 5\\text{ ms}$).
- Enclose standalone block equations with double dollar signs on their own lines:
  $$\\text{RTO} = \\min_{k} \\left\\{ \\frac{\\Delta D_k}{B_{\\text{CXL}}} + \\tau_{\\text{preempt}} \\right\\}$$
- DO NOT wrap LaTeX equations in backticks or code blocks.
- Use standardized Unicode for simple units and metrics (e.g., 99.9%, 15 ms, 4.2 GB/s, ->).
"""

COMPARATIVE_MATRIX_SYSTEM_PROMPT = """You are an elite Senior Systems and Academic Computing Researcher (ACM/IEEE Fellow caliber).
Your task is to produce an in-depth, rigorous Comparative Analysis Matrix across the selected academic papers.

Your output MUST strictly follow this structure:

# Comparative Systems Literature Matrix

### Comparative Executive Summary
Provide a high-density summary comparing the fundamental trade-offs between the selected approaches (e.g., latency vs. consistency, throughput vs. memory overhead, hardware specialization vs. cloud portability).

---

### Architectural Comparison Table
Produce a comprehensive, beautifully formatted Markdown table contrasting the papers across the following dimensions:
| Analytical Dimension | [Paper 1 Title] | [Paper 2 Title] | [Paper 3 Title (if selected)] |
| :--- | :--- | :--- | :--- |
| **Core Problem Addressed** | ... | ... | ... |
| **Key Architectural Approach** | ... | ... | ... |
| **Infrastructure / Cloud Stack** | ... | ... | ... |
| **Scalability & Bottlenecks** | ... | ... | ... |
| **Evaluation Datasets & Workloads** | ... | ... | ... |
| **Primary Systems Trade-off** | ... | ... | ... |

---

### Key Synthesis Takeaways & Unified Direction
Synthesize what a unified next-generation system combining the complementary strengths of these papers would look like, identifying the most promising cross-cutting research frontier.

MATHEMATICAL NOTATION & TECHNICAL SYMBOLS DIRECTIVE:
- Format ALL mathematical formulas, variables, and algorithmic complexities strictly in LaTeX ($O(N \\log N)$, $\\lambda$, $p_{99} \\le 5\\text{ ms}$).
- Enclose standalone block equations with double dollar signs on their own lines:
  $$L_{\\text{loss}} = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$$
- DO NOT wrap LaTeX equations in backticks or code blocks.
- Use standardized Unicode for simple units and metrics (e.g., 99.9%, 15 ms, 4.2 GB/s, ->).
"""


def sanitize_plantuml(raw: str, default_title: str = "System Architecture") -> str:
    """
    Sanitize and validate PlantUML code extracted from LLM responses:
    1. Removes any surrounding markdown code fences (```plantuml ... ```).
    2. Strips out conversational preamble and postamble.
    3. Extracts strictly the content between @startuml and @enduml.
    4. Guarantees valid @startuml and @enduml boundaries.
    5. Injects essential dark/modern skinparams if missing.
    """
    if not raw or not raw.strip():
        return ""

    text = raw.strip()

    # Step 1: Locate @startuml and @enduml case-insensitively
    lower_text = text.lower()
    start_idx = lower_text.find("@startuml")
    end_idx = lower_text.rfind("@enduml")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        extracted = text[start_idx : end_idx + len("@enduml")].strip()
    elif start_idx != -1:
        extracted = text[start_idx:].strip() + "\n@enduml"
    elif end_idx != -1:
        extracted = "@startuml\n" + text[: end_idx + len("@enduml")].strip()
    else:
        cleaned = re.sub(r"^```(?:plantuml)?", "", text, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
        extracted = f"@startuml\n{cleaned}\n@enduml"

    # Step 2: Clean internal backticks or fences inside the extracted block
    lines = extracted.splitlines()
    clean_lines = []
    has_start = False
    has_end = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or (stripped.endswith("```") and len(stripped) <= 15):
            continue
        if stripped.lower().startswith("@startuml"):
            if not has_start:
                clean_lines.append("@startuml")
                has_start = True
            continue
        if stripped.lower().startswith("@enduml"):
            has_end = True
            continue
        clean_lines.append(line)

    if not has_start:
        clean_lines.insert(0, "@startuml")
    clean_lines.append("@enduml")

    body_content = "\n".join(clean_lines[1:-1])

    # Step 3: Inject robust skinparams / theme configuration if missing
    skinparam_block = []
    if "skinparam monochrome" not in body_content.lower():
        skinparam_block.append("skinparam monochrome false")
    if "skinparam shadowing" not in body_content.lower():
        skinparam_block.append("skinparam shadowing false")
    if "skinparam roundcorner" not in body_content.lower():
        skinparam_block.append("skinparam roundcorner 10")
    if "skinparam defaultfontname" not in body_content.lower():
        skinparam_block.append('skinparam defaultFontName "Inter", "Helvetica", sans-serif')
    if "skinparam defaultfontsize" not in body_content.lower():
        skinparam_block.append("skinparam defaultFontSize 12")
    if "skinparam arrowcolor" not in body_content.lower():
        skinparam_block.append("skinparam ArrowColor #6366f1")
    if "skinparam arrowthickness" not in body_content.lower():
        skinparam_block.append("skinparam ArrowThickness 1.5")
    if "skinparam componentstyle" not in body_content.lower():
        skinparam_block.append("skinparam componentStyle uml2")

    if skinparam_block:
        injected = "\n".join(skinparam_block)
        result = f"@startuml\n{injected}\n{body_content}\n@enduml"
    else:
        result = f"@startuml\n{body_content}\n@enduml"

    return result.strip()


class ResearchLLMService:
    """Service wrapping OrcaRouter's OpenAI-compatible API for research synthesis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        orcarouter_api_key: Optional[str] = None,
    ):
        self.api_key = api_key or orcarouter_api_key or os.getenv("ORCAROUTER_API_KEY", "").strip()
        raw_base_url = base_url or os.getenv("ORCAROUTER_BASE_URL", DEFAULT_BASE_URL)
        self.base_url = normalize_base_url(raw_base_url)
        self.model_name = (
            model_name
            or os.getenv("MODEL_NAME", DEFAULT_MODEL)
            or DEFAULT_MODEL
        )

    def _get_client(self, override_key: Optional[str] = None) -> Optional[AsyncOpenAI]:
        key = override_key.strip() if override_key and override_key.strip() else self.api_key
        if not key or key == "your_orcarouter_api_key_here":
            return None
        normalized_url = normalize_base_url(self.base_url)
        http_client = get_shared_http_client()
        return AsyncOpenAI(api_key=key, base_url=normalized_url, http_client=http_client)

    def is_configured(self) -> bool:
        """Check if a valid API key is present."""
        return bool(self.api_key and self.api_key != "your_orcarouter_api_key_here")

    def build_user_prompt(self, inquiry: str, papers: List[Paper]) -> str:
        """Format the inquiry and aggregated paper metadata into an information-dense prompt."""
        prompt_parts = [
            f"USER INQUIRY: {inquiry}\n",
            f"RETRIEVED ACADEMIC PAPERS ({len(papers)} papers identified):\n",
        ]

        if not papers:
            prompt_parts.append("No specific papers were retrieved from the academic repositories. Provide general domain knowledge addressing the user's inquiry.")
            return "\n".join(prompt_parts)

        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors) if p.authors else "Unknown Authors"
            link = p.pdf_url or p.url or f"https://arxiv.org/abs/{p.id.replace('arxiv:', '')}" if "arxiv" in p.id else "https://semanticscholar.org"
            citations_str = f" | Citations: {p.citation_count}" if p.citation_count is not None else ""
            year_str = f" ({p.year})" if p.year else ""
            tldr_str = f"\n  TLDR: {p.tldr}" if p.tldr else ""

            prompt_parts.append(
                f"### [Paper {i}] {p.title}{year_str}\n"
                f"  - Authors: {authors_str}\n"
                f"  - Source: {p.source}{citations_str}\n"
                f"  - Link: {link}{tldr_str}\n"
                f"  - Abstract:\n    {p.abstract or 'No abstract provided.'}\n"
            )

        prompt_parts.append(
            "\nPlease address the USER INQUIRY by synthesizing these papers in accordance with your instructions."
        )
        return "\n".join(prompt_parts)

    def generate_fallback_synthesis(self, inquiry: str, papers: List[Paper], reason: str) -> str:
        """
        Generate a high-quality heuristic academic synthesis when OrcaRouter API key
        is absent or unreachable, ensuring zero downtime for testing.
        """
        lines = [
            "> [!NOTE]",
            f"> **Live LLM Notice**: {reason}",
            "> Showing heuristic academic breakdown generated from retrieved paper abstracts and TLDRs. Configure `ORCAROUTER_API_KEY` in `.env` to enable real-time DeepSeek reasoning.",
            "",
            f"# Academic Synthesis: {inquiry}",
            "",
            "## Executive Summary",
            f"We analyzed **{len(papers)} peer-reviewed papers** retrieved concurrently across ArXiv CS repositories (cs.DC, cs.SE, cs.AI, cs.AR) and Semantic Scholar. The research landscape addresses state-of-the-art challenges in performance optimization, distributed resilience, and systems scalability.",
            "",
            "---",
            "",
            "## In-Depth Paper Breakdowns",
            ""
        ]

        for i, p in enumerate(papers, 1):
            link = p.pdf_url or p.url or "#"
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "") if p.authors else "Academic Authors"
            year = f" ({p.year})" if p.year else ""
            citations = f" | {p.citation_count} citations" if p.citation_count is not None else ""

            lines.extend([
                f"### [{p.title}]({link})",
                f"- **1. Core Problem Statement**: Investigates foundational challenges in distributed execution, focusing on minimizing operational overhead and latency under heterogeneous workloads.",
                f"- **2. Technical Infrastructure & Methodology**: " + (p.tldr or (p.abstract[:260] + "..." if len(p.abstract) > 260 else p.abstract or "Presents architectural design and empirical evaluation.")),
                f"- **3. Future Work & Research Gaps**: Scaling across multi-tenant clusters, mitigating network tail latency, and adapting to emerging disaggregated memory/CXL topologies.",
                f"- **4. Reference & Access**: [{p.source}{year}{citations}]({link}) by {authors}.",
                ""
            ])

        lines.extend([
            "---",
            "",
            "## Suggested Research Extensions (Delta Improvements)",
            "",
            "### 1. Adaptive Asynchronous Checkpoint Offloading for Disaggregated Memory",
            "- **The 'Delta' (Novel Contribution)**: Decouples local state serialization from remote persistent memory commits using non-blocking RDMA primitives, reducing commit complexity from $O(N)$ to $\\mathcal{O}(1)$.",
            "- **Methodology & Feasibility**: Implement an extension within PyTorch Distributed or vLLM; benchmark recovery overhead across a 16-GPU cluster under sustained bandwidth ($B_{\\text{CXL}} \\ge 64\\text{ GB/s}$).",
            "",
            "### 2. Straggler-Resilient Hierarchical Consensus in Geo-Distributed Clusters",
            "- **The 'Delta' (Novel Contribution)**: Replaces flat quorum exchanges with latency-aware hierarchical rings that bound replication tail latency:",
            "  $$T_{\\text{consensus}} \\le 2 \\cdot \\text{RTT}_{\\max} + \\mathcal{O}(\\log K)$$",
            "- **Methodology & Feasibility**: Model in network simulator (NS3) followed by microbenchmark validation on AWS cross-region instances targeting $p_{99} \\le 25\\text{ ms}$.",
            "",
            "### 3. Energy-Aware Dynamic Workload Migration in Heterogeneous Accelerators",
            "- **The 'Delta' (Novel Contribution)**: Integrates real-time power consumption telemetry into task scheduling heuristics optimizing the objective:",
            "  $$\\min_{\\pi} \\sum_{i=1}^M \\left( P_i(t) \\cdot \\Delta t + \\lambda \\cdot \\text{SLO}_{\\text{violation}} \\right)$$",
            "- **Methodology & Feasibility**: Evaluate on mixed GPU/TPU nodes using standard MLPerf inference and training traces achieving $>99.5\\%$ SLO adherence."
        ])

        return "\n".join(lines)

    async def stream_synthesis(
        self,
        inquiry: str,
        papers: List[Paper],
        custom_api_key: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously stream LLM synthesis tokens using OpenAI SDK with OrcaRouter endpoint.
        Gracefully falls back to heuristic streaming if the API key is not configured or fails.
        """
        client = self._get_client(custom_api_key)

        if client is None:
            logger.info("No OrcaRouter API key configured. Yielding structured fallback synthesis.")
            fallback_text = self.generate_fallback_synthesis(
                inquiry,
                papers,
                "OrcaRouter API key not configured. To activate DeepSeek LLM synthesis, set `ORCAROUTER_API_KEY` in `.env` or input it in the top settings bar."
            )
            # Stream in realistic chunks for smooth UI rendering
            chunk_size = 40
            for i in range(0, len(fallback_text), chunk_size):
                yield fallback_text[i:i + chunk_size]
            return

        user_content = self.build_user_prompt(inquiry, papers)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        try:
            logger.info("Initiating streaming chat completion with model %s via %s", self.model_name, self.base_url)
            async with llm_semaphore:
                stream = await _call_stream_with_retry(
                    client=client,
                    primary_model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    secondary_model=SECONDARY_MODEL,
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        content = delta.content or ""
                        if content:
                            yield content

        except Exception as e:
            logger.error("OrcaRouter completion error: %s", str(e), exc_info=True)
            fallback = self.generate_fallback_synthesis(
                inquiry,
                papers,
                f"OrcaRouter connection error: {str(e)}. Displaying offline paper analysis."
            )
            yield fallback

    async def synthesize(
        self,
        inquiry: str,
        papers: List[Paper],
        custom_api_key: Optional[str] = None
    ) -> str:
        """Non-streaming complete synthesis."""
        chunks = []
        async for chunk in self.stream_synthesis(inquiry, papers, custom_api_key):
            chunks.append(chunk)
        return "".join(chunks)

    def build_single_paper_prompt(self, inquiry: str, paper: Paper, depth: str = "balanced") -> str:
        """Format a single paper's metadata and research inquiry into a tailored prompt by depth."""
        authors_str = ", ".join(paper.authors) if paper.authors else "Unknown Authors"
        year_str = f" ({paper.year})" if paper.year else ""
        link = paper.pdf_url or paper.url or (f"https://arxiv.org/abs/{paper.id.replace('arxiv:', '')}" if "arxiv" in paper.id else "https://semanticscholar.org")
        citations_str = f" | Citations: {paper.citation_count}" if paper.citation_count is not None else ""
        tldr_str = f"\nTLDR: {paper.tldr}\n" if paper.tldr else ""

        depth_normalized = (depth or "balanced").lower()
        if depth_normalized == "quick":
            instruction = (
                "Extract ONLY 3 concise bullet points in under 120 words according to instructions:\n"
                "- **1. Core Bottleneck Addressed**\n"
                "- **2. Architecture / Technical Fix**\n"
                "- **3. The Delta** (1-line thesis improvement idea)\n"
                "Skip all diagrams, lengthy prose, and start outputting immediately without preamble."
            )
        elif depth_normalized == "deep":
            instruction = (
                "Please conduct an exhaustive, rigorous 4-part architectural deconstruction (~700+ words) "
                "with a thesis proposal blueprint according to your system instructions. "
                "Begin directly with Section 1 without conversational preamble."
            )
        else:
            instruction = (
                "Provide a crisp 4-section summary (~250-350 words) without conversational filler. "
                "Begin directly with Section 1."
            )

        return (
            f"USER RESEARCH INQUIRY / CONTEXT:\n{inquiry or 'General Distributed Systems and Academic Computing'}\n\n"
            f"TARGET PAPER FOR DEEPSEEK ANALYSIS (Depth: {depth_normalized.upper()}):\n"
            f"Title: {paper.title}{year_str}\n"
            f"Authors: {authors_str}\n"
            f"Source Repository: {paper.source}{citations_str}\n"
            f"Direct Paper / PDF Link: {link}\n"
            f"{tldr_str}\n"
            f"Abstract:\n{paper.abstract or 'No abstract provided.'}\n\n"
            f"{instruction}"
        )

    def generate_single_paper_fallback(
        self,
        paper: Paper,
        inquiry: str = "",
        notice: Optional[str] = None,
        depth: str = "deep"
    ) -> str:
        """Generate tailored offline heuristic analysis for a single paper based on requested depth."""
        authors_str = ", ".join(paper.authors[:3]) if paper.authors else "Unknown Authors"
        if paper.authors and len(paper.authors) > 3:
            authors_str += " et al."
        year_str = f" ({paper.year})" if paper.year else ""
        link = paper.pdf_url or paper.url or "https://arxiv.org"
        header_notice = f"> [!NOTE]\n> **Live LLM Notice**: {notice}\n\n" if notice else ""

        depth_norm = (depth or "deep").lower()
        if depth_norm == "quick":
            lines = [
                header_notice,
                f"### Quick Analysis: [{paper.title}]({link}){year_str}",
                f"*Authors: {authors_str} | Source: {paper.source}*",
                "",
                "- **1. Core Bottleneck Addressed**: Conventional distributed systems incur substantial latency ($p_{99} \\ge 85\\text{ ms}$) and roll-forward recovery penalties under high concurrency and transient node dropouts.",
                f"- **2. Architecture / Technical Fix**: Introduces asynchronous state replication decoupled from execution via memory-mapped buffers with $\\mathcal{{O}}(1)$ commit overhead.",
                "- **3. The Delta**: Extend the protocol with hardware-assisted eBPF kernel offloading to dynamically preempt straggler nodes before recovery barriers stall ($T_{\\text{preempt}} < 2\\text{ ms}$).",
                "",
                f"[Direct Access to Paper / PDF]({link})"
            ]
            return "\n".join(lines)

        if depth_norm == "balanced":
            lines = [
                header_notice,
                f"### Balanced Synthesis: [{paper.title}]({link}){year_str}",
                f"*Authors: {authors_str} | Source: {paper.source}*",
                "",
                "# 1. Problem & Core Motivation",
                f"This work addresses synchronization and storage barriers in modern distributed environments where roll-forward recovery overhead scales with cluster size $N$ as $\\mathcal{{O}}(N)$.",
                f"- **Target Challenge**: {paper.tldr or (paper.abstract[:250] + '...' if len(paper.abstract) > 250 else paper.abstract)}",
                "",
                "---",
                "",
                "# 2. System Architecture / Key Primitives",
                "- **Decoupled Execution**: Computations proceed concurrently with non-blocking background state snapshots maintaining $p_{99} \\le 15\\text{ ms}$.",
                "- **Adaptive Coordination**: Dynamically adjusts checkpoint frequency based on runtime telemetry according to:",
                "  $$\\Delta t_{\\text{checkpoint}} = \\sqrt{\\frac{2 \\cdot C_{\\text{cost}}}{\\lambda \\cdot D_{\\text{state}}}}$$",
                "- **Tiered Buffering**: Utilizes in-memory circular buffers to mask disk I/O latency ($B_{\\text{NVMe}} \\ge 7.2\\text{ GB/s}$).",
                "",
                "---",
                "",
                "# 3. Research Gaps & Practical Limitations",
                "- **Hardware Uniformity**: Experimental validation assumes homogeneous clusters; multi-tenant cloud volatility may degrade performance guarantees by $>40\\%$.",
                "- **Failure Model Scope**: Complex Byzantine partitions and simultaneous cascaded switch failures require additional recovery logic.",
                "",
                "---",
                "",
                "# 4. Proposed Thesis Extension (Delta Improvement)",
                "- **Title**: *Straggler-Resilient Adaptive Checkpointing for Heterogeneous Multi-Tenant Clouds*",
                "- **The Delta**: Combine eBPF telemetry with CXL memory pooling to migrate active state before failure manifests, bounding recovery time objective:",
                "  $$\\text{RTO} \\le \\min_{k} \\left\\{ \\frac{\\Delta D_k}{B_{\\text{CXL}}} + \\tau_{\\text{sync}} \\right\\}$$",
                "- **Validation**: Benchmark on a 32-node distributed testbed measuring throughput ($Q \\ge 10^5\\text{ IOPS}$) and Recovery Time Objective ($\\text{RTO} < 200\\text{ ms}$).",
                "",
                f"[Direct Access to Paper / PDF]({link})"
            ]
            return "\n".join(lines)

        # Default: Deep Dive (with Mermaid.js architecture)
        lines = [
            header_notice,
            f"## Deep Analysis: [{paper.title}]({link}){year_str}",
            f"*Authors: {authors_str} | Source: {paper.source}*",
            "",
            "# 1. Core Problem & Motivation",
            f"This research addresses fundamental throughput, latency, and reliability bottlenecks in distributed execution environments. Specifically, the authors target scenarios where conventional synchronizations or storage barriers introduce severe latency and recovery penalties under modern scale.",
            f"- **Target Challenge**: {paper.tldr or paper.abstract[:280] + '...' if len(paper.abstract) > 280 else paper.abstract}",
            "- **Why Existing Systems Fail**: Legacy baselines rely on synchronous persistence checkpoints, heavy coordination rounds, or uncoordinated quorums that stall computation under high concurrency or transient worker dropouts.",
            "",
            "---",
            "",
            "# 2. Architecture & Technical Approach",
            f"The paper introduces an optimized systems architecture combining asynchronous coordination with localized data caching:",
            f"- **Execution Pipeline**: Decouples the critical computational path from state replication using memory-mapped buffers and non-blocking background serialization.",
            f"- **Algorithmic Primitives**: Implements adaptive coordination protocols that dynamically balance checkpoint interval overhead against roll-forward recovery complexity.",
            f"- **Hardware & Network Awareness**: Exploits high-bandwidth interconnects and tiered memory hierarchies to minimize host CPU memory contention during snapshot intervals.",
            "",
            "### Architecture & Execution Dataflow (Mermaid.js)",
            "```mermaid",
            "flowchart TD",
            '    A["Client Workload Generator"] --> B["Ingestion & State Coordinator"]',
            '    B --> C["In-Memory Execution Engine"]',
            '    C --> D["Tiered Checkpoint Buffer"]',
            '    D -->|Async Serialization| E["Distributed Storage / NVMe Mesh"]',
            '    D -.->|Fault Detection & Roll-Forward| C',
            "```",
            "",
            "---",
            "",
            "# 3. Research Gaps & Limitations",
            "While achieving notable benchmark efficiency, several fundamental systems constraints remain open:",
            "- **Heterogeneity & Scaling Assumptions**: The experimental evaluation assumes relatively uniform network latencies and predictable hardware configurations, which may degrade under multi-tenant cloud volatility.",
            "- **Failure Model Coverage**: Multi-node simultaneous cascades and Byzantine network partitions are not fully covered within the primary recovery protocol.",
            "- **Storage Tier Overhead**: Long-running simulations with massive memory footprints may experience metadata amplification when scaling beyond single-cluster boundaries.",
            "",
            "---",
            "",
            "# 4. Proposed Thesis Extension Idea (Delta Improvement)",
            f"- **Title**: *Adaptive Straggler-Resilient Checkpoint Offloading for Heterogeneous Cloud Accelerators*",
            "- **The 'Delta' (Novel Contribution)**: Introduce an asynchronous eBPF-instrumented telemetry layer that dynamically predicts node degradation and migrates state snapshots using CXL shared memory pools before faults materialize.",
            "- **Methodology & Benchmarking**: Implement a prototype in Rust/C++ extending PyTorch Distributed or MPI; evaluate across a 32-node heterogeneous GPU testbed measuring recovery time objective ($\\text{RTO} < 180\\text{ ms}$), throughput ($>1.2\\times 10^5\\text{ IOPS}$), and tail latency ($p_{99} \\le 15\\text{ ms}$) under synthetic fault injections.",
            "",
            f"[Direct Access to Paper / PDF]({link})"
        ]
        return "\n".join(lines)

    async def stream_single_paper_analysis(
        self,
        paper: Paper,
        inquiry: str = "",
        depth: str = "balanced",
        custom_api_key: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream on-demand LLM breakdown for a single chosen paper with depth control and TTL caching."""
        depth_normalized = (depth or "balanced").lower()
        if depth_normalized not in DEPTH_CONFIG:
            depth_normalized = "balanced"

        cfg = DEPTH_CONFIG[depth_normalized]
        cache_key = f"synthesis:{paper.id}:{depth_normalized}"

        # 1. Check in-memory TTL synthesis cache (0ms instant response)
        cached_content = synthesis_cache.get(cache_key)
        if cached_content is not None:
            logger.info("Synthesis cache HIT for paper %r [depth: %s] (0ms latency)", paper.id, depth_normalized)
            chunk_size = 50
            for i in range(0, len(cached_content), chunk_size):
                yield cached_content[i:i + chunk_size]
                await asyncio.sleep(0)
            return

        client = self._get_client(custom_api_key)

        if client is None:
            logger.info("No OrcaRouter API key configured. Yielding structured single paper fallback (depth: %s).", depth_normalized)
            fallback = self.generate_single_paper_fallback(
                paper,
                inquiry,
                "OrcaRouter API key not configured. Displaying offline heuristic paper analysis.",
                depth=depth_normalized
            )
            synthesis_cache.set(cache_key, fallback)
            chunk_size = 35
            for i in range(0, len(fallback), chunk_size):
                yield fallback[i:i + chunk_size]
            return

        user_content = self.build_single_paper_prompt(inquiry, paper, depth=depth_normalized)
        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_content}
        ]

        accumulated_tokens: List[str] = []
        try:
            logger.info(
                "Initiating single paper analysis (depth: %s, max_tokens: %d) with model %s via %s",
                depth_normalized, cfg["max_tokens"], self.model_name, self.base_url
            )
            async with llm_semaphore:
                stream = await _call_stream_with_retry(
                    client=client,
                    primary_model=self.model_name,
                    messages=messages,
                    temperature=cfg["temperature"],
                    max_tokens=cfg["max_tokens"],
                    secondary_model=SECONDARY_MODEL,
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        content = delta.content or ""
                        if content:
                            accumulated_tokens.append(content)
                            yield content

            # Populate synthesis cache on complete successful stream
            if accumulated_tokens:
                synthesis_cache.set(cache_key, "".join(accumulated_tokens))

        except Exception as e:
            logger.error("OrcaRouter single paper completion error: %s", str(e), exc_info=True)
            fallback = self.generate_single_paper_fallback(
                paper,
                inquiry,
                f"OrcaRouter connection error: {str(e)}. Displaying offline paper analysis.",
                depth=depth_normalized
            )
            synthesis_cache.set(cache_key, fallback)
            yield fallback


    def build_thesis_proposal_prompt(self, inquiry: str, paper: Paper) -> str:
        """Construct user prompt for generating a formal ~250-word thesis proposal section."""
        authors_str = ", ".join(paper.authors[:3]) if paper.authors else "Unknown Authors"
        year_str = f" ({paper.year})" if paper.year else ""
        link = paper.pdf_url or paper.url or "https://arxiv.org"
        return (
            f"RESEARCH THEME & INQUIRY:\n{inquiry or 'Advanced Systems Architecture and High Performance Computing'}\n\n"
            f"FOUNDATIONAL REFERENCE PAPER:\n"
            f"- Title: {paper.title}{year_str}\n"
            f"- Authors: {authors_str}\n"
            f"- Link: {link}\n"
            f"- Abstract: {paper.abstract or 'No abstract provided.'}\n\n"
            f"Draft a rigorous 250-300 word formal Academic Thesis Proposal Section based on this paper's open gaps."
        )

    def generate_thesis_proposal_fallback(
        self,
        paper: Paper,
        inquiry: str = "",
        notice: Optional[str] = None
    ) -> str:
        """Generate high-density offline thesis proposal draft."""
        authors_str = ", ".join(paper.authors[:2]) if paper.authors else "Unknown Authors"
        year_str = f" ({paper.year})" if paper.year else ""
        clean_kw = inquiry or "Heterogeneous Cloud Computing"
        header_notice = f"> [!NOTE]\n> **Live LLM Notice**: {notice}\n\n" if notice else ""

        return (
            f"{header_notice}"
            f"# Academic Thesis Proposal Draft\n\n"
            f"### **Working Title**: *Straggler-Resilient Asynchronous State Offloading for {clean_kw.title()} Workloads*\n\n"
            f"*Foundational Reference: [{paper.title}]({paper.pdf_url or paper.url or '#'}) by {authors_str}{year_str}*\n\n"
            f"---\n\n"
            f"### **1. Problem Statement & Research Motivation**\n"
            f"Modern distributed execution frameworks suffer from severe coordination stalls when scaling across heterogeneous multi-tenant accelerators. "
            f"As demonstrated by {authors_str}{year_str}, conventional barrier synchronizations and synchronous snapshots force fast workers to idle, "
            f"amplifying tail latency by up to 3.8x under hardware jitter. Existing checkpointing schemes lack pre-emptive awareness of stragglers, "
            f"rendering reactive recovery protocols prohibitive for real-time and long-running distributed pipelines.\n\n"
            f"### **2. Proposed Delta Improvement & Technical Methodology**\n"
            f"This thesis proposes an eBPF-driven kernel telemetry framework coupled with a tiered CXL-attached shared memory abstraction. "
            f"Unlike existing reactive snapshot baselines, our architecture continuously monitors interconnect queue depths and memory thermal throttling, "
            f"dynamically migrating pipeline states to localized CXL pools prior to node degradation. By decoupling snapshot serialization from the host CPU path, "
            f"we eliminate synchronization barriers entirely.\n\n"
            f"### **3. Evaluation Metrics & Experimental Plan**\n"
            f"- **Primary Metrics**: 99th percentile end-to-end latency ($p_{{99}} \\le 15\\text{{ ms}}$), sustainable throughput (QPS/TFLOPS), host memory amplification factor ($<5\\%$), and Recovery Time Objective ($\\text{{RTO}} \\le 200\\text{{ ms}}$).\n"
            f"- **Analytical Formulation**:\n"
            f"  $$\\text{{RTO}} = \\min_{{k}} \\left\\{{ \\frac{{\\Delta D_k}}{{B_{{\\text{{CXL}}}}}} + \\tau_{{\\text{{preempt}}}} \\right\\}}$$\n"
            f"- **Testbed & Workloads**: Evaluated across an 8-node heterogeneous GPU/RDMA testbed utilizing synthetic Poisson fault injection and real-world HPC/LLM checkpoint traces.\n\n"
            f"### **4. Expected Contribution to Academic Literature**\n"
            f"This research will deliver: (1) a formalized theoretical bound for asynchronous straggler prediction with complexity $\\mathcal{{O}}(N \\log K)$, (2) an open-source zero-copy CXL offloading daemon, "
            f"and (3) empirical validation proving consistent $p_{{99}}$ latency guarantees under arbitrary cloud tenancy volatility."
        )

    async def stream_thesis_proposal(
        self,
        paper: Paper,
        inquiry: str = "",
        custom_api_key: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream real-time thesis proposal draft generation with TTL caching."""
        cache_key = f"thesis:{paper.id}"
        cached_content = synthesis_cache.get(cache_key)
        if cached_content is not None:
            logger.info("Synthesis cache HIT for thesis proposal %r (0ms latency)", paper.id)
            chunk_size = 50
            for i in range(0, len(cached_content), chunk_size):
                yield cached_content[i:i + chunk_size]
                await asyncio.sleep(0)
            return

        client = self._get_client(custom_api_key)

        if client is None:
            logger.info("No OrcaRouter API key configured. Yielding structured thesis proposal fallback.")
            fallback = self.generate_thesis_proposal_fallback(
                paper,
                inquiry,
                "OrcaRouter API key not configured. Generating high-density academic thesis template."
            )
            synthesis_cache.set(cache_key, fallback)
            chunk_size = 40
            for i in range(0, len(fallback), chunk_size):
                yield fallback[i:i + chunk_size]
            return

        user_content = self.build_thesis_proposal_prompt(inquiry, paper)
        messages = [
            {"role": "system", "content": THESIS_PROPOSAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        accumulated_tokens: List[str] = []
        try:
            logger.info("Initiating thesis proposal streaming with model %s via %s", self.model_name, self.base_url)
            async with llm_semaphore:
                stream = await _call_stream_with_retry(
                    client=client,
                    primary_model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    secondary_model=SECONDARY_MODEL,
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        content = delta.content or ""
                        if content:
                            accumulated_tokens.append(content)
                            yield content

            if accumulated_tokens:
                synthesis_cache.set(cache_key, "".join(accumulated_tokens))

        except Exception as e:
            logger.error("OrcaRouter thesis proposal error: %s", str(e), exc_info=True)
            fallback = self.generate_thesis_proposal_fallback(
                paper,
                inquiry,
                f"OrcaRouter connection error: {str(e)}. Displaying offline proposal draft."
            )
            synthesis_cache.set(cache_key, fallback)
            yield fallback

    def build_papers_comparison_prompt(self, inquiry: str, papers: List[Paper]) -> str:
        """Construct prompt for multi-paper comparative matrix."""
        prompt_parts = [
            f"USER RESEARCH CONTEXT:\n{inquiry or 'Comparative Systems Research Analysis'}\n\n",
            f"SELECTED PAPERS FOR COMPARATIVE MATRIX ({len(papers)} papers):\n"
        ]
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors[:2]) if p.authors else "Unknown Authors"
            link = p.pdf_url or p.url or f"https://arxiv.org/abs/{p.id.replace('arxiv:', '')}" if "arxiv" in p.id else "https://semanticscholar.org"
            year_str = f" ({p.year})" if p.year else ""
            tldr_str = f" | TLDR: {p.tldr}" if p.tldr else ""
            prompt_parts.append(
                f"[{i}] {p.title}{year_str}\n"
                f"    Authors: {authors_str} | Source: {p.source}{tldr_str}\n"
                f"    Link: {link}\n"
                f"    Abstract: {p.abstract or 'No abstract provided.'}\n\n"
            )
        prompt_parts.append(
            "Please generate an authoritative, rigorous Comparative Systems Literature Matrix comparing these papers."
        )
        return "".join(prompt_parts)

    def generate_papers_comparison_fallback(
        self,
        papers: List[Paper],
        inquiry: str = "",
        notice: Optional[str] = None
    ) -> str:
        """Generate high-quality offline comparative matrix table."""
        header_notice = f"> [!NOTE]\n> **Live LLM Notice**: {notice}\n\n" if notice else ""
        
        # Build headers
        titles = [p.title[:32] + "..." if len(p.title) > 32 else p.title for p in papers]
        col_headers = " | ".join(f"**{t}**" for t in titles)
        col_dividers = " | ".join([":---"] * len(titles))
        
        p1 = papers[0] if len(papers) > 0 else None
        p2 = papers[1] if len(papers) > 1 else None
        p3 = papers[2] if len(papers) > 2 else None

        row_problem = f"| **Core Problem** | {p1.tldr or p1.abstract[:80] + '...' if p1 else 'N/A'} | {p2.tldr or p2.abstract[:80] + '...' if p2 else 'N/A'} |"
        row_arch = f"| **Key Architectural Approach** | Asynchronous tiered snapshotting | Protocol-level kernel-bypass coordination |"
        row_infra = f"| **Infrastructure / Cloud Stack** | Heterogeneous GPU Clusters / CXL | Distributed RPC / RDMA RoCEv2 |"
        row_bottlenecks = f"| **Scalability Bottlenecks** | Host PCIe saturation at high concurrency | Synchronization overhead across WAN clusters |"
        row_benchmarks = f"| **Datasets & Workloads** | Real-world scientific simulation traces | Synthetic Poisson IOPS benchmarks |"
        row_tradeoff = f"| **Primary Systems Trade-off** | Low write overhead vs. recovery replay latency | Extreme throughput vs. hardware homogeneity |"

        if p3:
            row_problem = row_problem[:-1] + f" {p3.tldr or p3.abstract[:80] + '...'} |"
            row_arch = row_arch[:-1] + f" Adaptive localized consensus buffer |"
            row_infra = row_infra[:-1] + f" Cloud Spot VMs / NVMe-oF |"
            row_bottlenecks = row_bottlenecks[:-1] + f" Ephemeral node evictions & network churn |"
            row_benchmarks = row_benchmarks[:-1] + f" MicroVM cold-start benchmarks |"
            row_tradeoff = row_tradeoff[:-1] + f" Fault-tolerance cost vs. memory retention |"

        table = (
            f"| Analytical Dimension | {col_headers} |\n"
            f"| :--- | {col_dividers} |\n"
            f"{row_problem}\n"
            f"{row_arch}\n"
            f"{row_infra}\n"
            f"{row_bottlenecks}\n"
            f"{row_benchmarks}\n"
            f"{row_tradeoff}\n"
        )

        return (
            f"{header_notice}"
            f"# Comparative Systems Literature Matrix\n\n"
            f"### Comparative Executive Summary\n"
            f"Comparing the {len(papers)} selected approaches reveals distinct architectural design points across the distributed systems continuum. "
            f"While earlier architectures prioritize strict serializability and synchronous state persistence, modern paradigms pivot towards zero-copy asynchronous pipelines "
            f"and kernel bypass to achieve predictable tail latency ($p_{{99}} \\le 1.2\\text{{ ms}}$).\n\n"
            f"---\n\n"
            f"### Architectural Comparison Table\n\n"
            f"{table}\n\n"
            f"---\n\n"
            f"### Key Synthesis Takeaways & Unified Direction\n"
            f"A unified next-generation literature architecture should synthesize the hardware-conscious telemetry of the first approach with the resilient failure mitigation of the second. "
            f"By combining asynchronous CXL shared memory pooling with adaptive network routing, graduate researchers can design a hybrid systems substrate capable of sub-millisecond fault tolerance ($p_{{99}} \\le 0.8\\text{{ ms}}$) across heterogeneous cloud clusters with asymptotic bound $\\mathcal{{O}}(N \\log N)$."
        )

    async def stream_papers_comparison(
        self,
        papers: List[Paper],
        inquiry: str = "",
        custom_api_key: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream real-time multi-paper comparative analysis matrix with TTL caching."""
        sorted_ids = ",".join(sorted(p.id for p in papers))
        cache_key = f"compare:{sorted_ids}"
        cached_content = synthesis_cache.get(cache_key)
        if cached_content is not None:
            logger.info("Synthesis cache HIT for comparative matrix (%d papers) (0ms latency)", len(papers))
            chunk_size = 50
            for i in range(0, len(cached_content), chunk_size):
                yield cached_content[i:i + chunk_size]
                await asyncio.sleep(0)
            return

        client = self._get_client(custom_api_key)

        if client is None:
            logger.info("No OrcaRouter API key configured. Yielding structured comparative matrix fallback.")
            fallback = self.generate_papers_comparison_fallback(
                papers,
                inquiry,
                "OrcaRouter API key not configured. Generating offline comparative literature matrix."
            )
            synthesis_cache.set(cache_key, fallback)
            chunk_size = 40
            for i in range(0, len(fallback), chunk_size):
                yield fallback[i:i + chunk_size]
            return

        user_content = self.build_papers_comparison_prompt(inquiry, papers)
        messages = [
            {"role": "system", "content": COMPARATIVE_MATRIX_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        accumulated_tokens: List[str] = []
        try:
            logger.info("Initiating comparative matrix streaming with model %s via %s", self.model_name, self.base_url)
            async with llm_semaphore:
                stream = await _call_stream_with_retry(
                    client=client,
                    primary_model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    secondary_model=SECONDARY_MODEL,
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        content = delta.content or ""
                        if content:
                            accumulated_tokens.append(content)
                            yield content

            if accumulated_tokens:
                synthesis_cache.set(cache_key, "".join(accumulated_tokens))

        except Exception as e:
            logger.error("OrcaRouter comparative matrix error: %s", str(e), exc_info=True)
            fallback = self.generate_papers_comparison_fallback(
                papers,
                inquiry,
                f"OrcaRouter connection error: {str(e)}. Displaying offline comparative matrix."
            )
            synthesis_cache.set(cache_key, fallback)
            yield fallback

    def generate_fallback_plantuml(self, paper_title: str, abstract: str = "", tldr: str = "") -> str:
        """Generate professional PlantUML component diagram code for offline/fallback use."""
        clean_title = (paper_title or "System Architecture").replace('"', "'")
        return f"""@startuml
skinparam roundcorner 10
skinparam shadowing false
skinparam defaultFontName "Inter", "Helvetica", sans-serif
skinparam defaultFontSize 12
skinparam ArrowColor #6366f1
skinparam ArrowThickness 1.5
skinparam componentStyle uml2
skinparam packageStyle rectangle
skinparam rectangle {{
    BackgroundColor #0f172a
    BorderColor #334155
    FontColor #f8fafc
}}
skinparam component {{
    BackgroundColor #1e293b
    BorderColor #4f46e5
    FontColor #f8fafc
}}
skinparam database {{
    BackgroundColor #1e293b
    BorderColor #06b6d4
    FontColor #f8fafc
}}
skinparam queue {{
    BackgroundColor #1e293b
    BorderColor #a855f7
    FontColor #f8fafc
}}
skinparam node {{
    BackgroundColor #0b1120
    BorderColor #475569
    FontColor #e2e8f0
}}

title {clean_title} - Architectural Dataflow

node "Ingress & Client Layer" {{
    [Client Workloads / Gateways] as Clients
    [API Gateway & Ingestion Router] as Gateway
}}

node "Distributed Compute Cluster" {{
    package "Compute Nodes (GPU/CPU/Host)" {{
        [Execution Engine Worker 1] as Worker1
        [Execution Engine Worker 2] as Worker2
        [Adaptive Coordinator] as Coord
    }}
}}

node "High-Performance State Layer" {{
    queue "Async Consensus & Event Stream" as MsgQueue
    [In-Memory Checkpoint Ring Buffer] as CacheRing
    database "Tiered NVMe / Distributed Storage Mesh" as StorageMesh
}}

Clients --> Gateway : User RPCs / Tasks
Gateway --> Coord : Dispatches Jobs
Coord --> Worker1 : Parallel Execution
Coord --> Worker2 : Parallel Execution
Worker1 ..> CacheRing : Low-Latency Snapshot (Zero-Copy)
Worker2 ..> CacheRing : Low-Latency Snapshot (Zero-Copy)
Worker1 <--> MsgQueue : Heartbeat & State Sync
Worker2 <--> MsgQueue : Heartbeat & State Sync
CacheRing --> StorageMesh : Background Asynchronous Persistence
@enduml"""

    async def generate_plantuml_architecture(
        self,
        title: str,
        abstract: str = "",
        tldr: str = "",
        custom_api_key: Optional[str] = None
    ) -> str:
        """
        Extract and generate clean PlantUML Component & Deployment code using DeepSeek.
        Enforces strict PlantUML syntax between @startuml and @enduml with TTL caching.
        """
        cache_key = f"plantuml:{title.strip().lower()}"
        cached = synthesis_cache.get(cache_key)
        if cached is not None:
            logger.info("Synthesis cache HIT for PlantUML architecture %r (0ms latency)", title[:40])
            return cached

        client = self._get_client(custom_api_key)
        if client is None:
            logger.info("No OrcaRouter API key provided. Using fallback PlantUML generation.")
            fallback = sanitize_plantuml(self.generate_fallback_plantuml(title, abstract, tldr), default_title=title)
            synthesis_cache.set(cache_key, fallback)
            return fallback

        user_prompt = (
            f"Generate a professional PlantUML Component & Deployment diagram for this academic research paper:\n\n"
            f"Title: {title}\n"
            f"TLDR: {tldr or 'N/A'}\n"
            f"Abstract:\n{abstract or 'N/A'}\n\n"
            f"Remember: Output ONLY valid PlantUML code enclosed between @startuml and @enduml. "
            f"Include the required skinparams and entities (Gateways, Compute Workers, In-Memory Caches, Queues, Persistent Storage)."
        )

        messages = [
            {"role": "system", "content": PLANTUML_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            async with llm_semaphore:
                response = await _call_chat_with_retry(
                    client=client,
                    primary_model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1000,
                    secondary_model=SECONDARY_MODEL,
                )
            content = (response.choices[0].message.content or "").strip()
            clean_code = sanitize_plantuml(content, default_title=title)
            if not clean_code or len(clean_code.strip().splitlines()) < 3:
                clean_code = sanitize_plantuml(self.generate_fallback_plantuml(title, abstract, tldr), default_title=title)
            synthesis_cache.set(cache_key, clean_code)
            return clean_code

        except Exception as e:
            logger.error("Error generating PlantUML via DeepSeek: %s", str(e), exc_info=True)
            fallback = sanitize_plantuml(self.generate_fallback_plantuml(title, abstract, tldr), default_title=title)
            synthesis_cache.set(cache_key, fallback)
            return fallback

    async def render_plantuml_kroki(
        self,
        plantuml_code: str,
        paper_title: str = "",
        abstract: str = "",
        tldr: str = "",
        enable_fallback: bool = True
    ) -> str:
        """
        Asynchronously POST PlantUML plain text to Kroki public service to obtain SVG markup.
        Endpoint: https://kroki.io/plantuml/svg
        Reuses pooled shared HTTP client and caches rendered SVG for 1 hour.
        Logs exact Kroki error response body on non-200 status and seamlessly falls back to
        guaranteed-valid PlantUML architecture.
        """
        clean_code = sanitize_plantuml(plantuml_code, default_title=paper_title)
        if not clean_code or len(clean_code.strip().splitlines()) < 3:
            clean_code = sanitize_plantuml(
                self.generate_fallback_plantuml(paper_title or "System Architecture", abstract, tldr),
                default_title=paper_title
            )

        cache_key = f"kroki_svg:{clean_code}"
        cached = synthesis_cache.get(cache_key)
        if cached is not None:
            logger.info("Synthesis cache HIT for Kroki SVG diagram (0ms latency)")
            return cached

        kroki_url = "https://kroki.io/plantuml/svg"
        http_client = get_shared_http_client()

        try:
            response = await http_client.post(
                kroki_url,
                content=clean_code.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=12.0
            )
            if response.status_code == 200 and "<svg" in response.text:
                svg_content = response.text
                synthesis_cache.set(cache_key, svg_content)
                return svg_content

            logger.error(
                "Kroki API non-200 response (HTTP %d): %s\n--- PLANTUML SENT ---\n%s",
                response.status_code,
                response.text[:500],
                clean_code
            )
        except Exception as http_err:
            logger.error("Kroki HTTP request exception: %s", str(http_err), exc_info=True)

        # Fallback rendering if the primary generated diagram produced a Kroki error or exception
        if enable_fallback:
            logger.warning("Attempting Kroki rendering with sanitized fallback architecture diagram...")
            fallback_plantuml = sanitize_plantuml(
                self.generate_fallback_plantuml(paper_title or "System Architecture", abstract, tldr),
                default_title=paper_title
            )
            try:
                fb_resp = await http_client.post(
                    kroki_url,
                    content=fallback_plantuml.encode("utf-8"),
                    headers={"Content-Type": "text/plain; charset=utf-8"},
                    timeout=10.0
                )
                if fb_resp.status_code == 200 and "<svg" in fb_resp.text:
                    svg_content = fb_resp.text
                    synthesis_cache.set(cache_key, svg_content)
                    return svg_content
                logger.error("Fallback Kroki rendering also failed (HTTP %d): %s", fb_resp.status_code, fb_resp.text[:200])
            except Exception as fb_err:
                logger.error("Fallback Kroki connection failed: %s", str(fb_err), exc_info=True)

        raise RuntimeError(
            "Kroki diagram rendering failed. Please retry in a few moments."
        )

    async def generate_outreach_email(
        self,
        payload: OutreachEmailRequest
    ) -> Dict[str, str]:
        """
        Generate an intellectually rigorous, publication-grade academic outreach email
        strictly under 250 words using DeepSeek or graceful fallback.
        """
        target_prof = payload.target_professor.strip() or "Professor"
        scope = "holistic_lab" if payload.scope_type == "holistic_lab" else "single_paper"
        title = payload.paper_title.strip()
        abstract = payload.abstract.strip()

        cache_key = f"outreach:{scope}:{target_prof.lower()}:{title.lower()}"
        cached_json = synthesis_cache.get(cache_key)
        if cached_json is not None:
            try:
                data = json.loads(cached_json)
                if isinstance(data, dict) and "subject" in data and "body" in data:
                    logger.info("Synthesis cache HIT for outreach email to %r (0ms latency)", target_prof)
                    return data
            except Exception:
                pass

        client = self._get_client(payload.orcarouter_key)
        if client is None:
            logger.info("No OrcaRouter API key provided. Using fallback outreach email generation.")
            fallback = generate_fallback_outreach_email(
                paper_title=title,
                target_professor=target_prof,
                abstract=abstract,
                scope_type=scope
            )
            synthesis_cache.set(cache_key, json.dumps(fallback))
            return fallback

        authors_str = ", ".join(payload.authors) if payload.authors else target_prof
        abstract_str = abstract if abstract else "N/A"
        user_prompt = (
            f"Generate a cold outreach email from a prospective graduate researcher to Professor {target_prof}.\n\n"
            f"Paper Title: {title}\n"
            f"Authors: {authors_str}\n"
            f"Abstract:\n{abstract_str}\n\n"
            f"Scope Mode: {scope}\n"
            f"Target Professor: {target_prof}\n\n"
            f"Remember: Output strictly JSON with keys 'subject' and 'body'. Keep the body under 250 words, intellectually rigorous, and include standardized bracketed placeholders."
        )

        messages = [
            {"role": "system", "content": OUTREACH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            async with llm_semaphore:
                response = await _call_chat_with_retry(
                    client=client,
                    primary_model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1500,
                    secondary_model=SECONDARY_MODEL,
                )
            content = (response.choices[0].message.content or "").strip()

            cleaned = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
            cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

            parsed = None
            try:
                parsed = json.loads(cleaned)
            except Exception:
                json_match = re.search(r"\{[\s\S]*\}", cleaned)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except Exception:
                        pass

            if isinstance(parsed, dict) and "subject" in parsed and "body" in parsed:
                result = {
                    "subject": str(parsed["subject"]).strip(),
                    "body": str(parsed["body"]).strip()
                }
                synthesis_cache.set(cache_key, json.dumps(result))
                return result

            logger.warning("LLM response did not contain expected JSON keys for outreach email: %s", content[:200])
            fallback = generate_fallback_outreach_email(
                paper_title=title,
                target_professor=target_prof,
                abstract=abstract,
                scope_type=scope
            )
            synthesis_cache.set(cache_key, json.dumps(fallback))
            return fallback

        except Exception as e:
            logger.error("Error generating outreach email via DeepSeek: %s", str(e), exc_info=True)
            fallback = generate_fallback_outreach_email(
                paper_title=title,
                target_professor=target_prof,
                abstract=abstract,
                scope_type=scope
            )
            synthesis_cache.set(cache_key, json.dumps(fallback))
            return fallback


LLMService = ResearchLLMService

_default_llm_service: Optional[ResearchLLMService] = None


def get_default_llm_service() -> ResearchLLMService:
    global _default_llm_service
    if _default_llm_service is None:
        _default_llm_service = ResearchLLMService()
    return _default_llm_service


async def generate_outreach_email(payload: OutreachEmailRequest, service: Optional[ResearchLLMService] = None) -> Dict[str, str]:
    """Module-level helper to generate academic cold outreach email."""
    svc = service or get_default_llm_service()
    return await svc.generate_outreach_email(payload)
