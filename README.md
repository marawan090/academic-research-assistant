# SWMP Labs

Systems literature discovery and synthesis for researchers in distributed systems, cloud infrastructure and AI systems.

Live app: https://app.swmp-labs.tech
Website: https://swmp-labs.tech

## What it does

SWMP Labs searches arXiv, Semantic Scholar and OpenAlex together, ranks and de-duplicates the results, and helps you go from a list of papers to a comparison, a diagram, a citation and an email. It is focused on four arXiv categories: cs.DC, cs.SE, cs.AI and cs.AR.

## Features

- Multi-source search by topic, author, university or lab, or a custom query, with year and source filters.
- Two-paper trade-off matrix covering architectural bottlenecks, bounds, complexity ceilings and failure modes. Copy as Markdown or print to PDF.
- Interactive architecture diagram of a paper's system design (drag, zoom), exportable as SVG or PNG.
- Citation export in BibTeX, IEEE and APA 7th edition.
- Outreach email drafts for a professor or lab, about a single paper or a whole lab, with bracketed placeholders to personalize before sending.
- Saved sessions, artifacts and bookmarks for signed-in users.

## How it is built

- Backend: FastAPI with async HTTP clients and a shared connection pool.
- Search layer (services/search.py): concurrent queries to arXiv, Semantic Scholar and OpenAlex, query cleaning, semantic concept expansion, de-duplication and ranking, and discovery of linked code repositories.
- Synthesis layer (services/llm.py): streaming paper analysis, comparison matrices and outreach drafts through a configurable OpenAI-compatible model router.
- Access and accounts (services/access.py, services/db.py): Supabase authentication, invite-based onboarding and per-user credits.
- Telemetry (services/telemetry.py): in-memory request log used for abuse prevention and moderation.
- Frontend: server-rendered Jinja template with a single-page interface in static/.

## Project structure

```text
main.py            application entry point and API routes
services/          search, LLM, access, database and telemetry modules
templates/         HTML templates
static/            JavaScript, CSS and images
data/              runtime data directory (not tracked)
.env.example       environment variable reference
requirements.txt   Python dependencies
```

## Getting started

Requirements: Python 3.10 or newer.

1. Clone the repository and create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

4. Start the server:

```bash
uvicorn main:app --reload
```

5. Open http://127.0.0.1:8000

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| ORCAROUTER_API_KEY | yes | Key for the model router |
| ORCAROUTER_BASE_URL | no | Router base URL |
| MODEL_NAME | no | Model identifier, configurable |
| SEMANTIC_SCHOLAR_API_KEY | no | Raises Semantic Scholar rate limits |
| ADMIN_PASSWORD | no | Enables admin routes; admin is disabled when unset |
| MASTER_CODE | no | Optional administrative master code |
| MASTER_ADMIN_KEYS | no | Optional administrative master keys, comma-separated |
| RESEARCHER_CODES | no | Optional seed list for cohort access codes, comma-separated |
| INVITE_CODES | no | Optional seed list for invite codes, comma-separated |
| RESEARCHER_CODE_EXPIRY | no | Expiry date for researcher codes (YYYY-MM-DD) |
| ALLOWED_ORIGINS | no | Comma-separated CORS allow-list |
| HOST | no | Server bind address |
| PORT | no | Server bind port |
| ENVIRONMENT | no | Set to test only in automated tests |
| Supabase variables | — | See `.env.example` for the Supabase URL and key variables |

## Deployment

The hosted app runs on Render. All secrets are set as environment variables in the hosting dashboard and are never committed. On a free instance the first request after idle time can take up to 45 seconds while the server wakes up.

## Security

Never commit secrets or access codes. Admin routes stay disabled unless ADMIN_PASSWORD is set. To report a security issue, contact the maintainer through https://swmp-labs.tech.

## Status

Early-stage and maintained by one person. Feedback and bug reports are welcome through GitHub issues.

## Acknowledgements

Thank you to arXiv for use of its open access interoperability. Metadata is also provided by the Semantic Scholar and OpenAlex APIs.

## Author

Marawan Mohamed, Founder & CEO. LinkedIn: https://www.linkedin.com/in/marawan-mohamed-elkzaz-07178a2a7/
