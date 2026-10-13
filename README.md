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

## Security

Never commit secrets or access codes. Admin routes stay disabled unless ADMIN_PASSWORD is set. To report a security issue, contact the maintainer through https://swmp-labs.tech.

## Status

Early-stage and maintained by one person. Feedback and bug reports are welcome through GitHub issues.

## Acknowledgements

Thank you to arXiv for use of its open access interoperability. Metadata is also provided by the Semantic Scholar and OpenAlex APIs.

## Author

Marawan Mohamed, Founder & CEO. LinkedIn: https://www.linkedin.com/in/marawan-mohamed-elkzaz-07178a2a7/
