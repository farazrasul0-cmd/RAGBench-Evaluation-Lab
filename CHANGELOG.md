# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-13

### Added
* **Academic Research Workbench:** Full-featured React 19 / Vite / Tailwind exploratory dashboard featuring multidimensional configuration sweeps, active Pareto frontier comparability checks, bounded radar metric plots, and claim-level citation trace attribution inspection.
* **Automated Publication Report Generator:** CLI command `ragbench report generate` that derives LaTeX tables (`booktabs`), executive Markdown summaries, and academic whitepaper manuscripts dynamically from frozen replication evidence.
* **Self-Contained LaTeX Distribution:** Bundled `IEEEtran.cls` (v1.8b CTAN class) as a repository template asset to guarantee local compilability without external TeX dependencies.
* **Production Containerization:** Hardened multi-stage Dockerfiles (`python:3.12-slim` runner with unprivileged `appuser`, `node:22-alpine` builder, `nginx:alpine` SPA router) and Docker Compose deployment topology.
* **Continuous Integration Quality Matrix:** GitHub Actions dual-job pipeline enforcing Ruff linting, format checking, strict Mypy typing, Pytest regressions, Oxlint, strict TypeScript compiler checks, and Vitest execution on all pushes and version tags.
* **Cryptographic Verification Manifest:** Dedicated `docs/ARTIFACT_HASHES.md` cataloging canonical SHA-256 digests for all benchmark corpora and empirical reference databases.

### Fixed
* **Dynamic Report Provenance:** Refactored `AcademicReportGenerator` to parse and validate authoritative evidence from `statistical_summary.json`, removing static hardcoded string templates.
* **H3 Hypothesis Statistics:** Resolved statistical reporting discrepancy for Hypothesis H3, ensuring true paired attenuation metrics ($\bar{\Delta}=-0.0989, d_z=-2.01, p < 0.001$) are rendered rather than unrelated single-condition figures.
* **CORS Security Hardening:** Replaced wildcard `allow_origins=["*"]` with explicit frontend origins (`http://localhost:3000`, `http://localhost:5173`) compliant with W3C credential standards.
* **Docker Compose Hygiene:** Removed orphaned top-level named volume declarations in favor of explicit host bind-mounts.
* **Version Metadata Alignment:** Synchronized frontend package metadata with backend `1.0.0` release version.

### Verified
* **Backend Regression Suite:** 268 passed tests with zero errors, including forensic data extraction, statistical boundary conditions, and CORS enforcement.
* **Frontend Test Suite:** 23 passed Vitest component tests with zero warnings under Oxlint and zero TypeScript compilation errors.
* **Research Baseline Immutability:** Verified byte-for-byte immutability of Phase G (`eb3bb4d`) empirical SQLite databases and replication bundle archives.
