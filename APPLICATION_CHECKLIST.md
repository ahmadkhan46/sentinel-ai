# PhD Application Checklist — UCC FLARE

**Deadline: 27 March 2026**
**Supervisor: Dr Ken Bruton, IERG, School of Engineering and Architecture, UCC**

---

## Documents to Prepare

- [ ] **Cover letter** — `reports/application/cover_letter_ucc_draft.md`
  - Fill in: Full Name, Address, Email, Phone
  - Add your GitHub repository URL (replace `[GitHub URL]`)
  - Review and personalise any phrasing

- [ ] **CV**
  - Education: TUS Athlone, MSc Software Design with AI (expected May–Aug 2026)
  - Skills: Python, PyTorch, scikit-learn, XGBoost, FastAPI, Next.js, Docker
  - Projects: SENTINEL (link to GitHub)
  - Any relevant work experience

- [ ] **Academic transcripts**
  - TUS Athlone MSc (in-progress transcript or current grades)
  - Undergraduate transcript if required

- [ ] **Two referee contacts**
  - Academic referees preferred
  - Confirm they are willing and aware of the application

- [ ] **Project evidence brief (PDF)**
  - Run: `python scripts/generate_project_evidence.py`
  - Output: `reports/application/project_evidence_ucc.pdf`

---

## SENTINEL Codebase Checklist

### Before sharing the GitHub link:

- [ ] Repository is public (or accessible to Dr Bruton)
- [ ] `README.md` is up to date with results table and architecture
  - Current: updated with full benchmark table, architecture diagram, quick-start
- [ ] `RESEARCH_FINDINGS.md` is present and complete (800+ words)
  - Current: written — findings on OCNorm, monotonic RUL, SHAP/LIME agreement
- [ ] Benchmarks run successfully end-to-end:
  - `python scripts/run_benchmarks.py --auto-download`
- [ ] GitHub Actions CI badge is green — badge URL already set to https://github.com/AhmadKhan46/sentinel-ai
- [ ] `.gitignore` excludes: `data/`, `.venv/`, `__pycache__/`, `*.db`, `.env`

### ML pipeline:

- [x] Phase 1: iForest + OCSVM + XGBoost RUL — all 4 subsets
- [x] Phase 2: LSTM/GRU autoencoder — all 4 subsets
- [x] Phase 3: SHAP + LIME + reconstruction error diagnostics
- [x] OCNorm enhancement (FD002, FD004)
- [x] Monotonic RUL enforcement (FD002, FD004)
- [x] Health Index computation
- [x] Digital Twin simulation
- [x] Maintenance metrics (MTTF, early warning lead time, OEE)

### Backend (FastAPI):

- [x] Auth (JWT, bcrypt)
- [x] Asset, sensor, inference, alert, work order, audit routers
- [x] Real-time WebSocket + Redis pub/sub
- [x] Celery workers for async inference + training
- [x] Demo seed data (Acme Manufacturing, 4 engines)
- [ ] Run seed: `python -m api.core.seed`

### Frontend (Next.js):

- [x] Login page (dark glassmorphic)
- [x] Fleet dashboard (animated KPIs, asset grid, live WebSocket banner)
- [x] Asset detail (inference trigger, SHAP chart, trend chart, history)
- [x] Alerts page (card layout, acknowledge/resolve, animated tabs)
- [x] Build verified: `cd frontend && npm run build`

### Infrastructure:

- [x] Dockerfile
- [x] docker-compose.yml (Postgres + Redis + API + worker)
- [x] docker-compose.dev.yml (SQLite + hot-reload)
- [x] Makefile
- [x] GitHub Actions CI (`python lint → pytest → next build`)

---

## Key Numbers to Quote in Interview / Cover Letter

| Metric | Value |
|--------|-------|
| FD002 LSTM AE F1 (baseline) | 0.06 |
| FD002 LSTM AE F1 (OCNorm)   | **0.61** (+916%) |
| FD004 LSTM AE F1 (baseline) | 0.02 |
| FD004 LSTM AE F1 (OCNorm)   | **0.45** (+2,150%) |
| FD001 LSTM AE F1            | 0.7408 |
| FD002 RMSE (monotonic)      | 17.62 (−8.4%) |
| FD004 RMSE (monotonic)      | 15.95 (−12.8%) |
| SHAP/LIME top-5 agreement   | >80% of test instances |

---

## Submission

- [ ] Email to Dr Ken Bruton (check UCC FLARE job posting for submission address)
- [ ] Subject line: `PhD Application — AI-Enabled Fault Detection and Predictive Maintenance — [Your Name]`
- [ ] Attachments: CV, cover letter (PDF), transcripts, evidence brief PDF
- [ ] Body: 2–3 sentences + GitHub link

**Deadline: 27 March 2026 — 8 days from today.**
