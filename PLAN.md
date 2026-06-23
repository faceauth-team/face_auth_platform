# 📋 Project Plan — Face Authentication Platform

A simple, shared playbook so all three of us build the same way.
Read this first before you start. No prior big-project experience needed.

---

## 1. What we are building (in one picture)

A service that recognizes a clinician by their **face** before they change a
medical record.

```
   ┌─────────────┐        ┌──────────────────┐        ┌─────────────┐
   │   CAMERA    │  face  │   OUR BACKEND    │  yes/no │     EMR     │
   │ (clinician) ├───────►│  (FastAPI + ML)  ├────────►│  (records)  │
   └─────────────┘        └──────────────────┘         └─────────────┘
        look              recognize + check live          allow change
```

Two screens use it:

```
  IT CONSOLE                         CLINICIAN LOGIN
  (enroll a face)                    (verify a face)
  run by IT staff                    run at point of care
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
                 OUR BACKEND
```

---

## 2. How the pieces fit (architecture)

```
                         ┌──────────────────────────────┐
                         │          FRONTEND             │
                         │  enroll-console.html (IT)     │
                         │  auth-kiosk.html (clinician)  │
                         └───────────────┬───────────────┘
                                         │ HTTP (REST API)
                                         ▼
                ┌─────────────────────────────────────────────┐
                │                  BACKEND (app/)               │
                │                                               │
                │   api/   ── the web endpoints (/enroll …)     │
                │   auth/  ── who is allowed (tokens)           │
                │   ml/    ── the face brain (recognize + live) │
                │   db/    ── where data is saved               │
                │   core/  ── settings, logging, errors         │
                │   workers/ ── background jobs                 │
                └───────────────┬───────────────┬───────────────┘
                                │               │
                                ▼               ▼
                         ┌────────────┐   ┌──────────────┐
                         │  DATABASE  │   │ MODEL WEIGHTS │
                         │ (Postgres) │   │ (downloaded)  │
                         └────────────┘   └──────────────┘
```

> **Model weights are NOT in git.** They are large files everyone downloads
> once on their own machine (see `models/README.md`).

---

## 3. Folder structure (what goes where)

```
face-auth-platform/
│
├── README.md            ← project intro
├── PLAN.md              ← this file
├── requirements.txt     ← Python libraries we need
├── .env.example         ← settings template (copy to .env, no secrets in git)
├── Dockerfile           ← how to package the app
│
├── app/                 ← THE BACKEND (all Python code)
│   ├── core/            ← settings, logging, errors  (foundation)
│   ├── db/              ← database tables + connection
│   ├── ml/              ← face recognition + liveness (the "brain")
│   ├── auth/            ← tokens, permissions
│   ├── api/             ← the REST endpoints
│   └── workers/         ← background jobs
│
├── alembic/             ← database version history (migrations)
├── db/                  ← raw SQL schema
├── sdk/                 ← client libraries + the two HTML screens
│   ├── python/          ← Python client
│   ├── js/              ← JavaScript client
│   └── demo/            ← enroll-console.html, auth-kiosk.html
│
├── tests/               ← automated tests
├── infra/               ← docker-compose (run everything together)
├── .github/             ← CI (auto-checks on every push)
└── models/              ← downloaded weights (IGNORED by git)
```

**Golden rule:** one person works in one folder at a time, so we never edit
the same file and collide.

---

## 4. Who builds what (no overlap = no conflicts)

| Person       | Owns these folders            |
|--------------|-------------------------------|
| **Satyam**   | `app/db/`, `app/api/`, `app/auth/` |
| **Member B** | `app/ml/`                     |
| **Member C** | `sdk/`, `tests/`, `docs/`, `.github/` |

---

## 5. Build order (slices) — each one is a Pull Request

We build in small steps. Each step = one branch → one Pull Request → merge.
Later steps need earlier ones, so follow the order.

```
 1. chore/scaffold              ✅ deps + settings + core      [DONE]
 2. feature/data-layer          db tables + migrations
 3. feature/api-skeleton        FastAPI app + /healthz + auth
 4. feature/ml-pipeline         face detect + embed + search
 5. feature/enrollment          /enroll  (register a face)
 6. feature/liveness            anti-spoof (reject photos)
 7. feature/step-up-verification /authorize (1:1 match)
 8. feature/identification      /identify (1:N match)
 9. feature/emr-write           /emr/entries (commit change)
10. feature/audit-logging       audit trail + delete data
11. feature/sdk                 python + js clients
12. feature/ui-consoles         the two HTML screens
13. chore/ci-and-infra          CI + docker-compose
```

**Can start in parallel right now:** #2 (Satyam), #4 (Member B), #11/docs (Member C)
— they live in different folders.

---

## 6. The git workflow (memorize this loop)

We NEVER push code straight to `main`. `main` is protected.
Everyone works on their own branch, then opens a Pull Request.

```
        main  (protected — needs a Pull Request + 1 approval)
          │
          │  1. start fresh
          ▼
   git checkout main
   git pull origin main          ← get everyone's latest work
          │
          │  2. make your branch
          ▼
   git checkout -b feature/my-task
          │
          │  3. do work, then save
          ▼
   git add .
   git commit -m "feat: explain what you did"
          │
          │  4. upload your branch
          ▼
   git push -u origin feature/my-task
          │
          │  5. on GitHub: open Pull Request → teammate approves → Merge
          ▼
   (changes are now in main — everyone runs `git pull origin main`)
```

### Branch name pattern
```
<type>/<short-description>
   │            │
   │            └── lowercase, words-joined-by-hyphens
   └── feature | fix | chore | docs
```
Examples: `feature/enrollment`, `fix/camera-bug`, `docs/setup-guide`

### Commit message pattern
```
<type>: <what you did, present tense>
```
Examples: `feat: add /enroll endpoint`, `fix: handle empty frames`

---

## 7. First-time setup (each person, once)

```bash
# 1. get the project
git clone https://github.com/faceauth-team/face_auth_platform.git
cd face_auth_platform

# 2. create a Python environment + install libraries
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. download the model weights (one time) — see models/README.md

# 4. copy the settings template
copy .env.example .env          # Windows  (cp on Mac/Linux)
```

---

## 8. Golden rules (keep us out of trouble)

1. ✅ Always branch off an up-to-date `main` (`git pull` first).
2. ✅ One folder per person — avoid editing the same files.
3. ✅ Small Pull Requests — easier to review.
4. ✅ Never commit secrets, `.env`, data, or model weights (`.gitignore` guards this).
5. ✅ Get 1 approval before merging.
6. ❌ Never `git push --force` to `main`.

---

*Questions? Ask in the team chat before pushing. When in doubt, make a branch —
branches are cheap and safe.*
