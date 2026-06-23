# Project Plan - Face Authentication Platform

This is our shared plan so all of us build the project the same way.
Please read this first before starting. You don't need big project
experience, everything is explained simply here.

---

## 1. What we are building

A service that recognizes a clinician by their face before they change a
medical record.

```
   ┌─────────────┐        ┌──────────────────┐        ┌─────────────┐
   │   CAMERA    │  face  │   OUR BACKEND    │  yes/no │     EMR     │
   │ (clinician) ├───────►│  (FastAPI + ML)  ├────────►│  (records)  │
   └─────────────┘        └──────────────────┘         └─────────────┘
        look              recognize + check live          allow change
```

There are two screens that use it:

```
  IT CONSOLE                         CLINICIAN LOGIN
  (register a face)                  (verify a face)
  used by IT staff                   used by doctors
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
                 OUR BACKEND
```

---

## 2. How the pieces fit together

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
                │   api/   the web endpoints (/enroll etc.)     │
                │   auth/  who is allowed (tokens)              │
                │   ml/    the face brain (recognize + live)    │
                │   db/    where data is saved                  │
                │   core/  settings, logging, errors            │
                │   workers/ background jobs                    │
                └───────────────┬───────────────┬───────────────┘
                                │               │
                                ▼               ▼
                         ┌────────────┐   ┌──────────────┐
                         │  DATABASE  │   │ MODEL WEIGHTS │
                         │ (Postgres) │   │ (downloaded)  │
                         └────────────┘   └──────────────┘
```

Note: the model weights are big files, so we do NOT put them in git.
Everyone downloads them once on their own machine. See models/README.md.

---

## 3. Folder structure (what goes where)

```
face-auth-platform/
│
├── README.md            project intro
├── PLAN.md              this file
├── requirements.txt     python libraries we need
├── .env.example         settings template (copy to .env)
├── Dockerfile           how to package the app
│
├── app/                 THE BACKEND (all python code)
│   ├── core/            settings, logging, errors  (the base)
│   ├── db/              database tables + connection
│   ├── ml/              face recognition + liveness (the brain)
│   ├── auth/            tokens, permissions
│   ├── api/             the REST endpoints
│   └── workers/         background jobs
│
├── alembic/             database version history (migrations)
├── db/                  raw SQL schema
├── sdk/                 client libraries + the two HTML screens
│   ├── python/          python client
│   ├── js/              javascript client
│   └── demo/            enroll-console.html, auth-kiosk.html
│
├── tests/               automated tests
├── infra/               docker-compose (run everything together)
├── .github/             CI (auto checks on every push)
└── models/              downloaded weights (git ignores this)
```

One simple habit that saves us a lot of pain: try to work in one folder at
a time, so two people are not editing the same file together.

---

## 4. Build order (we build in small steps)

We build the project step by step. Each step is one branch, then one Pull
Request, then merge. Some steps need the earlier ones, so follow the order.

```
 1. chore/scaffold              deps + settings + core      [DONE]
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

Steps 2, 4 and 11 live in different folders, so they can be done at the
same time without clashing.

---

## 5. How we use git (this is the important part)

We never push code straight to main. The main branch is protected, so every
change has to go through a Pull Request and get 1 approval first.

Each of us works on our own branch like this:

```
        main  (protected, needs a Pull Request + 1 approval)
          │
          │  1. start fresh
          ▼
   git checkout main
   git pull origin main          get everyone's latest work
          │
          │  2. make your own branch
          ▼
   git checkout -b feature/my-task
          │
          │  3. do the work, then save it
          ▼
   git add .
   git commit -m "feat: say what you did"
          │
          │  4. upload your branch
          ▼
   git push -u origin feature/my-task
          │
          │  5. on GitHub: open Pull Request, teammate approves, then Merge
          ▼
   now it is in main, everyone runs `git pull origin main`
```

How to name a branch:

```
<type>/<short-description>
   │            │
   │            └── lowercase, words joined by hyphens
   └── feature | fix | chore | docs
```

Examples: feature/enrollment, fix/camera-bug, docs/setup-guide

How to write a commit message:

```
<type>: <what you did>
```

Examples: feat: add /enroll endpoint, fix: handle empty frames

---

## 6. First time setup (each person, only once)

```bash
# 1. get the project
git clone https://github.com/faceauth-team/face_auth_platform.git
cd face_auth_platform

# 2. make a python environment and install the libraries
python -m venv .venv
.venv\Scripts\activate          # windows
pip install -r requirements.txt

# 3. download the model weights (one time). see models/README.md

# 4. copy the settings template
copy .env.example .env          # windows  (use cp on mac/linux)
```

---

## 7. Few simple rules to keep us safe

1. Always pull main before making a new branch.
2. Try to stay in one folder so we don't edit the same files.
3. Keep Pull Requests small, they are easier to review.
4. Never commit secrets, .env, data, or model weights. The .gitignore
   already blocks these, but be careful.
5. Get 1 approval before merging.
6. Never force push to main.

---

If something is confusing, just ask in our group before pushing. When in
doubt, make a branch. Branches are cheap and safe.
