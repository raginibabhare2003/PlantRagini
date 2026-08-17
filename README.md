# 🌱 Smart Crop AI — International AgriTech Prototype

A Flask-based, multilingual agricultural decision-support prototype combining broad plant identification, local disease screening, AI assistance, weather intelligence, farmer history, and safe fallbacks.

## What changed in this build

- **Pl@ntNet API-first plant identification** using the `all` project. This removes the 25-class TFLite model as the only plant-identification path.
- Existing **TFLite disease model is preserved** as an offline/local disease fallback for its actual supported classes.
- **Gemini Vision** can analyze disease/symptoms when configured.
- **Multilingual farmer assistant** with conversation context.
- **Offline assistant fallback** instead of crashing on Gemini 429/quota/network errors.
- Image quality checks for blurry/dark/very small images.
- Secure API-key configuration through `.env`.
- `/api/health` endpoint for deployment checks.
- Existing login, dashboard, feedback, admin, weather, camera, history, and supported-plants functionality preserved.

## Important limitation

No responsible system can guarantee identification of every plant or disease. Pl@ntNet returns ranked species predictions and confidence scores; this application rejects low-confidence plant matches instead of forcing a name. The current TFLite disease model remains limited to the classes in `labels.txt`.

## Setup

1. Create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`.
4. Add your **Pl@ntNet API key** to `PLANTNET_API_KEY`.
5. Optionally add `GEMINI_API_KEY` for AI disease analysis and the full farmer assistant.
6. Set your **MySQL** connection details (`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`) in `.env`. The database itself (e.g. `smart_crop_ai`) must already exist — the app creates the tables inside it automatically on first run.
7. Run: `python app.py`
8. Open: `http://127.0.0.1:5000`

## Database (MySQL)

The app now uses MySQL instead of SQLite (via `PyMySQL`). On startup it runs `init_db()`, which creates the `users`, `scans`, `feedback`, `suggestions`, and `contact_messages` tables if they don't exist yet, and promotes `ADMIN_EMAIL` to an admin account.

Locally (Windows), the simplest options are XAMPP/WAMP (bundles MySQL) or installing MySQL Community Server directly, then creating the database with:
```sql
CREATE DATABASE smart_crop_ai;
```

## Render / Gunicorn

Start command: `gunicorn app:app` (already set in the included `Procfile`, so Render picks it up automatically).

Render does not offer a managed MySQL database, so use an external MySQL host (e.g. Railway, Aiven, Clever Cloud, PlanetScale, or Hostinger) and put its connection details in Render's Environment settings as `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` — along with the other variables from `.env.example`. Never commit `.env` or API keys.

## Detection flow

`Image → Quality Check → Pl@ntNet species identification → local disease model when applicable → Gemini disease analysis when available → safe/unknown fallback`

## Mobile camera & fast multilingual chat (latest update)

- **Mobile camera**: the live-camera capture now has a front/back flip button and clear, specific error messages (permission denied, no camera found, camera busy, or the page not being served over HTTPS — mobile browsers block camera access on plain `http://`). Deploy behind HTTPS (Render/any real host does this automatically) for the camera to work on phones.
- **Chat in any language**: `/api/chat` now auto-detects the language the farmer actually typed in (via `langdetect`) and replies in that same language, instead of only replying in whatever language is selected in the site dropdown. Run `pip install -r requirements.txt` to pick up the new `langdetect` dependency.
- **Instant language switching**: the site-wide text dictionary and disease-guidance text used to be translated one string at a time (slow — many sequential network calls on the first visit to a new language). Translations are now fetched concurrently and cached to disk in `translation_cache/` (gitignored), so the *first* switch to a language is much faster and every switch after that is effectively instant, served straight from cache.

## Safety

The system is decision support, not a substitute for a qualified agricultural diagnosis. Chemical guidance is intentionally conservative and tells users to follow locally approved product labels and expert advice.
