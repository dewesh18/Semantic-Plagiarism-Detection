# Deployment Guide: PlagiarismAI

This guide details how to deploy **PlagiarismAI** to **Streamlit Community Cloud** and **Hugging Face Spaces**, optimized for memory efficiency, security, and zero-maintenance dependency installation.

---

## 🚀 1. Streamlit Community Cloud (Recommended)

Streamlit Community Cloud provides free, instant hosting directly from your GitHub repository.

### Prerequisites & Files
PlagiarismAI is pre-configured for Streamlit Community Cloud:
- **Python Version**: `.python-version` and `runtime.txt` specify **Python 3.12**.
- **Dependencies**: `requirements.txt` pins compatible versions and installs the `en_core_web_sm` spaCy model wheel directly during build.
- **Theme & Server**: `.streamlit/config.toml` configures an academic blue theme and headless server settings.
- **NLTK Corpora**: Downloaded automatically at startup with offline fallbacks.

### Step-by-Step Deployment
1. **Push your code to GitHub**:
   Ensure your code is committed to your repository (e.g. `main` branch).
   *(Never commit `.env` or `.streamlit/secrets.toml`)*.

2. **Sign in to Streamlit Community Cloud**:
   Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.

3. **Deploy New App**:
   - Click **"New app"**.
   - **Repository**: Select your `plagiarism_detector_AI-` repository.
   - **Branch**: `main`.
   - **Main file path**: `app.py`.
   - **App URL**: Choose a custom subdomain or use the generated default.

4. **Configure Secrets**:
   Click **"Advanced settings..."** -> **"Secrets"**, or go to **Settings > Secrets** in your app dashboard. Paste your secrets in TOML format:

   ```toml
   # LLM Assistant & Translation (Optional but recommended)
   GEMINI_API_KEY = "your_gemini_api_key_here"
   GEMINI_MODEL = "gemini-3.5-flash"

   # Live Web Search (At least one recommended)
   TAVILY_API_KEY = "your_tavily_api_key_here"
   BRAVE_API_KEY = "your_brave_api_key_here"

   # Optional Google Cloud Translation
   GOOGLE_TRANSLATE_API_KEY = ""

   # Optional Access Password & Quota Guard
   APP_PASSWORD = "your_strong_password_here"
   MAX_CHECKS_PER_SESSION = 10

   # Memory Optimization (Required for 1 GB free tier)
   LOW_MEMORY_MODE = "true"
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```

5. **Click "Deploy!"**:
   Streamlit will install dependencies and launch your application.

### Memory Optimization on Free Tier
Streamlit Community Cloud enforces a strict **1 GB RAM limit** per container:
- **Low-Memory Mode (`LOW_MEMORY_MODE = "true"`)**: PlagiarismAI uses `sentence-transformers/all-MiniLM-L6-v2` (~578 MB RAM total process footprint), easily fitting within the 1 GB container.
- **Multilingual Handling**: When Hindi or Marathi text is checked, the pipeline translates the passage to English first, allowing `all-MiniLM-L6-v2` to perform semantic sentence matching with high accuracy without loading a 1.3 GB multilingual model.
- **Lazy Loading**: Heavy transformer models are loaded only when the user clicks **"Check my document"**, ensuring lightning-fast startup and zero memory consumption for idle instances.

---

## 🤗 2. Hugging Face Spaces

Hugging Face Spaces provides 16 GB RAM on its free CPU tier, offering extra headroom for larger corpora.

### Step-by-Step Deployment
1. **Create Space**:
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
   - **Space name**: `plagiarism-ai`.
   - **License**: `mit` or `apache-2.0`.
   - **Space SDK**: Select **Streamlit**.
   - **Space hardware**: `CPU basic • 2 vCPU • 16GB RAM` (Free).

2. **Upload / Push Repository**:
   - Clone your Hugging Face Space repository locally:
     ```bash
     git remote add space https://huggingface.co/spaces/<your-username>/plagiarism-ai
     git push space main
     ```
   - Or connect your GitHub repository directly under Space Settings.

3. **Configure Secrets / Environment Variables**:
   In your Space dashboard:
   - Click **Settings** -> **Variables and secrets**.
   - Under **Secrets**, add:
     - `GEMINI_API_KEY` = your Gemini key
     - `TAVILY_API_KEY` = your Tavily key
     - `APP_PASSWORD` = (optional password to protect your space)
     - `MAX_CHECKS_PER_SESSION` = `10`
     - `LOW_MEMORY_MODE` = `true`

4. **Verify Build**:
   The Space will automatically build `requirements.txt` and launch `app.py`.

---

## 🔒 3. Access Control & Quota Protection

When sharing a public demo, search API queries (Tavily/Brave) consume limited API quotas:
1. **Password Barrier**:
   Setting `APP_PASSWORD` in your cloud secrets locks the entire application behind an access password. If unset, access is open to all visitors.
2. **Per-Session Rate Limiting**:
   `MAX_CHECKS_PER_SESSION` (default: `10`) limits how many document checks a user can run in a single browser session before being asked to refresh.
3. **Storage Persistence**:
   On cloud containers with ephemeral filesystems, `plagiarism_history.db` and `translation_cache.json` automatically fall back to the system temporary directory if the root directory is unwritable. A friendly note is displayed in the History tab.

---

## 🧪 4. Local Pre-Deployment Verification

Before pushing to production, verify the app and test suite locally:

```bash
# 1. Run all 78 automated unit and integration tests
python run.py --test

# 2. Test live search provider connection
python app.py --test-search "neural network language models"

# 3. Launch local Streamlit server
python run.py
```
