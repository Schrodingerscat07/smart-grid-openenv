# Agent's "Living Memory" (AGENTS.md)

This file is a dynamic reference for all AI agents working on this repository to avoid repeating mistakes and handle environment-specific "gotchas."

## 📁 Canonical Directory Structure (Hackathon Admin Standard)

To pass the automated Unstop/Hugging Face graders, you MUST follow this exact structure:

- **Root/**
    - **`server/`** (Contains the core simulation/API)
        - `__init__.py`
        - `app.py` (Entry point for FastAPI)
        - `grid_env.py` (Main Environment class)
        - `simulator.py`, `tasks.py` (Logic)
    - **`models.py`** (Pydantic Action/Observation/StepResult MUST be in the root)
    - **`openenv.yaml`**, **`Dockerfile`**, **`inference.py`**, **`pyproject.toml`** (All MUST be in the root)

---

## 🛡️ OpenEnv CLI & Environment Lessons

1.  **Validation Rule (Strict Regex)**:
    *   The `openenv validate` command uses a very strict check for the `if __name__ == "__main__":` block in `server/app.py`.
    *   **Rule**: `main()` must be called directly after the check. DO NOT include imports or extra logic inside the block.
    *   **Correct**:
        ```python
        if __name__ == "__main__":
            main()
        ```

2.  **CLI is not a Directory**:
    *   `openenv` is an executable command, not a folder. You cannot `cd` into it.
    *   Always run it from the project root using `.\venv\Scripts\openenv`.

3.  **UTF-8 Encoding (Windows)**:
    *   The CLI uses checkmark symbols (✓) which fail on some Windows terminals.
    *   **Fix**: Set `$env:PYTHONUTF8=1` before running `openenv init`.

---

## 💻 Windows/PowerShell Execution Policy

Windows blocks script execution by default. To activate the virtual environment (`venv`), always run this first in each terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

---

## 🚀 Hackathon Winning Strategy

*   **Submissions**: Must be links to **Hugging Face Spaces**.
*   **Evaluation**: The judges (including LLMs) want rich `metadata` in the `step()` function for better observability.
*   **Verification**: Always run `openenv validate <env_name>` before pushing to Hugging Face.
