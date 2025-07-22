# Metaforms Assignment

FastAPI service for converting unstructured text into structured format strictly following a desired JSON schema.

## Quick Start

1. **Activate virtual environment**
   ```bash
   source myenv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   mypy --install-types
   ```
3. **Add environment variables**
   ```bash
   tounch .env
   ```
   add OPENAI_API_KEY 
4. **Run server**
   ```bash
   python -m uvicorn main:app --reload
   ```

5. **Access API docs**
   - http://localhost:8000/docs

## Testing

**Postman collection** is available in the repository:
- Import `test.postman_collection.json` into Postman for ready-to-use API tests

## Development

**Type checking:**
```bash
mypy .
```

**Linting/formatting:**
```bash
ruff check .
ruff format .
```

Configuration in `pyproject.toml`.
