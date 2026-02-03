pip install -r requirements.txt
cp .env.example .env
# Edit .env → add OPENAI_API_KEY
python -m app.indexing
uvicorn app.main:app --host 0.0.0.0 --port 8000
