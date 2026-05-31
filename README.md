# CV Counseling RAG Assistant

An AI-powered chat application that analyzes resumes and provides personalized career advice. Built as a Bachelor's thesis project at Universita degli Studi di Bergamo.

## How it works

The system implements a Retrieval-Augmented Generation (RAG) pipeline:

1. **Document ingestion** — PDF resumes are parsed (PyMuPDF / pdf2image + poppler) and indexed.
2. **Retrieval** — LlamaIndex queries the indexed documents for relevant context.
3. **Generation** — OpenAI GPT produces tailored feedback on job titles, summary sections, design, salary expectations, and career paths.

Users interact through a Streamlit web UI, uploading their CV and engaging in multi-turn dialogue.

## Tech stack

- **Framework:** Streamlit
- **RAG pipeline:** LlamaIndex (readers, indices, query engine)
- **LLM:** OpenAI GPT (via API)
- **PDF processing:** PyMuPDF, pdf2image, poppler
- **Data handling:** pandas, NumPy, openpyxl

## Running locally

```bash
pip install -r requirements.txt
streamlit run th3sis_OpenAI.py
```

Requires an OpenAI API key configured in Streamlit secrets (`.streamlit/secrets.toml`).

## Variants

| File | Approach |
|------|----------|
| `th3sis_OpenAI.py` | Direct OpenAI API + LlamaIndex RAG |
| `th3sis_Llama.py` | LlamaIndex with OpenAI backend |
| `th3sis_Langchain.py` | LangChain orchestration |

## License

MIT
