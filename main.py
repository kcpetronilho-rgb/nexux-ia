from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx
import os

# ------------------------------------------------------------------
# Nexus Agent - assistente de IA standalone (uso particular)
# Roda 100% local via Ollama, sem chave de API, sem dependencias
# do projeto SUS Nexus.
# ------------------------------------------------------------------

app = FastAPI(title="Nexus Agent")

# Libera acesso local (util se for chamar de um frontend separado)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


class Mensagem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Mensagem]
    system: str = ""


@app.get("/")
async def raiz():
    return {"status": "ok", "agente": "Nexus Agent", "modelo": GROQ_MODEL}


@app.post("/chat")
async def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        return {"content": "GROQ_API_KEY nao configurada. Defina a variavel de ambiente com sua chave gratuita do Groq."}

    mensagens = []
    if payload.system:
        mensagens.append({"role": "system", "content": payload.system})
    mensagens.extend([m.model_dump() for m in payload.messages])

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": mensagens,
                },
            )
            data = resp.json()
            if "choices" not in data:
                erro_groq = data.get("error", data)
                return {"content": f"Erro retornado pelo Groq: {erro_groq}"}
            texto = data["choices"][0]["message"]["content"]
            return {"content": texto}
    except Exception as e:
        return {"content": f"Erro ao consultar IA: {str(e)[:150]}"}
