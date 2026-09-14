from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import httpx
import os

# ------------------------------------------------------------------
# Nexus Agent - assistente de IA standalone (uso particular)
# Usa Groq (nuvem, gratuito). Sem dependencias do projeto SUS Nexus.
# ------------------------------------------------------------------

app = FastAPI(title="Nexus Agent")

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


PAGINA_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Nexus Agent</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; background: #0f1117; color: #e6e6e6; }
  h1 { font-size: 1.3rem; color: #7dd3fc; }
  #chat { border: 1px solid #2a2d36; border-radius: 8px; padding: 16px; height: 60vh; overflow-y: auto; background: #161821; }
  .msg { margin: 10px 0; line-height: 1.4; white-space: pre-wrap; }
  .user { color: #9fd3ff; }
  .bot { color: #d6f5d6; }
  .label { font-weight: bold; opacity: 0.7; font-size: 0.85rem; }
  form { display: flex; gap: 8px; margin-top: 12px; }
  input[type=text] { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #2a2d36; background: #1c1f29; color: #fff; }
  button { padding: 10px 18px; border-radius: 6px; border: none; background: #3b82f6; color: white; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
  <h1>🤖 Nexus Agent</h1>
  <div id="chat"></div>
  <form id="form">
    <input type="text" id="entrada" placeholder="Digite sua mensagem..." autocomplete="off" required>
    <button type="submit" id="enviar">Enviar</button>
  </form>

<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const entrada = document.getElementById('entrada');
const botaoEnviar = document.getElementById('enviar');
let historico = [];

function adicionarMensagem(texto, classe, label) {
  const div = document.createElement('div');
  div.className = 'msg ' + classe;
  div.innerHTML = '<div class="label">' + label + '</div>' + texto;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const texto = entrada.value.trim();
  if (!texto) return;

  adicionarMensagem(texto, 'user', 'Você');
  historico.push({ role: 'user', content: texto });
  entrada.value = '';
  botaoEnviar.disabled = true;
  adicionarMensagem('Pensando...', 'bot', 'Nexus Agent');
  const placeholder = chat.lastChild;

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: historico, system: 'Você é o Nexus Agent, um assistente pessoal útil e direto.' })
    });
    const data = await resp.json();
    placeholder.remove();
    adicionarMensagem(data.content, 'bot', 'Nexus Agent');
    historico.push({ role: 'assistant', content: data.content });
  } catch (err) {
    placeholder.remove();
    adicionarMensagem('Erro de conexão: ' + err, 'bot', 'Nexus Agent');
  } finally {
    botaoEnviar.disabled = false;
    entrada.focus();
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def raiz():
    return PAGINA_HTML


@app.get("/status")
async def status():
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
