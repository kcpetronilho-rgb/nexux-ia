from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import httpx
import os
import re
import json
from urllib.parse import quote
from html.parser import HTMLParser

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


URL_REGEX = re.compile(r"https?://[^\s<>\"']+")
MAX_CHARS_POR_PAGINA = 6000


class ExtratorDeTexto(HTMLParser):
    """Extrai texto visivel de uma pagina HTML, ignorando script/style/tags."""

    def __init__(self):
        super().__init__()
        self.partes = []
        self.ignorar = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.ignorar += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.ignorar > 0:
            self.ignorar -= 1

    def handle_data(self, data):
        if self.ignorar == 0:
            texto = data.strip()
            if texto:
                self.partes.append(texto)

    def texto_final(self):
        return "\n".join(self.partes)


async def ler_pagina(url: str) -> str:
    """Busca uma URL e retorna um resumo do texto extraido da pagina."""
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NexusAgent/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            extrator = ExtratorDeTexto()
            extrator.feed(resp.text)
            texto = extrator.texto_final()
            texto = re.sub(r"\n{2,}", "\n", texto).strip()
            if len(texto) > MAX_CHARS_POR_PAGINA:
                texto = texto[:MAX_CHARS_POR_PAGINA] + "\n[...conteudo truncado...]"
            return texto or "(nao foi possivel extrair texto legivel desta pagina)"
    except Exception as e:
        return f"(erro ao acessar {url}: {str(e)[:150]})"


RESULTADO_REGEX = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)


def limpar_html_simples(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


async def buscar_na_web(query: str) -> str:
    """Pesquisa no DuckDuckGo e retorna os principais resultados (titulo, resumo, link)."""
    try:
        url = "https://html.duckduckgo.com/html/?q=" + quote(query)
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NexusAgent/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            achados = RESULTADO_REGEX.findall(resp.text)
            if not achados:
                return f"Nenhum resultado encontrado para: {query}"

            resultados = []
            for link, titulo, snippet in achados[:5]:
                titulo = limpar_html_simples(titulo)
                snippet = limpar_html_simples(snippet)
                resultados.append(f"- {titulo}\n  {snippet}\n  Fonte: {link}")

            return f"Resultados da pesquisa por \"{query}\":\n\n" + "\n\n".join(resultados)
    except Exception as e:
        return f"(erro ao pesquisar \"{query}\": {str(e)[:150]})"


FERRAMENTAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_na_web",
            "description": (
                "Pesquisa na internet informacoes atuais, recentes ou que voce nao tem certeza "
                "(noticias, precos, eventos, dados atualizados, fatos especificos). "
                "Use sempre que a pergunta exigir informacao que pode ter mudado ou que voce nao sabe com certeza."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Termos de busca, em poucas palavras",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


PAGINA_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nexus Agent</title>
<style>
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: radial-gradient(ellipse at center, #14152b 0%, #06060f 70%);
    color: #e8e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden;
  }
  #stars, #sphere { position: fixed; top: 0; left: 0; width: 100%; height: 100%; }
  #stars { z-index: 0; }
  #sphere { z-index: 1; }

  header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 28px;
  }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo .diamond { font-size: 1.4rem; color: #7dd3fc; }
  .logo-text { line-height: 1.1; }
  .logo-text .title { font-size: 1.3rem; font-weight: 700; letter-spacing: 2px; color: #cdeaff; }
  .logo-text .subtitle { font-size: 0.65rem; letter-spacing: 1.5px; color: #6b7280; }

  .status-pill {
    display: flex; align-items: center; gap: 6px;
    background: rgba(20, 30, 25, 0.6); border: 1px solid rgba(74, 222, 128, 0.3);
    padding: 6px 14px; border-radius: 20px; font-size: 0.75rem; color: #4ade80;
  }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; box-shadow: 0 0 8px #4ade80; animation: pulse 1.8s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  #painel {
    position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%);
    z-index: 5; width: min(560px, 90vw); max-height: 34vh; overflow-y: auto;
    background: rgba(12, 14, 26, 0.72); backdrop-filter: blur(10px);
    border: 1px solid rgba(125, 211, 252, 0.15); border-radius: 14px;
    padding: 16px 20px; font-size: 0.9rem; line-height: 1.5;
  }
  #painel .msg { margin-bottom: 14px; white-space: pre-wrap; }
  #painel .label { font-weight: 700; font-size: 0.75rem; letter-spacing: 1px; opacity: 0.75; margin-bottom: 3px; }
  #painel .user .label { color: #93c5fd; }
  #painel .bot .label { color: #7dd3fc; }
  #painel .user { color: #dbeafe; }
  #painel .bot { color: #d9f2ea; }

  #barra {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    z-index: 10; width: min(560px, 90vw);
    display: flex; align-items: center; gap: 10px;
    background: rgba(15, 16, 30, 0.85); backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 255, 0.15);
    border-radius: 30px; padding: 8px 8px 8px 18px;
  }
  #mic {
    width: 38px; height: 38px; border-radius: 50%; border: none;
    background: rgba(255,255,255,0.06); color: #a5b4fc; font-size: 1rem; cursor: pointer; flex-shrink: 0;
  }
  #entrada {
    flex: 1; background: transparent; border: none; outline: none;
    color: #f1f1f6; font-size: 0.95rem; padding: 8px 4px;
  }
  #entrada::placeholder { color: #6b7280; }
  #enviar {
    border: none; border-radius: 24px; padding: 10px 20px; font-weight: 600;
    background: linear-gradient(135deg, #7c3aed, #6366f1); color: white; cursor: pointer;
    font-size: 0.9rem; flex-shrink: 0;
  }
  #enviar:disabled { opacity: 0.5; cursor: default; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-thumb { background: rgba(125,211,252,0.25); border-radius: 3px; }
</style>
</head>
<body>

<canvas id="stars"></canvas>
<canvas id="sphere"></canvas>

<header>
  <div class="logo">
    <span class="diamond">◆</span>
    <div class="logo-text">
      <div class="title">NEXUS</div>
      <div class="subtitle">AI AGENT</div>
    </div>
  </div>
  <div class="status-pill"><span class="status-dot"></span> ONLINE</div>
</header>

<div id="painel"></div>

<div id="barra">
  <button id="mic" type="button" title="Voz (em breve)">🎤</button>
  <input type="text" id="entrada" placeholder="Digite sua pergunta..." autocomplete="off">
  <button id="enviar" type="button">Enviar ✦</button>
</div>

<script>
// ---------- Fundo estrelado ----------
const starsCanvas = document.getElementById('stars');
const starsCtx = starsCanvas.getContext('2d');
let stars = [];

function resize() {
  starsCanvas.width = window.innerWidth;
  starsCanvas.height = window.innerHeight;
  sphereCanvas.width = window.innerWidth;
  sphereCanvas.height = window.innerHeight;
}

function initStars() {
  stars = [];
  const n = Math.floor((window.innerWidth * window.innerHeight) / 4000);
  for (let i = 0; i < n; i++) {
    stars.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.2 + 0.2,
      a: Math.random()
    });
  }
}

function drawStars() {
  starsCtx.clearRect(0, 0, starsCanvas.width, starsCanvas.height);
  for (const s of stars) {
    s.a += (Math.random() - 0.5) * 0.02;
    s.a = Math.max(0.1, Math.min(1, s.a));
    starsCtx.beginPath();
    starsCtx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
    starsCtx.fillStyle = `rgba(255,255,255,${s.a})`;
    starsCtx.fill();
  }
}

// ---------- Esfera de partículas 3D ----------
const sphereCanvas = document.getElementById('sphere');
const sphereCtx = sphereCanvas.getContext('2d');
let particles = [];
let angle = 0;

function initSphere() {
  particles = [];
  const count = 900;
  const radius = Math.min(window.innerWidth, window.innerHeight) * 0.16;
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = golden * i;
    particles.push({
      x: Math.cos(theta) * r * radius,
      y: y * radius,
      z: Math.sin(theta) * r * radius,
      hue: Math.random() * 360
    });
  }
}

function drawSphere() {
  sphereCtx.clearRect(0, 0, sphereCanvas.width, sphereCanvas.height);
  const cx = window.innerWidth / 2;
  const cy = window.innerHeight * 0.4;
  angle += 0.0022;

  const projected = particles.map(p => {
    const cosA = Math.cos(angle), sinA = Math.sin(angle);
    const x = p.x * cosA - p.z * sinA;
    const z = p.x * sinA + p.z * cosA;
    const scale = 500 / (500 + z);
    return {
      x: cx + x * scale,
      y: cy + p.y * scale,
      scale,
      hue: p.hue,
      z
    };
  });

  projected.sort((a, b) => a.z - b.z);

  for (const p of projected) {
    const size = Math.max(0.6, 2.2 * p.scale);
    sphereCtx.beginPath();
    sphereCtx.arc(p.x, p.y, size, 0, Math.PI * 2);
    sphereCtx.fillStyle = `hsla(${p.hue}, 90%, 65%, ${0.5 + p.scale * 0.4})`;
    sphereCtx.shadowBlur = 6;
    sphereCtx.shadowColor = `hsla(${p.hue}, 90%, 65%, 0.8)`;
    sphereCtx.fill();
  }
  sphereCtx.shadowBlur = 0;
}

function loop() {
  drawStars();
  drawSphere();
  requestAnimationFrame(loop);
}

window.addEventListener('resize', () => { resize(); initStars(); initSphere(); });
resize();
initStars();
initSphere();
loop();

// ---------- Chat ----------
const painel = document.getElementById('painel');
const entrada = document.getElementById('entrada');
const botaoEnviar = document.getElementById('enviar');
let historico = [];

function adicionarMensagem(texto, classe, label) {
  const div = document.createElement('div');
  div.className = 'msg ' + classe;
  div.innerHTML = '<div class="label">' + label + '</div>' + texto;
  painel.appendChild(div);
  painel.scrollTop = painel.scrollHeight;
}

async function enviarMensagem() {
  const texto = entrada.value.trim();
  if (!texto) return;

  adicionarMensagem(texto, 'user', 'VOCÊ');
  historico.push({ role: 'user', content: texto });
  entrada.value = '';
  botaoEnviar.disabled = true;
  adicionarMensagem('Processando no núcleo quântico...', 'bot', 'NEXUS');
  const placeholder = painel.lastChild;

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: historico, system: 'Você é o Nexus, um assistente de IA pessoal. Responda em português do Brasil, de forma clara e direta.' })
    });
    const data = await resp.json();
    placeholder.remove();
    adicionarMensagem(data.content, 'bot', 'NEXUS');
    historico.push({ role: 'assistant', content: data.content });
  } catch (err) {
    placeholder.remove();
    adicionarMensagem('Erro de conexão: ' + err, 'bot', 'NEXUS');
  } finally {
    botaoEnviar.disabled = false;
    entrada.focus();
  }
}

botaoEnviar.addEventListener('click', enviarMensagem);
entrada.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); enviarMensagem(); }
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


async def chamar_groq(mensagens: list, com_ferramentas: bool = True) -> dict:
    payload = {
        "model": GROQ_MODEL,
        "messages": mensagens,
    }
    if com_ferramentas:
        payload["tools"] = FERRAMENTAS
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        return resp.json()


@app.post("/chat")
async def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        return {"content": "GROQ_API_KEY nao configurada. Defina a variavel de ambiente com sua chave gratuita do Groq."}

    mensagens = []
    if payload.system:
        mensagens.append({"role": "system", "content": payload.system})

    # Leitura automatica de URL(s) coladas na mensagem (alem da busca via function calling)
    if payload.messages:
        ultima = payload.messages[-1]
        urls = URL_REGEX.findall(ultima.content) if ultima.role == "user" else []
        if urls:
            paginas_lidas = []
            for url in urls[:3]:
                conteudo = await ler_pagina(url)
                paginas_lidas.append(f"### Conteudo de {url}\n{conteudo}")
            contexto = (
                "O usuario compartilhou o(s) link(s) abaixo. Use esse conteudo "
                "para responder a pergunta dele.\n\n" + "\n\n".join(paginas_lidas)
            )
            mensagens.append({"role": "system", "content": contexto})

    mensagens.extend([m.model_dump() for m in payload.messages])

    try:
        data = await chamar_groq(mensagens, com_ferramentas=True)
        if "choices" not in data:
            erro_groq = data.get("error", data)
            return {"content": f"Erro retornado pelo Groq: {erro_groq}"}

        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls")

        # Se o modelo pediu para usar uma ferramenta (buscar na web)
        if tool_calls:
            mensagens.append(msg)  # adiciona a decisao do modelo ao historico
            for call in tool_calls:
                nome_funcao = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                if nome_funcao == "buscar_na_web":
                    resultado = await buscar_na_web(args.get("query", ""))
                else:
                    resultado = f"Ferramenta desconhecida: {nome_funcao}"

                mensagens.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": resultado,
                })

            # Segunda chamada: agora com os resultados da busca, pede a resposta final
            data_final = await chamar_groq(mensagens, com_ferramentas=False)
            if "choices" not in data_final:
                erro_groq = data_final.get("error", data_final)
                return {"content": f"Erro retornado pelo Groq: {erro_groq}"}
            texto_final = data_final["choices"][0]["message"]["content"]
            return {"content": texto_final}

        texto = msg.get("content", "Sem resposta.")
        return {"content": texto}
    except Exception as e:
        return {"content": f"Erro ao consultar IA: {str(e)[:150]}"}
