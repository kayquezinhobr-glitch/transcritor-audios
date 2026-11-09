"""
Transcritor de Áudios
Upload de áudios -> transcrição (Whisper via Groq) -> análise pelo Claude (opcional).

Chaves (em Settings > Secrets no Streamlit Cloud):
    GROQ_API_KEY      = "gsk_..."        # obrigatória (transcrição)
    ANTHROPIC_API_KEY = "sk-ant-..."     # opcional (análise pelo Claude)
    APP_PASSWORD      = "sua-senha"      # opcional, mas recomendada
    CLAUDE_MODEL      = "claude-sonnet-5-5"  # opcional
"""

import csv
import hmac
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import anthropic
import groq
import imageio_ffmpeg
import streamlit as st

st.set_page_config(page_title="Transcritor de Áudios", page_icon="🎙️", layout="centered")

# ---------------------------------------------------------------- configuração

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # ffmpeg vem junto do pacote, sem instalar nada
MODELO_TRANSCRICAO = "whisper-large-v3-turbo"
DURACAO_PEDACO_SEG = 20 * 60  # áudios longos são divididos em partes de 20 min

FORMATOS = [
    "mp3", "m4a", "wav", "ogg", "opus", "oga", "webm", "mp4", "mpeg", "mpga",
    "flac", "aac", "amr", "3gp", "wma", "mov", "mkv",
]

IDIOMAS = {
    "Português": "pt",
    "Detectar automaticamente": None,
    "Inglês": "en",
    "Espanhol": "es",
}

ANALISES = {
    "Resumo": (
        "Faça um resumo claro do áudio: comece com 2 a 3 frases sobre o assunto "
        "principal e depois liste os pontos mais importantes em tópicos."
    ),
    "Ata de reunião (decisões e tarefas)": (
        "Monte uma ata com as seções: Assuntos discutidos, Decisões tomadas, "
        "Tarefas (com responsável e prazo quando forem citados) e Pendências/dúvidas."
    ),
    "Tópicos principais": (
        "Liste os principais tópicos abordados, na ordem em que aparecem, com uma "
        "frase explicando cada um."
    ),
    "Revisar e organizar o texto": (
        "Reescreva a transcrição corrigindo pontuação, erros evidentes de "
        "reconhecimento e vícios de fala (né, tipo, é...), dividindo em parágrafos. "
        "Não resuma e não mude o sentido de nada."
    ),
    "Pergunta personalizada": None,
}

SISTEMA_CLAUDE = (
    "Você analisa transcrições automáticas de áudio. Responda em português do Brasil, "
    "em Markdown. Baseie-se somente no que está na transcrição; se algo estiver "
    "ambíguo ou parecer erro de transcrição, diga isso em vez de inventar."
)


def segredo(nome, padrao=None):
    try:
        return st.secrets[nome]
    except Exception:
        return os.environ.get(nome, padrao)


# ---------------------------------------------------------------- senha

def acesso_liberado() -> bool:
    senha = segredo("APP_PASSWORD")
    if not senha or st.session_state.get("autenticado"):
        return True
    st.title("🎙️ Transcritor de Áudios")
    with st.form("login"):
        digitada = st.text_input("Senha", type="password")
        entrar = st.form_submit_button("Entrar")
    if entrar:
        if hmac.compare_digest(digitada.encode(), str(senha).encode()):
            st.session_state.autenticado = True
            st.rerun()
        st.error("Senha incorreta.")
    return False


# ---------------------------------------------------------------- áudio

def preparar_audio(dados: bytes, nome: str, pasta: Path):
    """Converte para mp3 mono 16 kHz (leve) e divide em partes.
    Retorna lista de (caminho, início_em_segundos)."""
    origem = pasta / ("original" + (Path(nome).suffix.lower() or ".bin"))
    origem.write_bytes(dados)
    lista = pasta / "partes.csv"
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(origem), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "32k",
        "-f", "segment", "-segment_time", str(DURACAO_PEDACO_SEG),
        "-segment_list", str(lista), "-segment_list_type", "csv",
        "-reset_timestamps", "1",
        str(pasta / "parte_%03d.mp3"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not lista.exists():
        raise RuntimeError("Não consegui ler esse arquivo de áudio. " + r.stderr[-300:])
    partes = []
    with open(lista, newline="") as f:
        for linha in csv.reader(f):
            if linha:
                partes.append((pasta / linha[0], float(linha[1])))
    if not partes:
        raise RuntimeError("O arquivo não tem áudio.")
    return partes


def transcrever_parte(cliente, caminho: Path, idioma):
    kwargs = dict(
        file=(caminho.name, caminho.read_bytes()),
        model=MODELO_TRANSCRICAO,
        response_format="verbose_json",
        temperature=0.0,
    )
    if idioma:
        kwargs["language"] = idioma
    for tentativa in range(5):
        try:
            resp = cliente.audio.transcriptions.create(**kwargs)
            dados = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            return dados.get("text", "") or "", dados.get("segments") or []
        except groq.RateLimitError:
            if tentativa == 4:
                raise
            time.sleep(15 * (tentativa + 1))
        except groq.APIConnectionError:
            if tentativa == 4:
                raise
            time.sleep(3)


def hhmmss(seg: float) -> str:
    seg = int(seg)
    return f"{seg // 3600:02d}:{seg % 3600 // 60:02d}:{seg % 60:02d}"


def transcrever_arquivo(cliente, dados: bytes, nome: str, idioma, aviso):
    with tempfile.TemporaryDirectory() as tmp:
        aviso.write("Preparando o áudio…")
        partes = preparar_audio(dados, nome, Path(tmp))
        textos, linhas_tempo = [], []
        for i, (caminho, inicio) in enumerate(partes, 1):
            aviso.write(f"Transcrevendo parte {i} de {len(partes)}…")
            texto, segmentos = transcrever_parte(cliente, caminho, idioma)
            textos.append(texto.strip())
            for s in segmentos:
                ini = inicio + float(s.get("start", 0))
                linhas_tempo.append(f"[{hhmmss(ini)}] {str(s.get('text', '')).strip()}")
            if not segmentos and texto.strip():
                linhas_tempo.append(f"[{hhmmss(inicio)}] {texto.strip()}")
    return {"texto": "\n\n".join(t for t in textos if t), "com_tempo": "\n".join(linhas_tempo)}


# ---------------------------------------------------------------- Claude

def analisar_com_claude(transcricao: str, instrucao: str):
    cliente = anthropic.Anthropic(api_key=segredo("ANTHROPIC_API_KEY"))
    modelo = segredo("CLAUDE_MODEL", "claude-sonnet-5-5")
    with cliente.messages.stream(
        model=modelo,
        max_tokens=16000,
        system=SISTEMA_CLAUDE,
        messages=[{
            "role": "user",
            "content": f"<transcricao>\n{transcricao}\n</transcricao>\n\n{instrucao}",
        }],
    ) as stream:
        for pedaco in stream.text_stream:
            yield pedaco


# ---------------------------------------------------------------- interface

def main():
    if not acesso_liberado():
        return

    st.title("🎙️ Transcritor de Áudios")
    st.caption("Envie áudios (WhatsApp, reuniões, aulas, gravações) para transcrever e analisar.")

    chave_groq = segredo("GROQ_API_KEY")
    tem_claude = bool(segredo("ANTHROPIC_API_KEY"))
    if not chave_groq:
        st.error("Falta a chave GROQ_API_KEY nas configurações (Secrets) do app.")
        return

    st.session_state.setdefault("transcricoes", {})
    st.session_state.setdefault("analises", [])

    with st.sidebar:
        st.header("Opções")
        idioma_nome = st.selectbox("Idioma do áudio", list(IDIOMAS))
        mostrar_tempo = st.toggle("Mostrar marcações de tempo", value=False)
        if st.button("Limpar tudo"):
            st.session_state.transcricoes = {}
            st.session_state.analises = []
            st.rerun()

    # 1) Upload e transcrição
    arquivos = st.file_uploader(
        "Escolha um ou mais áudios", type=FORMATOS, accept_multiple_files=True
    )
    if arquivos and st.button("Transcrever", type="primary", use_container_width=True):
        cliente = groq.Groq(api_key=chave_groq)
        for arq in arquivos:
            chave = f"{arq.name} ({arq.size // 1024} KB)"
            if chave in st.session_state.transcricoes:
                continue
            with st.status(f"{arq.name}", expanded=True) as status:
                try:
                    resultado = transcrever_arquivo(
                        cliente, arq.getvalue(), arq.name, IDIOMAS[idioma_nome], status
                    )
                    st.session_state.transcricoes[chave] = resultado
                    status.update(label=f"✅ {arq.name}", state="complete", expanded=False)
                except groq.AuthenticationError:
                    status.update(label=f"❌ {arq.name}", state="error")
                    st.error("A chave da Groq é inválida. Confira GROQ_API_KEY.")
                except groq.RateLimitError:
                    status.update(label=f"❌ {arq.name}", state="error")
                    st.error("Limite de uso da Groq atingido. Tente de novo mais tarde.")
                except Exception as e:  # noqa: BLE001
                    status.update(label=f"❌ {arq.name}", state="error")
                    st.error(f"Erro: {e}")

    transcricoes = st.session_state.transcricoes
    if not transcricoes:
        return

    # 2) Resultado das transcrições
    st.subheader("Transcrições")
    for nome, r in transcricoes.items():
        texto = r["com_tempo"] if mostrar_tempo else r["texto"]
        with st.expander(nome, expanded=len(transcricoes) == 1):
            st.text_area("Texto", texto, height=260, key=f"txt_{nome}_{mostrar_tempo}",
                         label_visibility="collapsed")
            st.download_button(
                "Baixar .txt", texto.encode("utf-8"),
                file_name=f"{Path(nome.split(' (')[0]).stem}_transcricao.txt",
                mime="text/plain", key=f"dl_{nome}_{mostrar_tempo}",
            )

    # 3) Análise com o Claude
    st.subheader("Analisar com o Claude")
    if not tem_claude:
        st.info("Para analisar, adicione ANTHROPIC_API_KEY nas configurações (Secrets) do app.")
        return

    opcoes = list(transcricoes)
    if len(opcoes) > 1:
        opcoes = ["Todas juntas"] + opcoes
    escolha = st.selectbox("Qual transcrição", opcoes)
    tipo = st.selectbox("O que fazer", list(ANALISES))
    instrucao = ANALISES[tipo]
    if instrucao is None:
        instrucao = st.text_area("Sua pergunta ou pedido sobre o áudio",
                                 placeholder="Ex.: Quais valores e datas foram citados?")

    if st.button("Analisar", type="primary", use_container_width=True,
                 disabled=not (instrucao or "").strip()):
        if escolha == "Todas juntas":
            base = "\n\n".join(f"### {n}\n{r['texto']}" for n, r in transcricoes.items())
        else:
            base = transcricoes[escolha]["texto"]
        try:
            resposta = st.write_stream(analisar_com_claude(base, instrucao))
            st.session_state.analises.insert(0, {
                "titulo": f"{tipo} — {escolha}",
                "texto": resposta,
                "hora": datetime.now().strftime("%d/%m %H:%M"),
            })
            st.rerun()
        except anthropic.AuthenticationError:
            st.error("A chave da Anthropic é inválida. Confira ANTHROPIC_API_KEY.")
        except anthropic.APIStatusError as e:
            st.error(f"Erro na API do Claude ({e.status_code}): {e.message}")

    for i, a in enumerate(st.session_state.analises):
        with st.expander(f"{a['titulo']} · {a['hora']}", expanded=i == 0):
            st.markdown(a["texto"])
            st.download_button("Baixar .md", a["texto"].encode("utf-8"),
                               file_name=f"analise_{i + 1}.md", mime="text/markdown",
                               key=f"dla_{i}")


main()
