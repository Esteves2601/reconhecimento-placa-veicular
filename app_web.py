"""Servidor web/PWA do Reconhecimento de Placas (igual app_web.py do detector).

    py app_web.py [--port 5000]

No celular (mesmo Wi-Fi): http://<IP-DO-PC>:5000 -> "Adicionar a tela
inicial". Vira app de tela cheia sem instalar APK (sem aviso de perigo).
"""
import argparse
import io
import socket

import cv2
import numpy as np
from flask import Flask, jsonify, request

import Motor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB
PRONTO = {"knn": False}

HTML = """<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="mobile-web-app-capable" content="yes">
<title>Placas</title>
<style>
body{background:#0b0b0d;color:#f4f4f5;font-family:Segoe UI,Arial;margin:0 auto;max-width:560px;padding:16px}
h1{font-size:20px}h1 span{color:#e10600}
.card{background:#131316;border:1px solid #26262c;border-radius:14px;padding:14px;margin:12px 0}
button{background:#e10600;color:#fff;border:0;border-radius:10px;padding:12px;width:100%;font-size:16px;font-weight:bold}
#placa{font-size:44px;font-weight:bold;text-align:center;font-family:Consolas,monospace}
#info{text-align:center;color:#9d9da8}
img{max-width:100%;border-radius:10px}
</style></head><body>
<h1><span>●</span> RECONHECIMENTO DE PLACAS</h1>
<div class="card"><form id="f">
<input type="file" name="arquivo" accept="image/*" capture="environment" required>
<p><button>DETECTAR PLACA</button></p></form></div>
<div class="card"><div id="placa">— — —</div><div id="info"></div></div>
<div class="card"><img id="prev"></div>
<script>const f=document.getElementById('f');
f.onsubmit=async e=>{e.preventDefault();
document.getElementById('placa').textContent='...';
const r=await fetch('/api/detect',{method:'POST',body:new FormData(f)});
const j=await r.json();
document.getElementById('placa').textContent=j.placa||'NÃO ENCONTRADA';
document.getElementById('info').textContent=
(j.nivel?('✓ '+j.nivel+' '+j.certeza+'%'):'')+(j.rotulo?(' ['+j.rotulo+']'):'');
if(j.imagem){document.getElementById('prev').src='data:image/jpeg;base64,'+j.imagem;}};</script>
</body></html>"""


@app.get("/")
def index():
    return HTML


@app.get("/api/saude")
def saude():
    return jsonify({"status": "ok", "knn": PRONTO["knn"]})


@app.post("/api/detect")
def detect():
    import base64
    arq = request.files.get("arquivo")
    if arq is None:
        return jsonify({"erro": "envie 'arquivo'"}), 400
    img = cv2.imdecode(np.frombuffer(arq.read(), dtype=np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"erro": "imagem inválida"}), 400
    regiao = request.form.get("regiao", "auto")
    res = Motor.combinar(img, regiao, False)
    out = {"placa": res.get("placa") or "", "rotulo": res.get("rotulo") or "",
           "nivel": res.get("nivel") or "", "certeza": res.get("certeza", 0)}
    if res.get("anotada") is not None:
        _, buf = cv2.imencode(".jpg", res["anotada"])
        out["imagem"] = base64.b64encode(bytes(buf)).decode("ascii")
    return jsonify(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    print("Treinando KNN (uma vez)…")
    PRONTO["knn"] = bool(Motor.treinar())
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "127.0.0.1"
    print(f"Abra no celular (mesmo Wi-Fi): http://{ip}:{args.port}")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
