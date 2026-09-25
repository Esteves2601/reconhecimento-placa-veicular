# Motor.py
# -*- coding: utf-8 -*-
"""Motor de deteccao da interface visual.

Usa o sistema ORIGINAL apenas como biblioteca (sem modifica-lo):
  Main.desenharRetanguloVermelhoAoRedorDaPlaca,
  Main.escreverCaracteresDaPlacaNaImagem,
  DetectarPlacas.DetectarPlacasInScene,
  DetectarCaracteres.DetectarCaracteresNasPlacas (.loadKNN...).

Tudo que e validacao de regiao, correcao, consenso e leitura neural
mora NESTE arquivo.
"""
import os
import re
import threading

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import Main as legacy
import DetectarPlacas
import DetectarCaracteres

# --------------------------------------------------------------------------
# Regioes e formatos (carros e motos compartilham o padrao em cada pais)
# --------------------------------------------------------------------------
REGIOES = {
    "brasil": [
        ("antigo", r"^[A-Z]{3}[0-9]{4}$", "LLLDDDD"),
        ("mercosul", r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$", "LLLDLDD"),
    ],
    "argentina": [
        ("antigo", r"^[A-Z]{3}[0-9]{3}$", "LLLDDD"),
        ("mercosul", r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$", "LLDDDLL"),
    ],
    "uruguai": [
        ("mercosul", r"^[A-Z]{3}[0-9]{4}$", "LLLDDDD"),
    ],
    "paraguai": [
        ("antigo", r"^[A-Z]{4}[0-9]{3}$", "LLLLDDD"),
        ("mercosul", r"^[A-Z]{3}[0-9]{3}$", "LLLDDD"),
    ],
    "internacional": [
        ("generica", r"^[A-Z0-9]{5,8}$", None),
    ],
}
ORDEM_AUTO = ("brasil", "argentina", "uruguai", "paraguai")

DIGITO_PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "4": "A",
                     "5": "S", "6": "G", "8": "B"}
LETRA_PARA_DIGITO = {"O": "0", "I": "1", "J": "1", "A": "4", "S": "5",
                     "B": "8", "G": "6", "Z": "2", "D": "0", "Q": "0"}

TROCAS = {"1": "I", "I": "1", "0": "O", "O": "0", "5": "S", "S": "5",
          "8": "B", "B": "8", "6": "G", "G": "6", "2": "Z", "Z": "2",
          "4": "A", "A": "4"}

PADRAO_GENERICO = re.compile(r"^[A-Z0-9]{4,8}$")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

ORCAMENTO_S = 14.0
LARGURA_MINIMA = 400


def normalizar(texto):
    return (texto or "").upper().replace("-", "").replace(" ", "").strip()


def _alvos(regiao):
    regiao = (regiao or "auto").lower()
    nomes = (regiao,) if regiao in REGIOES else ORDEM_AUTO
    for nome in nomes:
        for padrao, regex, slots in REGIOES[nome]:
            yield nome, padrao, re.compile(regex), slots


def valida(texto, regiao="auto"):
    texto = normalizar(texto)
    return any(rx.match(texto) for _, _, rx, _ in _alvos(regiao))


def _corrigir_slots(texto, espera_letra):
    saida = []
    for c, quer_letra in zip(texto, espera_letra):
        if quer_letra and c.isdigit():
            saida.append(DIGITO_PARA_LETRA.get(c, c))
        elif not quer_letra and c.isalpha():
            saida.append(LETRA_PARA_DIGITO.get(c, c))
        else:
            saida.append(c)
    return "".join(saida)


def interpretar(texto, regiao="auto"):
    """Retorna (placa, corrigida, rotulo). Ex.: ("ABC1D23", False,
    "Brasil · mercosul")."""
    texto = normalizar(texto)
    if not texto.isalnum():
        return "", False, ""
    alvos = list(_alvos(regiao))
    for nome, padrao, rx, slots in alvos:  # passada 1: acerto direto
        if slots is not None and len(texto) != len(slots):
            continue
        if rx.match(texto):
            return texto, False, f"{nome.capitalize()} · {padrao}"
    melhor, melhor_n, melhor_rot = "", 99, ""  # passada 2: menos trocas
    for nome, padrao, rx, slots in alvos:
        if slots is None or len(texto) != len(slots):
            continue
        tent = _corrigir_slots(texto, [s == "L" for s in slots])
        if rx.match(tent):
            n = sum(1 for a, b in zip(texto, tent) if a != b)
            if n < melhor_n:
                melhor, melhor_n = tent, n
                melhor_rot = f"{nome.capitalize()} · {padrao}"
    if melhor:
        return melhor, True, melhor_rot
    return "", False, ""


def aceitar_generica(texto, confianca=0.0, votos=0, corroborada=False):
    texto = normalizar(texto)
    if not PADRAO_GENERICO.match(texto or ""):
        return "", ""
    if len(texto) <= 4:
        forte = ((confianca or 0.0) >= 0.8 or votos >= 4)
    elif corroborada:
        forte = ((confianca or 0.0) >= 0.4 or votos >= 2)
    else:
        forte = ((confianca or 0.0) >= 0.6 or votos >= 3)
    if forte:
        return texto, "Internacional · genérica"
    return "", ""


def trocas(texto):
    texto = normalizar(texto)
    return [texto[:i] + TROCAS[c] + texto[i + 1:]
            for i, c in enumerate(texto) if c in TROCAS]


# --------------------------------------------------------------------------
# Leitura neural (EasyOCR) — opcional, com trava anti-download
# --------------------------------------------------------------------------
_leitor = None
_trava = threading.Lock()
_MODELOS = ("craft_mlt_25k.pth", "latin_g2.pth")
VALIDOS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
KW = dict(allowlist=VALIDOS, contrast_ths=0.15, adjust_contrast=0.7,
          text_threshold=0.6, low_text=0.35, link_threshold=0.3,
          mag_ratio=1.0)


def _pasta_modelos():
    return os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")


def ia_disponivel():
    return all(
        os.path.isfile(os.path.join(_pasta_modelos(), m))
        and os.path.getsize(os.path.join(_pasta_modelos(), m)) >= 10 * 1024 * 1024
        for m in _MODELOS)


def _obter_leitor():
    global _leitor
    if _leitor is None or callable(_leitor):
        with _trava:
            if _leitor is None or callable(_leitor):
                import easyocr
                _leitor = easyocr.Reader(["pt", "en"], gpu=False)
    return _leitor


def preparar_ia_async(callback=None):
    def _alvo():
        if not ia_disponivel():
            if callback:
                callback(False)
            return
        try:
            _obter_leitor()
            if callback:
                callback(True)
        except Exception:
            if callback:
                callback(False)
    threading.Thread(target=_alvo, daemon=True).start()


def _altura(caixa):
    try:
        ys = [p[1] for p in caixa]
        return float(max(ys) - min(ys))
    except Exception:
        return 0.0


def _limpar(texto):
    return "".join(c for c in texto.upper() if c in VALIDOS)


def _preparar_crop(crop):
    h, w = crop.shape[:2]
    up = crop
    if w > 480:
        up = cv2.resize(crop, (480, int(h * 480 / w)),
                        interpolation=cv2.INTER_AREA)
        h, w = up.shape[:2]
    if w < 400:
        up = cv2.resize(up, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    return up


def ler_recorte(crop, aceitar=None, ate_quando=None, rapido=False):
    """[(texto, conf, altura)] por relevancia. Sem modelos: excecao."""
    import time as _tempo
    if crop is None:
        return []
    if not ia_disponivel():
        raise RuntimeError("Modelos da IA ausentes.")
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return []
    up = _preparar_crop(crop)
    variantes = [cv2.cvtColor(up, cv2.COLOR_BGR2RGB)]
    try:
        cinza = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        h2, w2 = cinza.shape[:2]
        faixa = cinza[int(h2 * 0.18):int(h2 * 0.86), :]
        if faixa.shape[0] > 10:
            variantes.append(cv2.cvtColor(faixa, cv2.COLOR_GRAY2RGB))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        variantes.append(cv2.cvtColor(clahe.apply(cinza), cv2.COLOR_GRAY2RGB))
        _, binaria = cv2.threshold(cinza, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variantes.append(cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB))
    except Exception:
        pass
    leitor = _obter_leitor()
    saidas, vistas = [], set()
    for rgb in variantes:
        if ate_quando is not None and _tempo.perf_counter() > ate_quando:
            break
        try:
            res = leitor.readtext(rgb, **KW)
        except Exception:
            continue
        for caixa, texto, conf in res:
            limpo = _limpar(texto)
            if limpo and limpo not in vistas:
                vistas.add(limpo)
                saidas.append((limpo, float(conf), _altura(caixa)))
                if aceitar is not None and aceitar(limpo):
                    return _ordenar(saidas)
        if rapido:
            break
    return _ordenar(saidas)


def _ordenar(saidas):
    saidas.sort(key=lambda t: (t[2] * (0.5 + t[1]), t[1]), reverse=True)
    return saidas


FONTES_RENDER = ["DejaVuSansCondensed-Bold.ttf", "arialbd.ttf",
                 "bahnschrift.ttf"]


def prova_render(crop_bgr, texto):
    """Correlacao do recorte com o texto renderizado (3 fontes, max).

    Retorna NCC maximo (maior = mais parecido). Barato (~30ms/fonte).
    """
    texto = (texto or "").strip()
    if not texto or crop_bgr is None:
        return -1.0
    try:
        h, w = crop_bgr.shape[:2]
        cinza = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return -1.0
    claro = float(np.mean(cinza)) > 127.0
    cor_fundo, cor_texto = (255, 0) if claro else (0, 255)
    melhor = -1.0
    for nome_fonte in FONTES_RENDER:
        try:
            img = Image.new("L", (w, h), cor_fundo)
            d = ImageDraw.Draw(img)
            fonte = None
            for tam in range(8, 300, 4):
                try:
                    f = ImageFont.truetype(nome_fonte, tam)
                except Exception:
                    break
                bb = d.textbbox((0, 0), texto, font=f)
                if bb[3] - bb[1] > h * 0.72 or bb[2] - bb[0] > w * 0.92:
                    break
                fonte = f
            if fonte is None:
                continue
            bb = d.textbbox((0, 0), texto, font=fonte)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            d.text(((w - tw) / 2 - bb[0], (h - th) / 2 - bb[1]), texto,
                   font=fonte, fill=cor_texto)
            rend = np.asarray(img, dtype=np.float32)
            tw2, th2 = max(8, int(w * 0.92)), max(8, int(h * 0.92))
            tpl = cv2.resize(rend, (tw2, th2))
            res = cv2.matchTemplate(cinza.astype(np.float32), tpl,
                                    cv2.TM_CCOEFF_NORMED)
            valor = float(res.max())
            if valor > melhor:
                melhor = valor
        except Exception:
            continue
    return melhor


def treinar():
    """Treina o KNN do sistema original. Retorna True/False."""
    try:
        return bool(DetectarCaracteres.loadKNNDataAndTrainKNN())
    except Exception:
        return False


# --------------------------------------------------------------------------
# Busca Google (Key + CX) e download de URL
# --------------------------------------------------------------------------
def buscar_google(modelo, api_key, cx, max_resultados=8):
    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"q": f"{modelo.strip()} carro", "searchType": "image",
                "num": max(1, min(max_resultados, 10)),
                "key": api_key, "cx": cx},
        timeout=25)
    resp.raise_for_status()
    saidas, vistas = [], set()
    for it in resp.json().get("items", []):
        link = it.get("link")
        if not link or link in vistas:
            continue
        vistas.add(link)
        saidas.append({"link": link, "titulo": it.get("title", "")[:90]})
        if len(saidas) >= max_resultados:
            break
    return saidas


def baixar_imagem(url, timeout=20, largura_max=1280):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    if not r.content:
        raise ValueError("Resposta vazia.")
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Conteúdo não é imagem.")
    h, w = img.shape[:2]
    if w > largura_max:
        img = cv2.resize(img, (largura_max, int(h * largura_max / w)))
    return img


# --------------------------------------------------------------------------
# Deteccao: coleta votos (IA + KNN original) e decide por consenso
# --------------------------------------------------------------------------
def _variantes_cena(img):
    yield img
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yield cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                           cv2.COLOR_LAB2BGR)
    except Exception:
        pass
    try:
        yield cv2.convertScaleAbs(img, alpha=1.3, beta=12)
    except Exception:
        pass


def _escore(pl):
    try:
        (_cx, _cy), (w, h), _a = pl.rrLocationOfPlacaInScene
        r = float(max(w, h)) / max(1.0, float(min(w, h)))
        prop = min(abs(r - 3.0) / 3.0, abs(r - 0.85) / 0.85)
        area = float(w) * float(h)
        return prop + max(0.0, (8000.0 - area) / 8000.0) * 2.0
    except Exception:
        return 9.0


def _ranking(votos):
    ordem, cont, fontes = [], {}, {}
    for texto, fonte in votos:
        if texto not in cont:
            ordem.append(texto)
            cont[texto] = 0
            fontes[texto] = fonte
        cont[texto] += 1
    rk = [(t, fontes[t], cont[t]) for t in ordem]
    rk.sort(key=lambda r: -r[2])
    return rk


def _consenso(textos):
    grupos = {}
    for t in textos:
        grupos.setdefault(len(t), []).append(t)
    if not grupos:
        return "", 0
    tam = max(sorted(grupos), key=lambda l: (len(grupos[l]), l))
    cand = grupos[tam]
    letras = []
    for i in range(tam):
        cont, ordem = {}, []
        for t in cand:
            if t[i] not in cont:
                ordem.append(t[i])
                cont[t[i]] = 0
            cont[t[i]] += 1
        letras.append(max(ordem, key=lambda c: cont[c]))
    return "".join(letras), len(cand)


def _acordo(textos, consenso):
    grupo = [t for t in textos if len(t) == len(consenso)]
    if len(grupo) < 2:
        return False
    ok = sum(1 for i, ch in enumerate(consenso)
             if sum(1 for t in grupo if t[i] == ch) >= 2)
    return ok * 10 >= len(consenso) * 6


def detectar(img_bgr, regiao="auto", debug=False):
    """Retorna dict(placa, anotada, recorte, analise, rotulo, fonte,
    detalhe). Teto de ~14 s."""
    import time as _tempo
    inicio = _tempo.perf_counter()

    def dbg(m):
        if debug:
            print(f"[{_tempo.perf_counter() - inicio:5.1f}s] {m}", flush=True)

    def estouro():
        return _tempo.perf_counter() - inicio > ORCAMENTO_S

    def aceitar(t):
        return valida(normalizar(t), regiao)

    def finalizar(cand, placa, rotulo, fonte, detalhe=""):
        cand.strCaracteres = placa
        anotada = img_bgr.copy()
        try:
            legacy.desenharRetanguloVermelhoAoRedorDaPlaca(anotada, cand)
            legacy.escreverCaracteresDaPlacaNaImagem(anotada, cand)
        except Exception:
            pass
        analise = cand.imgThreshold
        if analise is None and cand.imgPlaca is not None:
            try:
                cz = cv2.cvtColor(cand.imgPlaca, cv2.COLOR_BGR2GRAY)
                _, analise = cv2.threshold(cz, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                analise = None
        return {"placa": placa, "anotada": anotada,
                "recorte": cand.imgPlaca, "analise": analise,
                "rotulo": rotulo, "fonte": fonte, "detalhe": detalhe}

    def vazio():
        return {"placa": "", "anotada": img_bgr, "recorte": None,
                "analise": None, "rotulo": "", "fonte": "", "detalhe": ""}

    votos, melhor = [], None
    melhor_ia = {"texto": "", "conf": 0.0}
    conf_ia, score_ia = {}, {}
    tem_ia = ia_disponivel()
    limite = inicio + ORCAMENTO_S - 1.0

    def tentar_trocas(norm, cand, fonte):
        for var in trocas(norm):
            pv, _c, rv = interpretar(var, regiao)
            if pv:
                return finalizar(cand, pv, rv, fonte, "variação 1↔I/0↔O")
        return None

    def considerar(norm, conf, cand, fonte, alt=0.0):
        if not norm:
            return None
        e_ia = fonte.startswith("ia")
        votos.append((norm, fonte))
        if e_ia:
            if conf > conf_ia.get(norm, 0.0):
                conf_ia[norm] = conf
            pts = alt * (0.5 + conf) if alt > 0 else conf
            if pts > score_ia.get(norm, -1.0):
                score_ia[norm] = pts
            if conf > melhor_ia["conf"]:
                melhor_ia["texto"], melhor_ia["conf"] = norm, conf
        pv, _c, rv = interpretar(norm, regiao)
        if pv:
            return finalizar(cand, pv, rv, fonte, f"confiança {conf:.2f}")
        if e_ia:
            return tentar_trocas(norm, cand, fonte)
        return None

    for variante in _variantes_cena(img_bgr):
        try:
            regioes = DetectarPlacas.DetectarPlacasInScene(variante.copy())
        except Exception:
            continue
        if not regioes:
            continue
        regioes.sort(key=_escore)
        for cand in regioes[:2]:
            if estouro():
                break
            if melhor is None and cand.imgPlaca is not None:
                melhor = cand
            try:
                _h, _w = cand.imgPlaca.shape[:2]
            except Exception:
                continue
            if _h < 14 or _w < 50:
                continue
            mini = _h * _w < 5000
            try:
                n0 = len(votos)
                for ti, cf, al in ler_recorte(cand.imgPlaca, aceitar,
                                              limite, rapido=mini):
                    fim = considerar(normalizar(ti), cf, cand, "ia", al)
                    if fim:
                        return fim
                dbg(f"rec {_w}x{_h}: +{len(votos) - n0} votos")
            except Exception as e:
                dbg(f"IA erro {type(e).__name__}")
            if estouro():
                break
        if estouro():
            break
        try:
            placas = DetectarCaracteres.DetectarCaracteresNasPlacas(
                regioes[:8])
        except Exception:
            continue
        placas.sort(key=lambda p: len(p.strCaracteres), reverse=True)
        for cand in placas[:3]:
            if estouro():
                break
            if melhor is None and cand.imgPlaca is not None:
                melhor = cand
            bruta = normalizar(cand.strCaracteres)
            if bruta:
                votos.append((bruta, "knn"))
                pv, _c, rv = interpretar(bruta, regiao)
                if pv:
                    return finalizar(cand, pv, rv, "knn", "leitura direta")
                achou = tentar_trocas(bruta, cand, "knn")
                if achou:
                    return achou
            if estouro():
                break
        if estouro():
            break

    for texto, fonte, n in _ranking(votos):
        pv, _c, rv = interpretar(texto, regiao)
        if pv:
            return finalizar(melhor, pv, rv, fonte, f"{n} leitura(s)")
    longos = [t for t, _ in votos if len(t) >= 4]
    consenso, n_grupo = _consenso(longos)
    if consenso and _acordo(longos, consenso):
        pv, _c, rv = interpretar(consenso, regiao)
        if pv:
            votos.append((consenso, "consenso"))
            return finalizar(melhor, pv, rv, "consenso", "combinação")
        gv, grv = aceitar_generica(consenso, 0.0, n_grupo)
        if gv:
            votos.append((consenso, "consenso"))
            return finalizar(melhor, gv, grv, "consenso",
                             f"{n_grupo} no grupo")
    # 2b) prova por render: correlaciona finalistas renderizados com
    # os pixels. Vence com margem clara (>= 0.015).
    if (melhor is not None and melhor.imgPlaca is not None
            and _tempo.perf_counter() < inicio + ORCAMENTO_S + 1.0):
        cands_r, vistos_r = [], set()
        for t in sorted(score_ia, key=lambda x: score_ia[x], reverse=True):
            if t and t not in vistos_r:
                vistos_r.add(t)
                cands_r.append(t)
            if len(cands_r) >= 8:
                break
        for t, f in votos:
            if f == "knn" and t not in vistos_r:
                vistos_r.add(t)
                cands_r.append(t)
            if len(cands_r) >= 10:
                break
        notas = []
        for t in cands_r:
            if _tempo.perf_counter() > inicio + ORCAMENTO_S + 1.0:
                break
            try:
                notas.append((prova_render(melhor.imgPlaca, t), t))
            except Exception:
                continue
        notas.sort(reverse=True)
        if len(notas) >= 2 and notas[0][0] - notas[1][0] >= 0.015:
            top = notas[0][1]
            pv, _c, rv = interpretar(top, regiao)
            if pv:
                return finalizar(melhor, pv, rv, "render",
                                 f"correlação {notas[0][0]:.3f}")
            if re.match(r"^[A-Z0-9]{4,8}$", top):
                return finalizar(melhor, top,
                                 "Internacional · genérica (render)",
                                 "render",
                                 f"correlação {notas[0][0]:.3f}")
    if melhor is None:
        return vazio()
    knn_set = {t for t, f in votos if f == "knn"}
    for texto in sorted(score_ia, key=lambda t: score_ia[t], reverse=True):
        gv, grv = aceitar_generica(texto, conf_ia[texto])
        if gv:
            return finalizar(melhor, gv, grv, "ia", "relevância")
        for var in trocas(texto):
            if var in conf_ia or var in knn_set:
                gv, grv = aceitar_generica(var, conf_ia.get(var, 0.5),
                                           corroborada=True)
                if gv:
                    return finalizar(melhor, gv, grv, "ia",
                                     "variação corroborada")
    if melhor_ia["texto"]:
        return {"placa": melhor_ia["texto"], "anotada": img_bgr,
                "recorte": melhor.imgPlaca, "analise": melhor.imgThreshold,
                "rotulo": "", "fonte": "", "detalhe": "melhor leitura da IA"}
    if not tem_ia:
        rk = _ranking(votos)
        if rk:
            return {"placa": rk[0][0], "anotada": img_bgr,
                    "recorte": melhor.imgPlaca,
                    "analise": melhor.imgThreshold,
                    "rotulo": "", "fonte": "", "detalhe": "mais votada"}
    return vazio()


LIMIAR_NORMAL = 55
LIMIAR_RIGOROSO = 80


def _diff(a, b):
    if not a or not b or len(a) != len(b):
        return 999
    return sum(1 for x, y in zip(a, b) if x != y)


def original_rapido(img):
    """So o pipeline original (KNN), sem validacao. Retorna texto ou ''."""
    try:
        ps = DetectarPlacas.DetectarPlacasInScene(img.copy())
        ps = DetectarCaracteres.DetectarCaracteresNasPlacas(ps)
        if not ps:
            return ""
        ps.sort(key=lambda p: len(p.strCaracteres), reverse=True)
        return normalizar(ps[0].strCaracteres or "")
    except Exception:
        return ""


def fundir(res, orig, rigoroso=False):
    """Porteira de certeza 0-100. So exibe se >= limiar."""
    placa = res.get("placa") or ""
    rotulo = res.get("rotulo") or ""
    pontos, motivos = 0, []
    if rotulo.split(" ·")[0] in ("Brasil", "Argentina", "Uruguai",
                                 "Paraguai"):
        pontos += 45
        motivos.append("padrão regional")
    elif rotulo:
        pontos += 30
        motivos.append("padrão aceito")
    if placa and orig:
        if placa == orig:
            pontos += 35
            motivos.append("original confirma exato")
        elif _diff(placa, orig) <= 2:
            pontos += 15
            motivos.append("original quase igual")
    pontos += 10
    motivos.append("leitura concluída")
    limiar = LIMIAR_RIGOROSO if rigoroso else LIMIAR_NORMAL
    nivel = ("alta" if pontos >= 75
             else "média" if pontos >= limiar else "baixa")
    saida = dict(res)
    saida.update({"placa": placa if (placa and pontos >= limiar) else "",
                  "certeza": pontos, "nivel": nivel, "limiar": limiar,
                  "motivos": motivos, "original": orig})
    return saida


def combinar(img, regiao="auto", rigoroso=False):
    """Roda original (paralelo) + motor completo e funde com certeza."""
    orig = {}

    def _o():
        orig["t"] = original_rapido(img)

    th = threading.Thread(target=_o, daemon=True)
    th.start()
    res = detectar(img, regiao)
    th.join(timeout=10)
    return fundir(res, orig.get("t", ""), rigoroso)
