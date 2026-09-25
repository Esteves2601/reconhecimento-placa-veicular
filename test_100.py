# test_100.py — 100 testes: 16 fotos x5 repeticoes + 4 variacoes x5.
# Modos: motor (sistema atual) | original (so pipeline KNN original).
# Uso: python test_100.py <motor|original> <fatia 0-3>   (25 casos por fatia)
# Acrescenta em test_100_resultados.jsonl (recria se fatia==0 e modo novo).
import json
import os
import sys
import time
import cv2
import numpy as np

import Motor
import DetectarPlacas
import DetectarCaracteres

GABARITO = {
    1: "MCLRNF1", 2: "LOLWATT", 3: "RIPLS1", 4: "NICESKY",
    5: "NVSBLE", 6: "NYSJ", 7: "ANBYOND", 8: "1ZM961",
    9: "PNYEXPS", 10: "ZOOMN65", 11: "HOR5SH1T", 12: "FALLYOU",
    13: "EZEY", 14: "NITESKY", 15: "U8NTBAD", 16: "GAY247",
}

BASE = os.path.dirname(os.path.abspath(__file__))


def saida_para(modo):
    return os.path.join(BASE, f"test_100_{modo}.jsonl")


def carregar(n):
    return cv2.imdecode(
        np.fromfile(os.path.join(BASE, "imagens", f"{n}.png"),
                    dtype=np.uint8),
        cv2.IMREAD_COLOR)


def transformar(img, nome):
    h, w = img.shape[:2]
    if nome == "rot+4":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 4, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if nome == "rot-4":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), -4, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if nome == "clara":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=35)
    if nome == "escura":
        return cv2.convertScaleAbs(img, alpha=0.75, beta=0)
    raise ValueError(nome)


def detectar_original(img):
    """Pipeline 100% original: so KNN, sem validacao (como o Main.py)."""
    ps = DetectarPlacas.DetectarPlacasInScene(img.copy())
    ps = DetectarCaracteres.DetectarCaracteresNasPlacas(ps)
    if not ps:
        return ""
    ps.sort(key=lambda p: len(p.strCaracteres), reverse=True)
    return ps[0].strCaracteres or ""


def montar_casos():
    casos = []
    for n, esp in GABARITO.items():
        for rep in range(5):
            casos.append((f"{n}.png", carregar(n), esp, rep))
    for n, tr in [(12, "rot+4"), (10, "rot-4"), (15, "clara"),
                  (2, "escura")]:
        for rep in range(5):
            img = transformar(carregar(n), tr)
            casos.append((f"{n}.png[{tr}]", img, GABARITO[n], rep))
    return casos  # 100 no total


def main():
    modo, fatia = sys.argv[1], int(sys.argv[2])
    assert modo in ("motor", "original") and 0 <= fatia <= 3
    assert Motor.treinar(), "treino KNN falhou"
    casos = montar_casos()
    assert len(casos) == 100
    parte = casos[fatia * 25:(fatia + 1) * 25]
    linhas = []
    for nome, img, esperado, rep in parte:
        t0 = time.time()
        fonte, detalhe = "", ""
        try:
            if modo == "motor":
                r = Motor.detectar(img.copy(), "auto")
                obtido = r["placa"] or ""
                fonte, detalhe = r["fonte"], r["detalhe"]
            else:
                obtido = detectar_original(img.copy())
        except Exception as e:
            obtido = f"ERRO:{type(e).__name__}"
        dt = time.time() - t0
        linhas.append({"modo": modo, "caso": nome, "rep": rep,
                       "esperado": esperado, "obtido": obtido,
                       "acertou": obtido == esperado, "tempo_s": round(dt, 1)})
        print(f"[{modo} {fatia}] {nome} r{rep}: "
              f"{esperado!r} x {obtido!r} via={fonte} {detalhe} {dt:.1f}s",
              flush=True)
    modo_arquivo = "w" if fatia == 0 else "a"
    with open(saida_para(modo), modo_arquivo, encoding="utf-8") as f:
        for linha in linhas:
            f.write(json.dumps(linha) + "\n")
    print(f"fatia {fatia}/{modo} gravada", flush=True)


if __name__ == "__main__":
    main()
