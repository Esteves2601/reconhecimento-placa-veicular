# test_suite20.py — 20 testes rotulados (16 fotos + 4 variações).
import sys
import time
import cv2
import numpy as np
import Motor

GABARITO = {
    1: "MCLRNF1", 2: "LOLWATT", 3: "RIPLS1", 4: "NICESKY",
    5: "NVSBLE", 6: "NYSJ", 7: "ANBYOND", 8: "1ZM961",
    9: "PNYEXPS", 10: "ZOOMN65", 11: "HOR5SH1T", 12: "FALLYOU",
    13: "EZEY", 14: "NITESKY", 15: "U8NTBAD", 16: "GAY247",
}

TETO_S = 15.0


def carregar(n):
    return cv2.imdecode(
        np.fromfile(f"imagens/{n}.png", dtype=np.uint8), cv2.IMREAD_COLOR)


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


def main():
    assert Motor.treinar(), "treino KNN falhou"
    casos = [(f"{n}.png", carregar(n), esp) for n, esp in GABARITO.items()]
    for n, tr in [(12, "rot+4"), (10, "rot-4"), (15, "clara"),
                  (2, "escura")]:
        casos.append((f"{n}.png[{tr}]", transformar(carregar(n), tr),
                      GABARITO[n]))
    assert len(casos) == 20
    ok, falhas, tempos = 0, [], []
    for nome, img, esperado in casos:
        t0 = time.time()
        r = Motor.detectar(img, "auto")
        dt = time.time() - t0
        tempos.append(dt)
        certo = (r["placa"] == esperado)
        lento = dt > TETO_S
        print(f"{'OK ' if (certo and not lento) else 'FAIL'} {nome}: "
              f"esperado={esperado!r} obtido={r['placa']!r} "
              f"via={r['fonte']} {dt:.1f}s", flush=True)
        if certo and not lento:
            ok += 1
        else:
            falhas.append(nome)
    print(f"\nPLACAR: {ok}/20  tempo medio: "
          f"{sum(tempos) / len(tempos):.1f}s", flush=True)
    if falhas:
        print("falhas:", falhas, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
