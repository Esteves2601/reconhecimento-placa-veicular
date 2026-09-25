# test_20.py — 20 testes (16 fotos + 4 variacoes), 1 repeticao.
import sys
import time
import cv2
import numpy as np
import Motor
from test_100 import GABARITO, carregar, transformar


def main():
    assert Motor.treinar(), "treino falhou"
    casos = [(f"{n}.png", carregar(n), esp) for n, esp in GABARITO.items()]
    for n, tr in [(12, "rot+4"), (10, "rot-4"), (15, "clara"),
                  (2, "escura")]:
        casos.append((f"{n}.png[{tr}]", transformar(carregar(n), tr),
                      GABARITO[n]))
    assert len(casos) == 20
    ok, tempos = 0, []
    for nome, img, esperado in casos:
        t0 = time.time()
        r = Motor.detectar(img.copy(), "auto")
        dt = time.time() - t0
        tempos.append(dt)
        obtido = r["placa"] or ""
        certo = obtido == esperado
        ok += certo
        print(f"{'OK ' if certo else 'FAIL'} {nome}: {esperado!r} x "
              f"{obtido!r} via={r['fonte']} {r['detalhe']} {dt:.1f}s",
              flush=True)
    print(f"\nPLACAR: {ok}/20  tempo medio {sum(tempos)/len(tempos):.1f}s",
          flush=True)
    sys.exit(0 if ok == 20 else 1)


if __name__ == "__main__":
    main()
