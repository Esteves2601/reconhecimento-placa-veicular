# ml_gerar_base.py — gera a base AMPLA do KNN (arquivos NOVOS, sem tocar
# nos classifications.txt / flattened_images.txt originais).
# 36 classes (0-9, A-Z) x 10 fontes x 3 espessuras x 5 deslocamentos
# x 3 rotacoes = 450/classe (16.200 amostras), 20x30, binarias.
# Saidas: base_kNN_ampla.npy + classes_kNN_ampla.npy (binario = carga rapida)
# Uso: .venv\Scripts\python.exe ml_gerar_base.py
# -*- coding: utf-8 -*-
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONTES = ["DejaVuSansCondensed-Bold.ttf", "DejaVuSansCondensed.ttf",
          "bahnschrift.ttf", "arialbd.ttf", "tahomabd.ttf",
          "DejaVuSans-Bold.ttf", "georgiab.ttf", "trebucbd.ttf",
          "arialbi.ttf", "DejaVuSansCondensed-Oblique.ttf"]
TAM_W, TAM_H = 20, 30
ESCALA = 3


def amostra(char, fonte_nome, stroke, dx, dy, angulo):
    # Imita o pipeline: corte justo (estreito/alto) esticado p/ 20x30.
    iw, ih = 14 * ESCALA, 28 * ESCALA
    grande = Image.new("L", (iw, ih), 0)
    d = ImageDraw.Draw(grande)
    tamanho = 60
    while tamanho > 10:
        f = ImageFont.truetype(fonte_nome, tamanho)
        bb = d.textbbox((0, 0), char, font=f, stroke_width=stroke)
        if (bb[2] - bb[0] <= iw * 0.94
                and bb[3] - bb[1] <= ih * 0.94):
            break
        tamanho -= 2
    cx = iw / 2 + dx * ESCALA
    cy = ih / 2 + dy * ESCALA
    d.text((cx, cy), char, font=f, fill=255, anchor="mm",
           stroke_width=stroke, stroke_fill=255)
    if angulo:
        grande = grande.rotate(angulo, resample=Image.BICUBIC,
                               center=(cx, cy))
    pequena = grande.resize((TAM_W, TAM_H), Image.LANCZOS)
    binaria = (np.asarray(pequena) > 127).astype(np.float32) * 255.0
    return binaria.reshape(-1)


def main():
    X, y = [], []
    for char in CLASSES:
        n = 0
        for fonte in FONTES:
            for stroke in (0, 1, 2):
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    for ang in (-4, 0, 4):
                        X.append(amostra(char, fonte, stroke, dx, dy, ang))
                        y.append(float(ord(char)))
                        n += 1
        print(f"{char}: {n} amostras", flush=True)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print("total:", X.shape, "classes:", len(set(y.tolist())), flush=True)
    np.save(os.path.join(BASE, "base_kNN_ampla.npy"), X)
    np.save(os.path.join(BASE, "classes_kNN_ampla.npy"), y)
    print("OK: base_kNN_ampla.npy + classes_kNN_ampla.npy", flush=True)


if __name__ == "__main__":
    main()
