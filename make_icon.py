# make_icon.py — gera icone_placa.ico (placa simples, sem fundo).
import os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
TEXTO = "ABC123"


def fonte(nome_opcoes, tamanho):
    for nome in nome_opcoes:
        try:
            return ImageFont.truetype(nome, tamanho)
        except Exception:
            continue
    return ImageFont.load_default()


def uma(s):
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    mx = int(s * 0.12)
    pw = s - 2 * mx
    ph = int(s * 0.44)
    px, py = mx, (s - ph) // 2
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=max(2, s // 28),
                        fill=(247, 247, 249, 255),
                        outline=(30, 30, 34, 255), width=max(2, s // 96))
    band_h = int(ph * 0.24)
    d.rounded_rectangle([px + 2, py + 2, px + pw - 2, py + 2 + band_h],
                        radius=max(1, s // 32), fill=(26, 63, 170, 255))
    d.rectangle([px + 2, py + 2 + band_h - max(1, s // 28),
                 px + pw - 2, py + 2 + band_h], fill=(26, 63, 170, 255))
    fb = fonte(("arial.ttf", "DejaVuSans-Bold.ttf"), max(6, int(band_h * 0.52)))
    t = "BRASIL"
    bb = d.textbbox((0, 0), t, font=fb)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text((px + (pw - tw) / 2 - bb[0], py + 2 + (band_h - th) / 2 - bb[1]),
           t, font=fb, fill=(255, 255, 255, 255))
    tamanho = max(8, int(ph * 0.44))
    f = fonte(("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"), tamanho)
    bb = d.textbbox((0, 0), TEXTO, font=f)
    while bb[2] - bb[0] > pw * 0.80 and tamanho > 6:
        tamanho -= 1
        f = fonte(("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"), tamanho)
        bb = d.textbbox((0, 0), TEXTO, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    ay, ah = py + 2 + band_h, py + ph - (py + 2 + band_h)
    d.text((px + (pw - tw) / 2 - bb[0], ay + (ah - th) / 2 - bb[1]),
           TEXTO, font=f, fill=(18, 22, 48, 255))
    return img


if __name__ == "__main__":
    tamanhos = [256, 128, 64, 48, 32, 16]
    imgs = [uma(s) for s in tamanhos]
    destino = os.path.join(BASE, "icone_placa.ico")
    imgs[0].save(destino, format="ICO", sizes=[(s, s) for s in tamanhos])
    print("OK:", destino)
