"""App Android do Reconhecimento de Placas (Kivy).

Espelha o visual do desktop (Interface.py: grafite + vermelho), adaptado
para smartphone: unidades dp/sp, rolagem, cartao de resultado com borda
vermelha, botoes planos.

Usa o Motor.py ORIGINAL como biblioteca (copias de build, sem alteracoes
nos arquivos da raiz). Regras herdadas do projeto:
- originais nunca sao modificados, so usados como biblioteca;
- sem chave de busca: o app nao inventa placa (exibe so com evidencia).

Detalhe tecnico: Main.py importa tkinter no topo (so usa dentro de
selecionarImagem(), que o mobile nunca chama). Como tkinter nao existe no
Android, um stub e registrado em sys.modules ANTES de importar o Motor.
"""

import os
import sys
import threading
import types

_RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(_RAIZ)  # DetectarCaracteres carrega os .txt via caminho relativo
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

# Stub do tkinter (ver docstring acima): Main.py faz
# `import tkinter as tk` + `from tkinter import filedialog` no topo.
_tk = types.ModuleType("tkinter")
_tk.filedialog = types.ModuleType("tkinter.filedialog")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.filedialog", _tk.filedialog)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import Motor  # noqa: E402

from kivy.app import App  # noqa: E402
from kivy.clock import Clock  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.graphics import Color, RoundedRectangle  # noqa: E402
from kivy.metrics import dp, sp  # noqa: E402
from kivy.uix.boxlayout import BoxLayout  # noqa: E402
from kivy.uix.button import Button  # noqa: E402
from kivy.uix.checkbox import CheckBox  # noqa: E402
from kivy.uix.image import Image  # noqa: E402
from kivy.uix.label import Label  # noqa: E402
from kivy.uix.scrollview import ScrollView  # noqa: E402
from kivy.uix.spinner import Spinner, SpinnerOption  # noqa: E402
from kivy.utils import get_color_from_hex  # noqa: E402

try:
    from plyer import camera, filechooser
except Exception:  # desktop sem plyer: botoes de captura desabilitados
    camera = None
    filechooser = None

# Paleta do desktop (Interface.py).
BG = get_color_from_hex("#0b0b0d")
PANEL = get_color_from_hex("#131316")
CARD = get_color_from_hex("#1b1b1f")
RED = get_color_from_hex("#e10600")
RED_DARK = get_color_from_hex("#8f0400")
TEXT = get_color_from_hex("#f4f4f5")
MUTED = get_color_from_hex("#9d9da8")
AMBAR = get_color_from_hex("#f5a623")
VERDE = get_color_from_hex("#22c55e")

REGIOES = ("auto", "brasil", "argentina", "uruguai", "paraguai",
           "internacional")


def _ler_imagem(caminho):
    """Leitura tolerante a acentos no caminho (np.fromfile + imdecode;
    imread puro falha com pastas/arquivos acentuados)."""
    try:
        buf = np.fromfile(caminho, dtype=np.uint8)
        if buf.size:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass
    try:
        return cv2.imread(caminho)
    except Exception:
        return None


def _placeholder(caminho):
    """Fundo escuro com moldura vermelha para o preview nao nascer branco."""
    img = np.full((640, 640, 3), (22, 19, 19), dtype=np.uint8)  # #131316
    cv2.rectangle(img, (0, 0), (639, 639), (0, 6, 225), 10)  # #e10600
    cv2.imwrite(caminho, img)


class Card(BoxLayout):
    """Cartao escuro com borda (efeito via dois retangulos)."""

    def __init__(self, borda=RED, fundo=CARD, raio=None, esp=None,
                 **kwargs):
        super().__init__(**kwargs)
        raio = dp(12) if raio is None else raio
        esp = dp(2) if esp is None else esp
        self._esp = esp
        with self.canvas.before:
            Color(*borda)
            self._r1 = RoundedRectangle(radius=[raio])
            Color(*fundo)
            self._r2 = RoundedRectangle(radius=[max(raio - esp, 1)])
        self.bind(size=self._ajustar, pos=self._ajustar)

    def _ajustar(self, *_):
        e = self._esp
        self._r1.pos = self.pos
        self._r1.size = self.size
        self._r2.pos = (self.x + e, self.y + e)
        self._r2.size = (self.width - 2 * e, self.height - 2 * e)


class OpcaoEscura(SpinnerOption):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_color = CARD
        self.color = TEXT
        self.height = dp(48)


def alinhar(lbl):
    """Faz halign/valign valerem (text_size acompanha o widget)."""
    lbl.bind(size=lambda i, v: setattr(i, "text_size", v))
    return lbl


def rotulo(texto, tamanho=13, cor=MUTED, negrito=False, altura=None):
    return alinhar(Label(text=texto, font_size=sp(tamanho), color=cor,
                         bold=negrito, size_hint_y=None,
                         height=dp(altura or 24), halign="left",
                         valign="middle"))


class Tela(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.regiao = "auto"
        self.rigoroso = False
        self._arquivo_saida = os.path.join(_RAIZ, "saida_mobile.png")
        _placeholder(self._arquivo_saida)

        col = BoxLayout(orientation="vertical", size_hint_y=None,
                        spacing=dp(10), padding=dp(12))
        col.bind(minimum_height=col.setter("height"))
        self.add_widget(col)

        # Faixa vermelha + cabecalho com selo P e status.
        faixa = BoxLayout(size_hint_y=None, height=dp(3))
        with faixa.canvas.before:
            Color(*RED)
            _r = RoundedRectangle(radius=[0])
            faixa.bind(size=lambda i, v: setattr(_r, "size", v),
                       pos=lambda i, v: setattr(_r, "pos", v))
        col.add_widget(faixa)

        topo = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(10))
        selo = Label(text="P", font_size=sp(20), bold=True, color=TEXT,
                     size_hint_x=None, width=dp(40))
        with selo.canvas.before:
            Color(*RED)
            _s = RoundedRectangle(radius=[dp(10)])
            selo.bind(size=lambda i, v: setattr(_s, "size", v),
                      pos=lambda i, v: setattr(_s, "pos", v))
        topo.add_widget(selo)
        topo.add_widget(alinhar(Label(text="RECONHECIMENTO DE PLACAS",
                                      font_size=sp(15), bold=True, color=RED,
                                      halign="left", valign="middle")))
        self.status_dot = Label(text="●", font_size=sp(14), color=AMBAR,
                                size_hint_x=None, width=dp(20))
        topo.add_widget(self.status_dot)
        self.status_txt = alinhar(Label(text="PRONTO", font_size=sp(11),
                                         color=MUTED, size_hint_x=None,
                                         width=dp(52), halign="left",
                                         valign="middle"))
        topo.add_widget(self.status_txt)
        col.add_widget(topo)

        # Preview.
        moldura = Card(size_hint_y=None, height=dp(300),
                       padding=dp(10))
        self.preview = Image(source=self._arquivo_saida,
                             allow_stretch=True, keep_ratio=True)
        moldura.add_widget(self.preview)
        col.add_widget(moldura)

        # Cartao de resultado (como o do desktop).
        cartao = Card(size_hint_y=None, height=dp(168), padding=dp(12),
                      spacing=dp(4), orientation="vertical")
        cartao.add_widget(rotulo("PLACA DETECTADA", tamanho=11, negrito=True))
        self.resultado = Label(text="— — —", font_size=sp(40), bold=True,
                               color=TEXT, size_hint_y=None, height=dp(62))
        cartao.add_widget(self.resultado)
        self.detalhe = alinhar(Label(text="Escolha uma foto ou fotografe",
                                     font_size=sp(13), color=MUTED,
                                     size_hint_y=None, height=dp(48),
                                     halign="left", valign="top"))
        cartao.add_widget(self.detalhe)
        col.add_widget(cartao)

        # Controles.
        controles = Card(size_hint_y=None, height=dp(168), padding=dp(12),
                         spacing=dp(6), orientation="vertical")
        controles.add_widget(rotulo("Região", tamanho=11, negrito=True))
        self.spinner = Spinner(text="auto", values=REGIOES,
                               option_cls=OpcaoEscura, background_normal="",
                               background_color=PANEL, color=TEXT,
                               font_size=sp(15), size_hint_y=None,
                               height=dp(52))
        self.spinner.bind(text=self._ao_trocar_regiao)
        controles.add_widget(self.spinner)
        linha_rigor = BoxLayout(size_hint_y=None, height=dp(48))
        linha_rigor.add_widget(alinhar(Label(text="Modo rigoroso",
                                             font_size=sp(14), color=TEXT,
                                             halign="left",
                                             valign="middle")))
        self.check = CheckBox(active=False, color=RED, size_hint_x=None,
                              width=dp(48))
        self.check.bind(active=self._ao_trocar_rigor)
        linha_rigor.add_widget(self.check)
        controles.add_widget(linha_rigor)
        col.add_widget(controles)

        # Botoes.
        botoes = BoxLayout(size_hint_y=None, height=dp(60), spacing=dp(10))
        b_foto = self._botao("Foto")
        b_foto.bind(on_release=self._escolher_foto)
        botoes.add_widget(b_foto)
        b_cam = self._botao("Câmera")
        b_cam.bind(on_release=self._fotografar)
        botoes.add_widget(b_cam)
        if filechooser is None or camera is None:
            b_foto.disabled = True
            b_cam.disabled = True
            self._status("PLYER AUSENTE", AMBAR)
        col.add_widget(botoes)

    @staticmethod
    def _botao(texto):
        return Button(text=texto, font_size=sp(16), bold=True, color=TEXT,
                      background_normal="", background_down="",
                      background_color=RED)

    def _status(self, texto, cor):
        self.status_txt.text = texto
        self.status_dot.color = cor

    def _ao_trocar_regiao(self, spinner, texto):
        self.regiao = (texto or "auto").lower()

    def _ao_trocar_rigor(self, check, ativo):
        self.rigoroso = bool(ativo)

    def _escolher_foto(self, *_):
        try:
            filechooser.open_file(filters=["*.jpg", "*.jpeg", "*.png",
                                            "*.bmp", "*.webp"],
                                  on_selection=self._ao_selecionar)
        except Exception as exc:
            self._mostrar_erro(f"Falha ao abrir galeria: {exc}")

    def _fotografar(self, *_):
        destino = os.path.join(_RAIZ, "foto_mobile.jpg")
        try:
            camera.take_picture(filename=destino,
                                on_complete=self._ao_fotografar)
        except Exception as exc:
            self._mostrar_erro(f"Falha ao abrir câmera: {exc}")

    def _ao_selecionar(self, selecao):
        if selecao:
            self._processar(str(selecao[0]))

    def _ao_fotografar(self, caminho):
        if caminho:
            self._processar(str(caminho))

    def _processar(self, caminho):
        self.resultado.text = "..."
        self.detalhe.text = "Analisando..."
        self._status("LENDO", AMBAR)
        regiao, rigoroso = self.regiao, self.rigoroso

        def _alvo():
            try:
                img = _ler_imagem(caminho)
                if img is None:
                    raise ValueError("não foi possível ler a imagem")
                saida = Motor.combinar(img, regiao, rigoroso)
                Clock.schedule_once(lambda _dt: self._exibir(saida), 0)
            except Exception as exc:
                Clock.schedule_once(
                    lambda _dt: self._mostrar_erro(str(exc)), 0)

        threading.Thread(target=_alvo, daemon=True).start()

    def _exibir(self, saida):
        placa = (saida.get("placa") or "").strip()
        if placa:
            self.resultado.text = placa
            self.detalhe.text = (
                f"{saida.get('rotulo', '')} · "
                f"certeza {saida.get('certeza', 0)} "
                f"({saida.get('nivel', '')}) via {saida.get('fonte', '')}"
            )
            self._status("OK", VERDE)
        else:
            self.resultado.text = "NÃO ENCONTRADA"
            self.detalhe.text = ("Sem evidência suficiente — "
                                 "o app não inventa placa.")
            self._status("VAZIO", AMBAR)
        try:
            anotada = saida.get("anotada")
            if anotada is not None:
                cv2.imwrite(self._arquivo_saida, anotada)
                self.preview.source = self._arquivo_saida
                self.preview.reload()
        except Exception:
            pass

    def _mostrar_erro(self, mensagem):
        self.resultado.text = "ERRO"
        self.detalhe.text = mensagem[:200]
        self._status("ERRO", RED)


class AppPlacas(App):
    def build(self):
        self.title = "Reconhecimento de Placas"
        Window.clearcolor = BG
        return Tela()


if __name__ == "__main__":
    AppPlacas().run()
