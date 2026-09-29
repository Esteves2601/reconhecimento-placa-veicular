"""App Android do Reconhecimento de Placas (Kivy).

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
from kivy.utils import get_color_from_hex  # noqa: E402
from kivy.uix.boxlayout import BoxLayout  # noqa: E402
from kivy.uix.button import Button  # noqa: E402
from kivy.uix.checkbox import CheckBox  # noqa: E402
from kivy.uix.image import Image  # noqa: E402
from kivy.uix.label import Label  # noqa: E402
from kivy.uix.spinner import Spinner  # noqa: E402

try:
    from plyer import camera, filechooser
except Exception:  # desktop sem plyer: botoes de captura desabilitados
    camera = None
    filechooser = None

VERMELHO = get_color_from_hex("#E10600")
FUNDO = get_color_from_hex("#101014")
CINZA = get_color_from_hex("#1C1C22")

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


def botao_vermelho(texto):
    return Button(text=texto, size_hint_y=None, height=56,
                  background_color=VERMELHO, bold=True)


class Tela(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=12, spacing=10,
                         **kwargs)
        self.regiao = "auto"
        self.rigoroso = False
        self._arquivo_saida = os.path.join(_RAIZ, "saida_mobile.png")

        titulo = Label(text="RECONHECIMENTO DE PLACAS", bold=True,
                       size_hint_y=None, height=40, color=VERMELHO)
        self.add_widget(titulo)

        self.preview = Image(allow_stretch=True, keep_ratio=True)
        self.add_widget(self.preview)

        self.resultado = Label(text="Escolha uma foto ou fotografe",
                               font_size=28, bold=True, size_hint_y=None,
                               height=60)
        self.add_widget(self.resultado)

        self.detalhe = Label(text="", font_size=14, size_hint_y=None,
                             height=60)
        self.add_widget(self.detalhe)

        linha = BoxLayout(size_hint_y=None, height=48, spacing=10)
        linha.add_widget(Label(text="Região:", size_hint_x=None, width=70))
        self.spinner = Spinner(text="auto", values=REGIOES)
        self.spinner.bind(text=self._ao_trocar_regiao)
        linha.add_widget(self.spinner)
        linha.add_widget(Label(text="Rigoroso:", size_hint_x=None, width=80))
        self.check = CheckBox(active=False)
        self.check.bind(active=self._ao_trocar_rigor)
        linha.add_widget(self.check)
        self.add_widget(linha)

        botoes = BoxLayout(size_hint_y=None, height=56, spacing=10)
        b_foto = botao_vermelho("Foto")
        b_foto.bind(on_release=self._escolher_foto)
        botoes.add_widget(b_foto)
        b_cam = botao_vermelho("Câmera")
        b_cam.bind(on_release=self._fotografar)
        botoes.add_widget(b_cam)
        if filechooser is None or camera is None:
            b_foto.disabled = True
            b_cam.disabled = True
            self.resultado.text = "plyer indisponível neste ambiente"
        self.add_widget(botoes)

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
        self.resultado.text = "Analisando..."
        self.detalhe.text = ""
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
        else:
            self.resultado.text = "Não identificada"
            self.detalhe.text = ("Sem evidência suficiente — "
                                 "o app não inventa placa.")
        try:
            anotada = saida.get("anotada")
            if anotada is not None:
                cv2.imwrite(self._arquivo_saida, anotada)
                self.preview.source = self._arquivo_saida
                self.preview.reload()
        except Exception:
            pass

    def _mostrar_erro(self, mensagem):
        self.resultado.text = "Erro"
        self.detalhe.text = mensagem[:200]


class AppPlacas(App):
    def build(self):
        self.title = "Reconhecimento de Placas"
        return Tela()


if __name__ == "__main__":
    AppPlacas().run()
