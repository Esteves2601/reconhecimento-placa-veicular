"""App Android do Reconhecimento de Placas (Kivy) — v1.6.

Visual espelhado no desktop (grafite + vermelho), adaptado para celular:
cartoes arredondados, placa confirmada em branco / provavel em ambar,
barra de certeza, historico das ultimas leituras, status animado.

Motor original usado como biblioteca (copias de build, arquivos da raiz
intactos). Regras: nada inventado (porteira de certeza), erro sempre com
codigo (ver mobile_app/DEBUG.md).

Detalhe tecnico: Main.py importa tkinter no topo (so usa dentro de
selecionarImagem(), que o mobile nunca chama). Como tkinter nao existe no
Android, um stub e registrado em sys.modules ANTES de importar o Motor.
"""

import os
import random
import sys
import threading
import types

_RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(_RAIZ)  # DetectarCaracteres carrega os .txt via caminho relativo
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

# Stub do tkinter (ver docstring acima).
_tk = types.ModuleType("tkinter")
_tk.filedialog = types.ModuleType("tkinter.filedialog")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.filedialog", _tk.filedialog)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import Motor  # noqa: E402
import DetectarCaracteres  # noqa: E402 (treino em 2 etapas, ver _treinar)

from kivy.app import App  # noqa: E402
from kivy.clock import Clock  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.graphics import Color, Ellipse, RoundedRectangle  # noqa: E402
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

# ---------------------------------------------------------------- paleta
# Mesmas cores do desktop (Interface.py). SEC = botoes secundarios grafite.
BG = get_color_from_hex("#0b0b0d")
PANEL = get_color_from_hex("#131316")
CARD = get_color_from_hex("#1b1b1f")
SEC = get_color_from_hex("#2a2a30")
SEC_DOWN = get_color_from_hex("#35353c")
RED = get_color_from_hex("#e10600")
RED_DARK = get_color_from_hex("#8f0400")
TEXT = get_color_from_hex("#f4f4f5")
MUTED = get_color_from_hex("#9d9da8")
AMBAR = get_color_from_hex("#f5a623")
VERDE = get_color_from_hex("#22c55e")
CINZA_OFF = get_color_from_hex("#3a3a42")

PASTA_AMOSTRAS = os.path.join(_RAIZ, "amostras")

# Versao do conteudo (rodapé; versionCode do APK nao muda para a loja
# tratar como atualizacao). Ver mobile_app/DEBUG.md.
VERSAO = "1.7"

REGIOES = ("auto", "brasil", "argentina", "uruguai", "paraguai",
           "internacional")

TETO_LARGURA = 1280  # fotos maiores sao reduzidas (14x mais rapido)
TETO_ANALISE_S = 45  # paciencia por deteccao (UI segue responsiva)


# ---------------------------------------------------------------- motor
def _limitar(img, teto=TETO_LARGURA):
    """Reduz fotos gigantes para o teto util (placa segue legivel)."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= teto:
        return img
    f = teto / m
    return cv2.resize(img, (int(w * f), int(h * f)),
                      interpolation=cv2.INTER_AREA)


def _combinar_paciente(img, regiao, rigoroso):
    """Mesma matematica do Motor.combinar, com paciencia de celular: o
    combinar() original aplica join(timeout=10) no pipeline original, e
    num aparelho lento isso rouba ate 35 pontos da certeza mesmo quando a
    leitura estava certa. Aqui os dois lados terminam."""
    caixas = {}

    def _r():
        caixas["res"] = Motor.detectar(img, regiao)

    def _o():
        caixas["orig"] = Motor.original_rapido(img)

    t1 = threading.Thread(target=_r, daemon=True)
    t2 = threading.Thread(target=_o, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=TETO_ANALISE_S)
    t2.join(timeout=TETO_ANALISE_S)
    if "res" not in caixas:
        raise TimeoutError("tempo esgotado na análise")
    return Motor.fundir(caixas["res"], caixas.get("orig", ""), rigoroso)


def _orientacao_exif(caminho):
    """Tag Orientation do EXIF (JPEG de celular) em puro Python. Retorna 1
    se ausente/ilegivel. O cv2 ignora EXIF: foto em retrato chega girada
    90 graus e nenhuma placa fica legivel."""
    try:
        with open(caminho, "rb") as f:
            dados = f.read(128 * 1024)
        if dados[0:2] != b"\xff\xd8":
            return 1
        i = 2
        while i + 4 < len(dados):
            if dados[i] != 0xFF:
                break
            marca = dados[i + 1]
            tam = (dados[i + 2] << 8) + dados[i + 3]
            if marca == 0xE1 and dados[i + 4:i + 10] == b"Exif\x00\x00":
                t = i + 10
                if dados[t:t + 2] == b"II":
                    endian = "<"
                elif dados[t:t + 2] == b"MM":
                    endian = ">"
                else:
                    return 1
                import struct as _st
                base = t
                off = _st.unpack(endian + "I", dados[t + 4:t + 8])[0]
                n = _st.unpack(endian + "H",
                               dados[base + off:base + off + 2])[0]
                for k in range(n):
                    e = base + off + 2 + 12 * k
                    tag, tipo, num = _st.unpack(endian + "HHI",
                                               dados[e:e + 8])
                    if tag == 0x0112 and tipo == 3 and num == 1:
                        return _st.unpack(endian + "H",
                                          dados[e + 8:e + 10])[0]
                return 1
            if marca in (0xD8, 0xD9):
                break
            i += 2 + tam
    except Exception:
        pass
    return 1


def _desgirar(img, orientacao):
    if orientacao == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientacao == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientacao == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _ler_imagem(caminho):
    """Leitura tolerante a acentos (fromfile+imdecode) e a rotacao EXIF."""
    try:
        buf = np.fromfile(caminho, dtype=np.uint8)
        if buf.size:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return _desgirar(img, _orientacao_exif(caminho))
    except Exception:
        pass
    try:
        img = cv2.imread(caminho)
        if img is not None:
            return _desgirar(img, _orientacao_exif(caminho))
        return None
    except Exception:
        return None


def _placeholder(caminho):
    """Fundo escuro com moldura vermelha para o preview nao nascer branco."""
    img = np.full((640, 640, 3), (22, 19, 19), dtype=np.uint8)  # #131316
    cv2.rectangle(img, (0, 0), (639, 639), (0, 6, 225), 10)  # #e10600
    cv2.imwrite(caminho, img)


def _amostras():
    try:
        arqs = sorted(
            os.path.join(PASTA_AMOSTRAS, a) for a in os.listdir(PASTA_AMOSTRAS)
            if a.lower().endswith((".png", ".jpg", ".jpeg", ".bmp",
                                   ".webp")))
    except Exception:
        arqs = []
    return [a for a in arqs if os.path.isfile(a)]


def _hash_dados():
    """MD5 curto dos 3 arquivos de treino (detecta empacotamento
    corrompido: hash diferente do repo = dados adulterados no APK)."""
    import hashlib
    h = hashlib.md5()
    try:
        for nome in ("classifications.txt", "flattened_images.txt",
                     "base_kNN_ampla.npy", "classes_kNN_ampla.npy"):
            with open(os.path.join(_RAIZ, nome), "rb") as f:
                while True:
                    bloco = f.read(1024 * 1024)
                    if not bloco:
                        break
                    h.update(bloco)
        return h.hexdigest()[:8]
    except Exception:
        return "SEM-ARQ"


def _selftest_ml():
    """KNN sintetico no aparelho: espera 'A'. Se falhar, o cv2.ml do
    aparelho nao funciona (nenhum leitor KNN funcionara)."""
    try:
        dados = np.float32([[0] * 600, [255] * 600])
        rotulos = np.float32([[65], [66]])
        knn = cv2.ml.KNearest_create()
        knn.train(dados, cv2.ml.ROW_SAMPLE, rotulos)
        _r, res, _v, _d = knn.findNearest(np.float32([[0] * 600]), k=1)
        return chr(int(res[0][0]))
    except Exception as exc:
        return f"ERRO:{type(exc).__name__}"


# ---------------------------------------------------------------- widgets
def alinhar(lbl):
    """Faz halign/valign valerem (text_size acompanha o widget)."""
    lbl.bind(size=lambda i, v: setattr(i, "text_size", v))
    return lbl


def rotulo(texto, tamanho=12, cor=MUTED, negrito=False, altura=None):
    return alinhar(Label(text=texto, font_size=sp(tamanho), color=cor,
                         bold=negrito, size_hint_y=None,
                         height=dp(altura or 22), halign="left",
                         valign="middle"))


class Card(BoxLayout):
    """Cartao escuro com borda fina e cantos arredondados."""

    def __init__(self, borda=RED, fundo=CARD, raio=None, esp=None,
                 **kwargs):
        super().__init__(**kwargs)
        raio = dp(14) if raio is None else raio
        esp = dp(1) if esp is None else esp
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


class Ponto(BoxLayout):
    """Bolinha de status desenhada (o glifo unicode nao existe na fonte)."""

    def __init__(self, cor=AMBAR, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            self._tinta = Color(*cor)
            self._bola = Ellipse()
        self.bind(size=self._ajustar, pos=self._ajustar)

    def _ajustar(self, *_):
        d = min(self.width, self.height) * 0.55
        self._bola.size = (d, d)
        self._bola.pos = (self.center_x - d / 2, self.center_y - d / 2)

    def set(self, cor):
        self._tinta.rgba = cor


class BotaoArredondado(Button):
    """Botao plano com cantos arredondados e feedback de toque."""

    def __init__(self, cor=RED, cor_press=None, raio=None, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self._cor = cor
        self._cor_press = cor_press or RED_DARK
        with self.canvas.before:
            self._tinta = Color(*cor)
            self._forma = RoundedRectangle(radius=[raio or dp(14)])
        self.bind(size=self._ajustar, pos=self._ajustar,
                  state=self._pintar, disabled=self._pintar)

    def _ajustar(self, *_):
        self._forma.pos = self.pos
        self._forma.size = self.size

    def _pintar(self, *_):
        if self.disabled:
            self._tinta.rgba = CINZA_OFF
        elif self.state == "down":
            self._tinta.rgba = self._cor_press
        else:
            self._tinta.rgba = self._cor


class OpcaoEscura(SpinnerOption):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_color = CARD
        self.color = TEXT
        self.height = dp(48)


class BarraCerteza(BoxLayout):
    """Barra fina proporcional a certeza (vermelha < limiar, verde >=)."""

    def __init__(self, **kwargs):
        super().__init__(size_hint_y=None, height=dp(6), **kwargs)
        with self.canvas:
            Color(*PANEL)
            self._fundo = RoundedRectangle(radius=[dp(3)])
            self._tinta = Color(*RED)
            self._barra = RoundedRectangle(radius=[dp(3)])
        self.bind(size=self._ajustar, pos=self._ajustar)
        self._valor = 0
        self._limiar = 55

    def _ajustar(self, *_):
        self._fundo.pos = self.pos
        self._fundo.size = self.size
        larg = self.width * max(0, min(self._valor, 100)) / 100.0
        self._barra.pos = self.pos
        self._barra.size = (larg, self.height)

    def set(self, valor, limiar=55):
        self._valor = valor or 0
        self._limiar = limiar
        self._tinta.rgba = VERDE if self._valor >= limiar else RED
        self._ajustar()


# ---------------------------------------------------------------- tela
class Tela(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.regiao = "auto"
        self.rigoroso = False
        self.knn_ok = False
        self._treino_falhou = False
        self._ultima_amostra = None
        self._busy_ev = None
        self._busy_n = 0
        self._arquivo_saida = os.path.join(_RAIZ, "saida_mobile.png")
        _placeholder(self._arquivo_saida)

        col = BoxLayout(orientation="vertical", size_hint_y=None,
                        spacing=dp(8), padding=dp(12))
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

        topo = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        selo = Label(text="P", font_size=sp(20), bold=True, color=TEXT,
                     size_hint_x=None, width=dp(38))
        with selo.canvas.before:
            Color(*RED)
            _s = RoundedRectangle(radius=[dp(10)])
            selo.bind(size=lambda i, v: setattr(_s, "size", v),
                      pos=lambda i, v: setattr(_s, "pos", v))
        topo.add_widget(selo)
        topo.add_widget(alinhar(Label(text="Reconhecimento de placas",
                                      font_size=sp(14), bold=True, color=TEXT,
                                      halign="left", valign="middle")))
        self.status_dot = Ponto(cor=AMBAR, size_hint_x=None, width=dp(20))
        topo.add_widget(self.status_dot)
        self.status_txt = alinhar(Label(text="TREINANDO", font_size=sp(11),
                                         color=MUTED, size_hint_x=None,
                                         width=dp(76), halign="left",
                                         valign="middle"))
        topo.add_widget(self.status_txt)
        col.add_widget(topo)

        # Preview.
        moldura = Card(size_hint_y=None, height=dp(240), padding=dp(8))
        self.preview = Image(source=self._arquivo_saida,
                             allow_stretch=True, keep_ratio=True)
        moldura.add_widget(self.preview)
        col.add_widget(moldura)

        # Cartao de resultado (como o do desktop).
        cartao = Card(size_hint_y=None, height=dp(168), padding=dp(10),
                      spacing=dp(2), orientation="vertical")
        cartao.add_widget(rotulo("PLACA DETECTADA", tamanho=11, negrito=True))
        self.resultado = Label(text="— — —", font_size=sp(34), bold=True,
                               color=TEXT, size_hint_y=None, height=dp(54))
        cartao.add_widget(self.resultado)
        self.barra = BarraCerteza()
        cartao.add_widget(self.barra)
        self.detalhe = alinhar(Label(
            text="Escolha uma foto, fotografe ou tente uma aleatória",
            font_size=sp(12), color=MUTED, size_hint_y=None, height=dp(44),
            halign="left", valign="top"))
        cartao.add_widget(self.detalhe)
        col.add_widget(cartao)

        # Historico das ultimas leituras.
        hist = Card(size_hint_y=None, height=dp(118), padding=dp(10),
                    spacing=dp(2), orientation="vertical")
        hist.add_widget(rotulo("ÚLTIMAS LEITURAS", tamanho=11, negrito=True))
        self.hist_linhas = []
        for _ in range(3):
            linha = rotulo("—", tamanho=12, altura=22)
            self.hist_linhas.append(linha)
            hist.add_widget(linha)
        col.add_widget(hist)

        # Controles.
        controles = Card(size_hint_y=None, height=dp(150), padding=dp(10),
                         spacing=dp(4), orientation="vertical")
        controles.add_widget(rotulo("Região", tamanho=11, negrito=True))
        self.spinner = Spinner(text="auto", values=REGIOES,
                               option_cls=OpcaoEscura, background_normal="",
                               background_color=PANEL, color=TEXT,
                               font_size=sp(14), size_hint_y=None,
                               height=dp(48))
        self.spinner.bind(text=self._ao_trocar_regiao)
        controles.add_widget(self.spinner)
        linha_rigor = BoxLayout(size_hint_y=None, height=dp(44))
        linha_rigor.add_widget(alinhar(Label(text="Modo rigoroso",
                                             font_size=sp(13), color=TEXT,
                                             halign="left",
                                             valign="middle")))
        self.check = CheckBox(active=False, color=RED, size_hint_x=None,
                              width=dp(44))
        self.check.bind(active=self._ao_trocar_rigor)
        linha_rigor.add_widget(self.check)
        controles.add_widget(linha_rigor)
        col.add_widget(controles)

        # Botoes principais.
        botoes = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        b_foto = BotaoArredondado(text="Foto", font_size=sp(15), bold=True,
                                  color=TEXT)
        b_foto.bind(on_release=self._escolher_foto)
        botoes.add_widget(b_foto)
        b_cam = BotaoArredondado(text="Câmera", font_size=sp(15), bold=True,
                                 color=TEXT)
        b_cam.bind(on_release=self._fotografar)
        botoes.add_widget(b_cam)
        if filechooser is None or camera is None:
            b_foto.disabled = True
            b_cam.disabled = True
            self._status("SEM PLYER", AMBAR)
        col.add_widget(botoes)

        # Botao secundario (grafite, como no desktop): foto aleatoria.
        b_alea = BotaoArredondado(text="Foto aleatória", font_size=sp(14),
                                  bold=True, color=TEXT, cor=SEC,
                                  cor_press=SEC_DOWN, size_hint_y=None,
                                  height=dp(48))
        b_alea.bind(on_release=self._foto_aleatoria)
        col.add_widget(b_alea)

        self.lbl_versao = rotulo(f"v{VERSAO} · ml:-- · auto:--",
                                 tamanho=10, altura=20)
        col.add_widget(self.lbl_versao)

        # Treino KNN em fundo, como o desktop. SEMPRE por ultimo: a thread
        # toca nos widgets de status, que precisam ja existir (iniciar
        # antes = crash silencioso e "TREINANDO" eterno).
        threading.Thread(target=self._treinar, daemon=True).start()

    # ---------------------------------------------------------- estado
    def _status(self, texto, cor):
        try:
            txt = getattr(self, "status_txt", None)
            if txt is not None:
                txt.text = texto
            self.status_dot.set(cor)
        except Exception:
            pass

    def _treinar(self):
        # Etapa 1 (KNN pequeno) libera o app em segundos; etapa 2 (base
        # ampla de 37 MB) pode demorar e so adiciona o leitor extra (R5).
        # Tudo protegido: thread que morre em silencio = "TREINANDO" eterno.
        try:
            self._status("TREINO 1/2", AMBAR)
            try:
                ok1 = bool(DetectarCaracteres.loadKNNDataAndTrainKNN())
            except Exception:
                ok1 = False
            self.knn_ok = ok1
            if not ok1:
                self._treino_falhou = True
                self._status("SEM KNN", RED)
                return
            self._status("PRONTO", VERDE)
            self._status("TREINO 2/2", AMBAR)
            try:
                Motor.carregar_amplo()
            except Exception:
                pass
            self._status("PRONTO", VERDE)
            self._autoteste()
        except Exception:
            self.knn_ok = False
            self._status("SEM KNN", RED)

    def _autoteste(self):
        """Prova no aparelho: KNN sintetico + deteccao real na amostra 5
        + integridade dos dados de treino. Tudo vai para o rodape
        (ml:X · auto:Y · dados:Z)."""
        ml = _selftest_ml()
        auto = "--"
        try:
            alvos = [a for a in _amostras()
                     if os.path.basename(a) == "5.png"] or _amostras()
            if alvos:
                img = _limitar(_ler_imagem(alvos[0]))
                if img is not None:
                    saida = _combinar_paciente(img, "auto", False)
                    auto = (saida.get("placa") or
                            f"E{saida.get('certeza', 0)}")
        except Exception as exc:
            auto = f"X99-{type(exc).__name__}"
        dados = _hash_dados()
        texto = f"v{VERSAO} · ml:{ml} · auto:{auto} · dados:{dados}"
        Clock.schedule_once(lambda _dt: self._atualiza_rodape(texto), 0)

    def _atualiza_rodape(self, texto):
        try:
            self.lbl_versao.text = texto
        except Exception:
            pass

    def _ocupado(self, ligado):
        try:
            if self._busy_ev is not None:
                self._busy_ev.cancel()
                self._busy_ev = None
            if ligado:
                self._busy_n = 0
                self._busy_ev = Clock.schedule_interval(
                    self._anima_status, 0.4)
        except Exception:
            pass

    def _anima_status(self, _dt):
        try:
            self._busy_n = (self._busy_n + 1) % 4
            self.status_txt.text = "LENDO" + "." * self._busy_n
        except Exception:
            pass

    def _historico(self, texto):
        try:
            for i in range(len(self.hist_linhas) - 1, 0, -1):
                self.hist_linhas[i].text = self.hist_linhas[i - 1].text
            self.hist_linhas[0].text = texto[:48]
        except Exception:
            pass

    # ---------------------------------------------------------- eventos
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
            self._mostrar_erro(f"Falha ao abrir galeria: {exc}", "D01")

    def _fotografar(self, *_):
        destino = os.path.join(_RAIZ, "foto_mobile.jpg")
        try:
            camera.take_picture(filename=destino,
                                on_complete=self._ao_fotografar)
        except Exception as exc:
            self._mostrar_erro(f"Falha ao abrir câmera: {exc}", "D01")

    def _foto_aleatoria(self, *_):
        try:
            lista = _amostras()
            if not lista:
                self._mostrar_erro("Sem fotos de exemplo neste aparelho.",
                                   "D01")
                return
            opcoes = [a for a in lista if a != self._ultima_amostra]
            caminho = random.choice(opcoes or lista)
            self._ultima_amostra = caminho
            self._processar(caminho,
                            legenda=f"Aleatória: {os.path.basename(caminho)}")
        except Exception as exc:
            self._mostrar_erro(f"Falha na aleatória: {exc}", "D01")

    def _ao_selecionar(self, selecao):
        if selecao:
            self._processar(str(selecao[0]))

    def _ao_fotografar(self, caminho):
        if caminho:
            self._processar(str(caminho))

    def _processar(self, caminho, legenda=""):
        if not self.knn_ok:
            self.resultado.text = "..."
            if self._treino_falhou:
                self.detalhe.text = ("[K00] Sem base KNN: reinicie o app. "
                                     "Se repetir, me informe o código K00.")
            else:
                self.detalhe.text = "[T00] Aguarde o fim do treinamento."
            return
        self.resultado.text = "..."
        self.resultado.color = TEXT
        self.detalhe.text = "Analisando..."
        self.barra.set(0)
        self._ocupado(True)
        regiao, rigoroso = self.regiao, self.rigoroso

        def _alvo():
            try:
                img = _ler_imagem(caminho)
                if img is None:
                    raise ValueError("não foi possível ler a imagem")
                saida = _combinar_paciente(_limitar(img), regiao, rigoroso)
                if legenda:
                    saida["legenda"] = legenda
                Clock.schedule_once(lambda _dt: self._exibir(saida), 0)
            except TimeoutError:
                Clock.schedule_once(
                    lambda _dt: self._mostrar_erro(
                        "Tempo esgotado na análise.", "T02"), 0)
            except ValueError as exc:
                Clock.schedule_once(
                    lambda _dt: self._mostrar_erro(str(exc), "L01"), 0)
            except Exception as exc:
                Clock.schedule_once(
                    lambda _dt: self._mostrar_erro(
                        f"{type(exc).__name__}: {exc}",
                        f"X99-{type(exc).__name__}"), 0)

        threading.Thread(target=_alvo, daemon=True).start()

    def _exibir(self, saida):
        self._ocupado(False)
        placa = (saida.get("placa") or "").strip()
        certeza = saida.get("certeza", 0) or 0
        limiar = saida.get("limiar", 55) or 55
        legenda = saida.get("legenda", "")
        if legenda:
            legenda = legenda + " · "
        self.barra.set(certeza, limiar)
        if placa:
            self.resultado.text = placa
            self.resultado.color = TEXT
            self.detalhe.text = (
                f"{legenda}{saida.get('rotulo', '')} · "
                f"certeza {certeza} "
                f"({saida.get('nivel', '')}) via {saida.get('fonte', '')}"
            )
            self._status("OK", VERDE)
            self._historico(f"{placa} · {certeza}")
        elif saida.get("provavel"):
            prov = saida.get("provavel")
            self.resultado.text = f"PROVÁVEL: {prov}"
            self.resultado.color = AMBAR
            self.detalhe.text = (
                f"{legenda}[E{certeza}] {saida.get('rotulo', '')} "
                f"via {saida.get('fonte', '')} (abaixo do limiar — "
                f"confira na imagem)."
            )
            self._status("DUVIDA", AMBAR)
            self._historico(f"~{prov} · {certeza}")
        else:
            self.resultado.text = "NÃO ENCONTRADA"
            self.resultado.color = TEXT
            motivos = ", ".join(saida.get("motivos", []) or [])
            detalhe_extra = f"[E{certeza}"
            if saida.get("original"):
                detalhe_extra += f" orig:{saida.get('original')}"
            detalhe_extra += "]"
            if motivos:
                detalhe_extra += f" {motivos}"
            self.detalhe.text = (f"{legenda}{detalhe_extra}. "
                                 "Sem evidência suficiente — "
                                 "o app não inventa placa.")
            self._status("VAZIO", AMBAR)
        try:
            anotada = saida.get("anotada")
            if anotada is not None:
                # Arquivo por deteccao (nunca reutilizar o placeholder):
                # alem de evitar cache de textura, prova o que foi analisado.
                self._n_img = getattr(self, "_n_img", 0) + 1
                caminho = os.path.join(
                    _RAIZ, f"vista_{self._n_img % 8}.png")
                if cv2.imwrite(caminho, anotada):
                    self.preview.source = caminho
                    self.preview.reload()
        except Exception:
            pass

    def _mostrar_erro(self, mensagem, codigo="X99"):
        self._ocupado(False)
        self.resultado.text = "ERRO"
        self.resultado.color = TEXT
        self.detalhe.text = f"[{codigo}] {mensagem[:180]}"
        self._status("ERRO", RED)


class AppPlacas(App):
    def build(self):
        self.title = "Reconhecimento de Placas"
        Window.clearcolor = BG
        return Tela()


if __name__ == "__main__":
    AppPlacas().run()
