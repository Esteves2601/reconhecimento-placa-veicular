"""Reconhecimento de Placas — app desktop (tkinter, sem terminal: pyw app_gui.py).

Segue o padrao dos apps irmaos (detector_objetos/app_gui.py):
GUI fina em tkinter + motor em Motor.py (sistema original intacto).
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import Main as legacy
import Motor

# ---------- Tema (preto + vermelho) ----------
BG = "#0b0b0d"
PANEL = "#131316"
CARD = "#1b1b1f"
LINE = "#26262c"
RED = "#e10600"
RED_HOVER = "#ff2323"
RED_DARK = "#8f0400"
TEXT = "#f4f4f5"
MUTED = "#9d9da8"
GREEN = "#22c55e"
FONT_TITULO = ("Segoe UI", 19, "bold")
FONT_SUB = ("Segoe UI", 11)
FONT_SEC = ("Segoe UI", 11, "bold")
FONT_BODY = ("Segoe UI", 11)
FONT_BTN = ("Segoe UI", 11, "bold")
FONT_PLATE = ("Consolas", 38, "bold")

REGIOES = {"Automática": "auto", "Brasil": "brasil",
           "Argentina": "argentina", "Uruguai": "uruguai",
           "Paraguai": "paraguai", "Internacional": "internacional"}


def aplicar_tema(root):
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except Exception:
        pass
    st.configure("TNotebook", background=PANEL, borderwidth=0)
    st.configure("TNotebook.Tab", background="#1e1e22", foreground=MUTED,
                 font=("Segoe UI", 11, "bold"), padding=(14, 8))
    st.map("TNotebook.Tab",
           background=[("selected", RED)],
           foreground=[("selected", "white")])
    st.configure("TCombobox", fieldbackground="#0e0e10",
                 background=PANEL, foreground=TEXT,
                 arrowcolor=RED, borderwidth=0)
    st.configure("TProgressbar", background=RED, troughcolor="#0e0e10",
                 borderwidth=0, thickness=8)
    st.configure("TEntry", fieldbackground="#0e0e10", foreground=TEXT)
    return st


def foto_tk(img, max_w=640, max_h=380):
    if img is None:
        return None
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    esc = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * esc)), max(1, int(h * esc))
    return ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (nw, nh))))


class BotaoHover(tk.Button):
    """Botao flat com hover (vermelho ou grafite)."""

    def __init__(self, pai, vermelho=True, **kw):
        self._cor = RED if vermelho else "#2a2a30"
        self._hover = RED_HOVER if vermelho else "#35353c"
        super().__init__(pai, bg=self._cor, fg="white",
                         activebackground=self._hover,
                         activeforeground="white", relief="flat",
                         font=FONT_BTN, cursor="hand2", pady=9, **kw)
        self.bind("<Enter>", lambda _e: self.configure(bg=self._hover))
        self.bind("<Leave>", lambda _e: self.configure(bg=self._cor))


class Cartao(tk.Frame):
    """Quadro escuro com titulo em vermelho e borda sutil."""

    def __init__(self, pai, titulo, **kw):
        super().__init__(pai, bg=CARD, highlightbackground=LINE,
                         highlightthickness=1, **kw)
        tk.Label(self, text=titulo, bg=CARD, fg=RED_HOVER,
                 font=("Segoe UI", 10, "bold")).pack(pady=(10, 2))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Reconhecimento de Placas")
        root.geometry("1180x760")
        root.minsize(1020, 660)
        root.configure(bg=BG)
        aplicar_tema(root)
        try:
            root.update_idletasks()
            x = (root.winfo_screenwidth() - 1180) // 2
            y = (root.winfo_screenheight() - 760) // 2
            root.geometry(f"1180x760+{max(x, 0)}+{max(y, 0)}")
        except Exception:
            pass
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            root.iconbitmap(os.path.join(base, "icone_placa.ico"))
        except Exception:
            pass

        self.knn_ok = False
        self.cap = None
        self.cam_on = False
        self.img_atual = None
        self.buscando = False
        self.detectando = False
        self.foto_ref = None
        self.mini_refs = []

        header = tk.Frame(root, bg=PANEL)
        header.pack(fill="x")
        tk.Frame(header, bg=RED, height=3).pack(fill="x")
        marca = tk.Label(header, text="P", bg=RED, fg="white",
                         font=("Segoe UI", 22, "bold"), width=2)
        marca.pack(side="left", padx=(18, 12), pady=12)
        textos = tk.Frame(header, bg=PANEL)
        textos.pack(side="left", pady=10)
        tk.Label(textos, text="RECONHECIMENTO DE PLACAS",
                 bg=PANEL, fg=TEXT, font=FONT_TITULO).pack(anchor="w")
        tk.Label(textos, text="Imagem • Câmera ao vivo • Busca online",
                 bg=PANEL, fg=MUTED, font=FONT_SUB).pack(anchor="w")
        self.lbl_status = tk.Label(header, text="● Iniciando…", bg=PANEL,
                                   fg="#f5a623", font=FONT_BODY)
        self.lbl_status.pack(side="right", padx=18)

        corpo = tk.Frame(root, bg=BG)
        corpo.pack(fill="both", expand=True, padx=14, pady=14)
        lateral = tk.Frame(corpo, bg=PANEL, width=340,
                           highlightbackground=LINE, highlightthickness=1)
        lateral.pack(side="left", fill="y", padx=(0, 14))
        lateral.pack_propagate(False)
        direita = tk.Frame(corpo, bg=PANEL, highlightbackground=LINE,
                           highlightthickness=1)
        direita.pack(side="left", fill="both", expand=True)

        self.abas = ttk.Notebook(lateral)
        self.abas.pack(padx=12, pady=12, fill="x")
        for nome in ("Imagem", "Ao vivo", "Buscar carro"):
            self.abas.add(tk.Frame(self.abas, bg=CARD), text=nome)
        tabs = self.abas.tabs()
        self._aba_imagem(self.abas.nametowidget(tabs[0]))
        self._aba_camera(self.abas.nametowidget(tabs[1]))
        self._aba_online(self.abas.nametowidget(tabs[2]))

        regf = Cartao(lateral, "REGIÃO DAS PLACAS")
        regf.pack(fill="x", padx=12, pady=(0, 6))
        self.opcao_regiao = ttk.Combobox(regf, values=list(REGIOES),
                                         state="readonly", width=26)
        self.opcao_regiao.set("Automática")
        self.opcao_regiao.pack(padx=10, pady=(2, 6))
        tk.Label(regf, text="EXIGÊNCIA", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10, "bold")).pack(pady=(6, 2))
        self.opcao_rigor = ttk.Combobox(regf, values=["Normal", "Rigoroso"],
                                        state="readonly", width=26)
        self.opcao_rigor.set("Normal")
        self.opcao_rigor.pack(padx=10, pady=(2, 12))

        self.preview = tk.Label(direita, text="Nenhuma imagem\ncarregada",
                                bg=PANEL, fg=MUTED, font=("Segoe UI", 15))
        self.preview.pack(padx=16, pady=(16, 8), fill="both", expand=True)

        minis = tk.Frame(direita, bg=PANEL)
        minis.pack(fill="x", padx=16, pady=(0, 8))
        self.lbl_mini_analise = self._mini(minis, "ANÁLISE DO SISTEMA", 0)
        self.lbl_mini_recorte = self._mini(minis, "RECORTE DA PLACA", 1)

        cartao = tk.Frame(direita, bg=CARD, highlightbackground=RED,
                          highlightthickness=2)
        cartao.pack(fill="x", padx=16, pady=8)
        tk.Label(cartao, text="PLACA DETECTADA", bg=CARD, fg=RED_HOVER,
                 font=FONT_SEC).pack(pady=(8, 0))
        self.lbl_placa = tk.Label(cartao, text="— — —", bg=CARD, fg=TEXT,
                                  font=FONT_PLATE)
        self.lbl_placa.pack(pady=(0, 8))

        self.lbl_fonte = tk.Label(direita, text="Fonte: —", bg=PANEL,
                                  fg=MUTED, font=FONT_BODY, wraplength=640,
                                  justify="left")
        self.lbl_fonte.pack(fill="x", padx=16, pady=(0, 4))
        self.log = tk.Text(direita, height=7, bg="#0e0e11", fg=MUTED,
                           font=FONT_BODY, relief="flat",
                           highlightthickness=1, highlightbackground=LINE)
        self.log.pack(fill="x", padx=16, pady=(4, 16))
        self.logar("Sistema iniciado. Treinando classificador…")
        threading.Thread(target=self.treinar, daemon=True).start()
        Motor.preparar_ia_async(
            lambda ok: self.root.after(0, lambda: self.logar(
                "IA neural pronta (leitura principal)." if ok else
                "IA neural ausente; usando sistema original.")))
        root.protocol("WM_DELETE_WINDOW", self.fechar)

    # ---------- construção ----------
    def _botao(self, pai, texto, comando, vermelho=True):
        b = BotaoHover(pai, text=texto, command=comando, vermelho=vermelho)
        b.pack(fill="x", padx=10, pady=4)
        return b

    def _entrada(self, pai, titulo, secreto=False):
        tk.Label(pai, text=titulo, bg=CARD, fg=MUTED,
                 font=FONT_SEC).pack(anchor="w", padx=10, pady=(8, 0))
        e = tk.Entry(pai, bg="#0e0e10", fg=TEXT, insertbackground=TEXT,
                     relief="flat", font=FONT_BODY,
                     show="•" if secreto else "",
                     highlightthickness=1, highlightbackground=LINE)
        e.pack(fill="x", padx=10, pady=4, ipady=6)
        return e

    def _aba_imagem(self, aba):
        self._botao(aba, "📁  Selecionar imagem", self.selecionar_arquivo)
        self._botao(aba, "🎲  Imagem aleatória", self.imagem_aleatoria, False)
        self.btn_detectar = self._botao(aba, "⚡  DETECTAR PLACA",
                                        self.detectar_atual)

    def _aba_camera(self, aba):
        self._botao(aba, "▶  Iniciar câmera", self.iniciar_camera)
        self._botao(aba, "📸  Capturar e detectar",
                    self.capturar_e_detectar, False)
        self._botao(aba, "⏹  Parar câmera", self.parar_camera, False)

    def _aba_online(self, aba):
        self.entry_modelo = self._entrada(aba, "Modelo do veículo")
        self.entry_key = self._entrada(aba, "Google API Key", secreto=True)
        self.entry_cx = self._entrada(aba, "Google CX")
        self.btn_buscar = self._botao(aba, "🌐  BUSCAR NO GOOGLE",
                                      self.buscar_online)
        self.entry_url = self._entrada(aba, "Ou URL direta da imagem")
        self._botao(aba, "⬇  BAIXAR URL E DETECTAR",
                    self.baixar_url_detectar, False)
        self.barra = ttk.Progressbar(aba, mode="indeterminate")
        self.barra.pack(fill="x", padx=10, pady=6)
        self.lbl_busca = tk.Label(aba, text="Digite o modelo e busque.",
                                  bg=CARD, fg=MUTED, font=FONT_BODY,
                                  wraplength=280, justify="left")
        self.lbl_busca.pack(fill="x", padx=10, pady=(0, 10))

    def _mini(self, pai, titulo, coluna):
        moldura = tk.Frame(pai, bg=CARD, highlightbackground=LINE,
                           highlightthickness=1)
        moldura.grid(row=0, column=coluna, sticky="ew",
                     padx=(0, 8) if coluna == 0 else (8, 0))
        pai.grid_columnconfigure(coluna, weight=1)
        tk.Label(moldura, text=titulo, bg=CARD, fg=RED_HOVER,
                 font=("Segoe UI", 10, "bold")).pack(pady=(6, 2))
        rotulo = tk.Label(moldura, text="—", bg=CARD, fg=MUTED,
                          font=FONT_BODY, height=5)
        rotulo.pack(pady=(0, 6))
        return rotulo

    # ---------- base ----------
    def regiao(self):
        return REGIOES.get(self.opcao_regiao.get(), "auto")

    def rigoroso(self):
        try:
            return self.opcao_rigor.get() == "Rigoroso"
        except Exception:
            return False

    def logar(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def mostrar(self, img):
        self.img_atual = img
        foto = foto_tk(img)
        if foto:
            self.foto_ref = foto
            self.preview.configure(image=foto, text="")

    def mostrar_detalhes(self, recorte, analise):
        for img, rotulo in ((recorte, self.lbl_mini_recorte),
                            (analise, self.lbl_mini_analise)):
            foto = foto_tk(img, max_w=300, max_h=86) if img is not None else None
            if foto:
                self.mini_refs.append(foto)
                rotulo.configure(image=foto, text="")
            else:
                rotulo.configure(image="", text="—")

    def mostrar_resultado(self, res):
        placa = res.get("placa") or ""
        if res.get("anotada") is not None:
            self.mostrar(res["anotada"])
        self.mostrar_detalhes(res.get("recorte"), res.get("analise"))
        rotulo = res.get("rotulo") or ""
        if not rotulo and placa:
            bruta = Motor.normalizar(placa)
            _, _, rotulo = Motor.interpretar(bruta, self.regiao())
        como = {"ia": "lida por IA", "knn": "lida por KNN",
                "knn-amplo": "KNN amplo (k=7)",
                "ia-cena": "lida por IA (cena)",
                "render": "prova por render",
                "consenso": "consenso das leituras"}.get(
                    res.get("fonte") or "", "")
        como = f" ({como})" if como else ""
        det = f" [{res['detalhe']}]" if res.get("detalhe") else ""
        if "certeza" in res:
            niv = f" ✓ {res['nivel']} {res['certeza']}%"
            mot = f" ({', '.join(res.get('motivos', []))})"
        else:
            niv, mot = "", ""
        if placa:
            self.lbl_placa.configure(text=placa)
            self.logar(f"Placa: {placa} [{rotulo}]{como}{det}{niv}{mot}")
        else:
            self.lbl_placa.configure(text="NÃO ENCONTRADA")
            if "certeza" in res:
                self.logar(f"Inconclusiva ({res['certeza']}% < "
                           f"{res.get('limiar', 55)}%){mot}.{como}{det}")
            else:
                self.logar("Nenhuma placa reconhecida nesta imagem.")

    def treinar(self):
        ok = Motor.treinar()
        self.knn_ok = ok
        if ok:
            self.lbl_status.configure(text="● Pronto", fg="#22c55e")
            self.logar("Classificador treinado. Pronto.")
        else:
            self.lbl_status.configure(text="● Falha no treino", fg=RED_HOVER)

    # ---------- imagem ----------
    def selecionar_arquivo(self):
        caminho = filedialog.askopenfilename(
            title="Escolha uma imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("Todos", "*.*")])
        if not caminho:
            return
        try:
            img = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            try:
                messagebox.showerror("Erro", "Não foi possível ler o arquivo.")
            except Exception:
                pass
            return
        self.mostrar(img)
        self.logar(f"Carregado: {os.path.basename(caminho)}")

    def imagem_aleatoria(self):
        try:
            import random
            lista = legacy.listarImagens()
            if not lista:
                self.logar("Pasta de imagens vazia.")
                return
            caminho = random.choice(lista)
            img = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is not None:
                self.mostrar(img)
                self.logar(f"Aleatória: {os.path.basename(caminho)}")
        except Exception as e:
            self.logar(f"Erro: {e}")

    def detectar_atual(self):
        if not self.knn_ok:
            self.logar("Aguarde o fim do treinamento.")
            return
        if self.img_atual is None:
            self.logar("Carregue ou capture uma imagem primeiro.")
            return
        if getattr(self, "_detectando", False):
            self.logar("Já há uma análise em curso; aguarde.")
            return
        self._detectando = True
        self.btn_detectar.configure(state="disabled")
        self.lbl_placa.configure(text="...")
        self.mostrar_detalhes(None, None)
        self.logar("Analisando imagem…")
        threading.Thread(target=self._detectar_bg,
                         args=(self.img_atual.copy(),), daemon=True).start()

    def _detectar_bg(self, img):
        import time as _tempo
        t0 = _tempo.perf_counter()
        try:
            res = Motor.combinar(img, self.regiao(), self.rigoroso())
            dt = _tempo.perf_counter() - t0
            self.root.after(0, lambda: (
                self.mostrar_resultado(res),
                self.logar(f"Análise em {dt:.1f}s (teto 14s).")))
        except Exception as e:
            self.root.after(0, lambda: self.logar(f"Erro: {e}"))
        finally:
            self.root.after(0, self._liberar)

    def _liberar(self):
        self._detectando = False
        try:
            self.btn_detectar.configure(state="normal")
        except Exception:
            pass

    # ---------- câmera ----------
    def iniciar_camera(self):
        if self.cam_on:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.logar("Webcam não encontrada.")
            self.cap = None
            return
        self.cam_on = True
        self.logar("Câmera iniciada.")
        self._frame()

    def _frame(self):
        if not self.cam_on or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self.mostrar(frame)
        self.root.after(40, self._frame)

    def capturar_e_detectar(self):
        if self.img_atual is None:
            self.logar("Inicie a câmera primeiro.")
            return
        self.detectar_atual()

    def parar_camera(self):
        self.cam_on = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.logar("Câmera parada.")

    # ---------- busca ----------
    @staticmethod
    def _dominio(url):
        try:
            return url.split("/")[2].lower()
        except Exception:
            return url[:40]

    def buscar_online(self):
        if self.buscando:
            return
        modelo = self.entry_modelo.get().strip()
        key = self.entry_key.get().strip()
        cx = self.entry_cx.get().strip()
        if not modelo:
            self.logar("Digite o modelo do veículo.")
            return
        if not key or not cx:
            self.logar("Informe API Key e CX. Ou cole a URL direta abaixo.")
            return
        if not self.knn_ok:
            self.logar("Aguarde o fim do treinamento.")
            return
        self.buscando = True
        self.btn_buscar.configure(state="disabled", text="BUSCANDO…")
        self.barra.start()
        self.lbl_fonte.configure(text="Fonte: —")
        self.lbl_busca.configure(text=f"Buscando “{modelo}”…")
        threading.Thread(target=self._busca_bg,
                         args=(modelo, key, cx), daemon=True).start()

    def _busca_bg(self, modelo, key, cx):
        try:
            itens = Motor.buscar_google(modelo, key, cx)
        except Exception as e:
            self.root.after(0, lambda: self._busca_fim(f"Falha: {e}"))
            return
        if not itens:
            self.root.after(0, lambda: self._busca_fim("Sem resultados."))
            return
        urls, vistos = [], set()
        resto = [it["link"] for it in itens]
        while resto:
            for u in list(resto):
                dom = self._dominio(u)
                if dom not in vistos:
                    urls.append(u)
                    vistos.add(dom)
                    resto.remove(u)
                    break
            else:
                urls.extend(resto)
                break
        reg, rig = self.regiao(), self.rigoroso()
        primeira = None
        for i, url in enumerate(urls, 1):
            dom = self._dominio(url)
            self.root.after(0, lambda i=i, n=len(urls), dom=dom: (
                self.lbl_busca.configure(text=f"Tentativa {i}/{n} • {dom}…"),
                self.barra.step(100 / max(n, 1))))
            try:
                img = Motor.baixar_imagem(url)
                if img.shape[1] < Motor.LARGURA_MINIMA:
                    continue
                if primeira is None:
                    primeira = {"url": url, "img": img}
                res = Motor.combinar(img.copy(), reg, rig)
                if res["placa"]:
                    self.root.after(
                        0, lambda url=url, res=res: self._busca_ok(
                            url, res))
                    return
            except Exception:
                continue
        if primeira is not None:
            self.root.after(0, lambda: self._busca_sem_placa(primeira))
        else:
            self.root.after(0, lambda: self._busca_fim("Nada pôde ser baixado."))

    def baixar_url_detectar(self):
        url = self.entry_url.get().strip()
        if not url:
            self.logar("Cole a URL da imagem.")
            return
        if not self.knn_ok:
            self.logar("Aguarde o fim do treinamento.")
            return
        if self.buscando:
            return
        self.buscando = True
        self.btn_buscar.configure(state="disabled", text="BAIXANDO…")
        self.barra.start()
        self.lbl_busca.configure(text="Baixando imagem…")
        threading.Thread(target=self._url_bg, args=(url,), daemon=True).start()

    def _url_bg(self, url):
        try:
            img = Motor.baixar_imagem(url)
        except Exception as e:
            self.root.after(0, lambda: self._busca_fim(f"Falha: {e}"))
            return
        try:
            res = Motor.combinar(img.copy(), self.regiao(), self.rigoroso())
        except Exception as e:
            self.root.after(0, lambda: self._busca_fim(f"Falha: {e}"))
            return
        self.root.after(0, lambda: self._busca_ok(url, res))

    def _busca_ok(self, url, res):
        self.barra.stop()
        self.buscando = False
        self.btn_buscar.configure(state="normal", text="BUSCAR NO GOOGLE")
        self.lbl_busca.configure(text="Foto encontrada com placa válida.")
        dom = self._dominio(url)
        self.lbl_fonte.configure(text=f"Fonte (foto real): {dom}\n{url[:110]}")
        self.mostrar_resultado(res)
        self.logar(f"Fonte: {dom}")

    def _busca_sem_placa(self, primeira):
        self.barra.stop()
        self.buscando = False
        self.btn_buscar.configure(state="normal", text="BUSCAR NO GOOGLE")
        dom = self._dominio(primeira["url"])
        self.lbl_busca.configure(text="Foto sem placa válida visível.")
        self.lbl_fonte.configure(
            text=f"Fonte (foto real): {dom}\n{primeira['url'][:110]}")
        self.mostrar(primeira["img"])
        self.mostrar_detalhes(None, None)
        self.lbl_placa.configure(text="NÃO ENCONTRADA")

    def _busca_fim(self, msg):
        self.buscando = False
        self.btn_buscar.configure(state="normal", text="BUSCAR NO GOOGLE")
        self.barra.stop()
        self.logar(msg)
        self.lbl_busca.configure(text=msg)

    def fechar(self):
        self.parar_camera()
        self.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
