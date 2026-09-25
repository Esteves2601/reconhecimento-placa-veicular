# Interface.py
# -*- coding: utf-8 -*-
"""Interface visual do reconhecimento de placas (tema grafite + vermelho).

NÃO altera o sistema original: usa Main/Dtectar*/Motor como biblioteca.
"""
import os
import threading

import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

import Main as legacy
import Motor

BG = "#0b0b0d"
PANEL = "#131316"
CARD = "#1b1b1f"
LINE = "#26262c"
RED = "#e10600"
RED_HOVER = "#ff2323"
TEXT = "#f4f4f5"
MUTED = "#9d9da8"

ctk.set_appearance_mode("dark")

FONT_TITLE = ("Segoe UI", 19, "bold")
FONT_SUB = ("Segoe UI", 12)
FONT_SEC = ("Segoe UI", 11, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_PLATE = ("Consolas", 40, "bold")

REGIOES_UI = {"Automática": "auto", "Brasil": "brasil",
              "Argentina": "argentina", "Uruguai": "uruguai",
              "Paraguai": "paraguai", "Internacional": "internacional"}


def foto_ctk(img_bgr, max_w=680, max_h=400):
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    esc = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * esc)), max(1, int(h * esc))
    pil = Image.fromarray(cv2.resize(rgb, (nw, nh)))
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=(nw, nh))


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reconhecimento de Placas")
        self.geometry("1180x760")
        self.minsize(1020, 660)
        self.configure(fg_color=BG)
        self.knn_ok = False
        self.cap = None
        self.cam_on = False
        self.img_atual = None
        self.buscando = False
        self._detectando = False

        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        header.pack(fill="x")
        ctk.CTkFrame(header, fg_color=RED, height=3, corner_radius=0).pack(
            fill="x")
        marca = ctk.CTkFrame(header, fg_color=RED, width=44, height=44,
                             corner_radius=12)
        marca.pack(side="left", padx=(18, 12), pady=12)
        marca.pack_propagate(False)
        ctk.CTkLabel(marca, text="P", font=("Segoe UI", 22, "bold"),
                     text_color="white").pack(expand=True)
        textos = ctk.CTkFrame(header, fg_color="transparent")
        textos.pack(side="left", pady=10)
        ctk.CTkLabel(textos, text="RECONHECIMENTO DE PLACAS",
                     font=FONT_TITLE, text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(
            textos, text="Detecção por imagem, câmera ao vivo ou busca",
            font=FONT_SUB, text_color=MUTED).pack(anchor="w")
        self.pill = ctk.CTkFrame(header, fg_color=CARD, corner_radius=20,
                                 border_width=1, border_color=LINE)
        self.pill.pack(side="right", padx=18)
        self.dot = ctk.CTkLabel(self.pill, text="●", font=("Segoe UI", 14),
                                text_color="#f5a623")
        self.dot.pack(side="left", padx=(14, 4), pady=8)
        self.lbl_pill = ctk.CTkLabel(self.pill, text="Iniciando…",
                                     font=FONT_BODY, text_color=MUTED)
        self.lbl_pill.pack(side="left", padx=(0, 14), pady=8)

        corpo = ctk.CTkFrame(self, fg_color="transparent")
        corpo.pack(fill="both", expand=True, padx=14, pady=14)
        corpo.grid_columnconfigure(0, weight=0)
        corpo.grid_columnconfigure(1, weight=1)
        corpo.grid_rowconfigure(0, weight=1)

        lateral = ctk.CTkFrame(corpo, fg_color=PANEL, corner_radius=16,
                               border_width=1, border_color=LINE, width=340)
        lateral.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        lateral.pack_propagate(False)

        self.tabs = ctk.CTkTabview(
            lateral, fg_color=CARD, segmented_button_fg_color=PANEL,
            segmented_button_selected_color=RED,
            segmented_button_selected_hover_color=RED_HOVER, text_color=MUTED,
            segmented_button_unselected_hover_color="#2a2a30", width=310)
        self.tabs.pack(padx=12, pady=12, fill="x")
        self.tabs.add("Imagem")
        self.tabs.add("Ao vivo")
        self.tabs.add("Buscar carro")

        btn_cfg = dict(corner_radius=10, height=38, font=FONT_BODY)
        aba = self.tabs.tab("Imagem")
        ctk.CTkButton(aba, text="Selecionar imagem", fg_color=RED,
                      hover_color=RED_HOVER, command=self.selecionar_arquivo,
                      **btn_cfg).pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkButton(aba, text="Imagem aleatória da pasta", fg_color="#2a2a30",
                      hover_color="#35353c", command=self.imagem_aleatoria,
                      **btn_cfg).pack(fill="x", padx=10, pady=6)
        self.btn_detectar = ctk.CTkButton(
            aba, text="DETECTAR PLACA", fg_color=RED, hover_color=RED_HOVER,
            font=("Segoe UI", 13, "bold"), corner_radius=10, height=44,
            command=self.detectar_atual)
        self.btn_detectar.pack(fill="x", padx=10, pady=(12, 10))

        aba = self.tabs.tab("Ao vivo")
        ctk.CTkButton(aba, text="Iniciar câmera", fg_color=RED,
                      hover_color=RED_HOVER, command=self.iniciar_camera,
                      **btn_cfg).pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkButton(aba, text="Capturar e detectar", fg_color="#2a2a30",
                      hover_color="#35353c", command=self.capturar_e_detectar,
                      **btn_cfg).pack(fill="x", padx=10, pady=6)
        ctk.CTkButton(aba, text="Parar câmera", fg_color="#2a2a30",
                      hover_color="#35353c", command=self.parar_camera,
                      **btn_cfg).pack(fill="x", padx=10, pady=(6, 10))

        aba = self.tabs.tab("Buscar carro")
        ctk.CTkLabel(aba, text="Modelo do veículo", font=FONT_SEC,
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(10, 2))
        self.entry_modelo = ctk.CTkEntry(aba, placeholder_text="Ex: Honda Civic",
                                         height=40, corner_radius=10,
                                         fg_color=PANEL, border_color=LINE)
        self.entry_modelo.pack(fill="x", padx=10, pady=4)
        self.entry_modelo.bind("<Return>", lambda _e: self.buscar_online())
        ctk.CTkLabel(aba, text="Google API Key", font=FONT_SEC,
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8, 2))
        self.entry_key = ctk.CTkEntry(aba, placeholder_text="AIza…", show="•",
                                      height=40, corner_radius=10,
                                      fg_color=PANEL, border_color=LINE)
        self.entry_key.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(aba, text="Google CX", font=FONT_SEC,
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8, 2))
        self.entry_cx = ctk.CTkEntry(aba, placeholder_text="CX do buscador",
                                     height=40, corner_radius=10,
                                     fg_color=PANEL, border_color=LINE)
        self.entry_cx.pack(fill="x", padx=10, pady=4)
        self.btn_buscar = ctk.CTkButton(aba, text="BUSCAR NO GOOGLE",
                                        fg_color=RED, hover_color=RED_HOVER,
                                        font=("Segoe UI", 13, "bold"),
                                        corner_radius=10, height=44,
                                        command=self.buscar_online)
        self.btn_buscar.pack(fill="x", padx=10, pady=(12, 6))
        ctk.CTkLabel(aba, text="ou baixar de um link direto", font=FONT_SEC,
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8, 2))
        self.entry_url = ctk.CTkEntry(aba, placeholder_text="https://…",
                                      height=40, corner_radius=10,
                                      fg_color=PANEL, border_color=LINE)
        self.entry_url.pack(fill="x", padx=10, pady=4)
        ctk.CTkButton(aba, text="BAIXAR URL E DETECTAR", fg_color="#2a2a30",
                      hover_color="#35353c", font=("Segoe UI", 12, "bold"),
                      corner_radius=10, height=40,
                      command=self.baixar_url_detectar).pack(
                          fill="x", padx=10, pady=(4, 6))
        self.barra = ctk.CTkProgressBar(aba, fg_color=PANEL, progress_color=RED,
                                        height=8, corner_radius=8)
        self.barra.pack(fill="x", padx=10, pady=6)
        self.barra.set(0)
        self.lbl_busca = ctk.CTkLabel(aba, text="Digite o modelo e busque.",
                                      font=("Segoe UI", 11), text_color=MUTED,
                                      wraplength=280, justify="left")
        self.lbl_busca.pack(fill="x", padx=10, pady=(0, 10))

        reg_frame = ctk.CTkFrame(lateral, fg_color=CARD, corner_radius=10)
        reg_frame.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkLabel(reg_frame, text="REGIÃO DAS PLACAS", font=FONT_SEC,
                     text_color=RED_HOVER).pack(pady=(10, 2))
        self.opcao_regiao = ctk.CTkOptionMenu(
            reg_frame, values=list(REGIOES_UI),
            fg_color=PANEL, button_color=RED, button_hover_color=RED_HOVER,
            dropdown_fg_color=CARD, text_color=TEXT,
            corner_radius=10, height=36)
        self.opcao_regiao.set("Automática")
        self.opcao_regiao.pack(fill="x", padx=10, pady=(2, 6))
        ctk.CTkLabel(reg_frame, text="EXIGÊNCIA", font=FONT_SEC,
                     text_color=MUTED).pack(pady=(6, 2))
        self.opcao_rigor = ctk.CTkSegmentedButton(
            reg_frame, values=["Normal", "Rigoroso"],
            fg_color=PANEL, selected_color=RED,
            selected_hover_color=RED_HOVER, unselected_color=PANEL,
            unselected_hover_color="#2a2a30", text_color=MUTED)
        self.opcao_rigor.set("Normal")
        self.opcao_rigor.pack(fill="x", padx=10, pady=(2, 12))

        direita = ctk.CTkFrame(corpo, fg_color=PANEL, corner_radius=16,
                               border_width=1, border_color=LINE)
        direita.grid(row=0, column=1, sticky="nsew")
        direita.grid_columnconfigure(0, weight=1)
        direita.grid_rowconfigure(0, weight=1)
        self.preview = ctk.CTkLabel(direita, text="Nenhuma imagem\ncarregada",
                                    font=("Segoe UI", 15), text_color=MUTED)
        self.preview.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))

        self.paineis = ctk.CTkFrame(direita, fg_color="transparent")
        self.paineis.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.paineis.grid_columnconfigure((0, 1), weight=1)
        _, self.lbl_mini_analise = self._mini_painel(
            self.paineis, 0, "ANÁLISE DO SISTEMA")
        _, self.lbl_mini_recorte = self._mini_painel(
            self.paineis, 1, "RECORTE DA PLACA")

        self.cartao = ctk.CTkFrame(direita, fg_color=CARD, corner_radius=12,
                                   border_width=2, border_color=RED)
        self.cartao.grid(row=2, column=0, sticky="ew", padx=16, pady=8)
        ctk.CTkLabel(self.cartao, text="PLACA DETECTADA",
                     font=("Segoe UI", 11, "bold"),
                     text_color=RED_HOVER).pack(pady=(8, 0))
        self.lbl_placa = ctk.CTkLabel(self.cartao, text="— — —",
                                      font=FONT_PLATE, text_color=TEXT)
        self.lbl_placa.pack(pady=(0, 8))

        self.lbl_fonte = ctk.CTkLabel(direita, text="Fonte: —",
                                      font=("Segoe UI", 11), text_color=MUTED,
                                      wraplength=640, justify="left")
        self.lbl_fonte.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 4))
        self.log = ctk.CTkTextbox(direita, height=110, fg_color="#0e0e11",
                                  text_color=MUTED, font=("Segoe UI", 11),
                                  corner_radius=10, border_width=1,
                                  border_color=LINE)
        self.log.grid(row=4, column=0, sticky="ew", padx=16, pady=(4, 16))
        self.log.insert("end", "Sistema iniciado. Treinando classificador…\n")
        self.log.configure(state="disabled")

        threading.Thread(target=self.treinar, daemon=True).start()
        Motor.preparar_ia_async(
            lambda ok: self.after(0, lambda: self.logar(
                "IA neural pronta (leitura principal)." if ok else
                "IA neural ausente; usando sistema original. "
                "Para ativar: rode Motor com internet boa uma vez.")))
        self.protocol("WM_DELETE_WINDOW", self.fechar)

    # ---------- base ----------
    def regiao(self):
        return REGIOES_UI.get(self.opcao_regiao.get(), "auto")

    def logar(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    @staticmethod
    def _mini_painel(pai, coluna, titulo):
        moldura = ctk.CTkFrame(pai, fg_color=CARD, corner_radius=10,
                               border_width=1, border_color=LINE)
        moldura.grid(row=0, column=coluna, sticky="ew",
                     padx=(0, 8) if coluna == 0 else (8, 0))
        ctk.CTkLabel(moldura, text=titulo, font=("Segoe UI", 10, "bold"),
                     text_color=RED_HOVER).pack(pady=(6, 2))
        rotulo = ctk.CTkLabel(moldura, text="—", font=("Segoe UI", 11),
                              text_color=MUTED, height=86)
        rotulo.pack(pady=(0, 6))
        return moldura, rotulo

    def mostrar(self, img):
        self.img_atual = img
        foto = foto_ctk(img)
        if foto:
            self.preview.configure(image=foto, text="")
            self.preview.image = foto

    def mostrar_detalhes(self, recorte, analise):
        if getattr(self, "_vazio", None) is None:
            vazia = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
            self._vazio = ctk.CTkImage(light_image=vazia, dark_image=vazia,
                                       size=(4, 4))
        for img, rotulo in ((recorte, self.lbl_mini_recorte),
                            (analise, self.lbl_mini_analise)):
            if img is None:
                rotulo.configure(image=self._vazio, text="—")
                rotulo.image = self._vazio
                continue
            foto = foto_ctk(img, max_w=300, max_h=86)
            if foto:
                rotulo.configure(image=foto, text="")
                rotulo.image = foto

    def rigoroso(self):
        try:
            return self.opcao_rigor.get() == "Rigoroso"
        except Exception:
            return False

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
                "ia-cena": "lida por IA (cena)",
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
                self.logar(f"Inconclusiva ({res['certeza']}% "
                           f"< {res.get('limiar', 55)}%){mot}."
                           f"{como}{det}")
            else:
                self.logar("Nenhuma placa reconhecida nesta imagem.")

    def treinar(self):
        ok = Motor.treinar()
        self.knn_ok = ok
        if ok:
            self.dot.configure(text_color="#22c55e")
            self.lbl_pill.configure(text="Pronto")
            self.logar("Classificador treinado. Pronto.")
        else:
            self.lbl_pill.configure(text="Falha no treino")

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
            self.logar("Não foi possível ler o arquivo.")
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
            self.after(0, lambda: (
                self.mostrar_resultado(res),
                self.logar(f"Análise em {dt:.1f}s (teto 14s).")))
        except Exception as e:
            self.after(0, lambda: self.logar(f"Erro na detecção: {e}"))
        finally:
            self.after(0, self._liberar)

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
        self.after(40, self._frame)

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
            self.after(0, lambda: self._busca_fim(f"Falha na busca: {e}"))
            return
        if not itens:
            self.after(0, lambda: self._busca_fim("Sem resultados."))
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
        reg = self.regiao()
        rig = self.rigoroso()
        primeira = None
        for i, url in enumerate(urls, 1):
            dom = self._dominio(url)
            self.after(0, lambda i=i, n=len(urls), dom=dom: (
                self.lbl_busca.configure(text=f"Tentativa {i}/{n} • {dom}…"),
                self.barra.set(i / n)))
            try:
                img = Motor.baixar_imagem(url)
                if img.shape[1] < Motor.LARGURA_MINIMA:
                    continue
                if primeira is None:
                    primeira = {"url": url, "img": img}
                res = Motor.combinar(img.copy(), reg, rig)
                if res["placa"]:
                    self.after(
                        0, lambda url=url, res=res: self._busca_ok(
                            url, res))
                    return
            except Exception:
                continue
        if primeira is not None:
            self.after(0, lambda: self._busca_sem_placa(primeira))
        else:
            self.after(0, lambda: self._busca_fim("Nada pôde ser baixado."))

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
            self.after(0, lambda: self._busca_fim(f"Falha: {e}"))
            return
        try:
            res = Motor.combinar(img.copy(), self.regiao(), self.rigoroso())
        except Exception as e:
            self.after(0, lambda: self._busca_fim(f"Falha: {e}"))
            return
        self.after(0, lambda: self._busca_ok(url, res))

    def _busca_ok(self, url, res):
        self.barra.stop()
        self.barra.set(1)
        self.buscando = False
        self.btn_buscar.configure(state="normal", text="BUSCAR NO GOOGLE")
        self.lbl_busca.configure(text="Foto encontrada com placa válida.")
        dom = self._dominio(url)
        self.lbl_fonte.configure(text=f"Fonte (foto real): {dom}\n{url[:110]}")
        self.mostrar_resultado(res)
        self.logar(f"Fonte: {dom}")

    def _busca_sem_placa(self, primeira):
        self.barra.stop()
        self.barra.set(1)
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
        self.barra.set(0)
        self.lbl_busca.configure(text=msg)
        self.logar(msg)

    def fechar(self):
        self.parar_camera()
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
