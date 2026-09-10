# Main.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import random
import tkinter as tk
from tkinter import filedialog

import DetectarCaracteres
import DetectarPlacas


#cores
ESCALA_PRETO = (0.0, 0.0, 0.0)
ESCALA_BRANCO = (255.0, 255.0, 255.0)
ESCALA_AMARELO = (0.0, 255.0, 255.0)
ESCALA_VERDE = (0.0, 255.0, 0.0)
ESCALA_VERMELHO = (0.0, 0.0, 255.0)

mostrarPassos = False


#listar_imagens
def listarImagens():

    pasta = "imagens"

    extensoes = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if not os.path.exists(pasta):
        return []

    imagens = []

    for arquivo in os.listdir(pasta):

        if arquivo.lower().endswith(extensoes):
            imagens.append(os.path.join(pasta, arquivo))

    return imagens


#imagem_aleatoria
def escolherImagemAleatoria(imagemAtual=None):

    imagens = listarImagens()

    if len(imagens) == 0:
        return None

    #evitar_repeticao
    if imagemAtual is not None and len(imagens) > 1:

        imagensDisponiveis = [
            imagem for imagem in imagens
            if imagem != imagemAtual
        ]

        return random.choice(imagensDisponiveis)

    return random.choice(imagens)


#selecionar_imagem
def selecionarImagem():

    root = tk.Tk()

    root.withdraw()

    root.attributes("-topmost", True)

    caminho = filedialog.askopenfilename(
        title="Escolha uma imagem",
        filetypes=[
            ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Todos os arquivos", "*.*")
        ]
    )

    root.destroy()

    if caminho == "":
        return None

    return caminho


#processar_imagem
#processar_imagem
def processarImagem(caminhoImagem):

    cv2.destroyAllWindows()

    #ler_imagem
    try:
        dados = np.fromfile(caminhoImagem, dtype=np.uint8)
        imgCenaOriginal = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    except Exception:
        imgCenaOriginal = None

    if imgCenaOriginal is None:

        print("\nErro: Arquivo de imagem não lido\n")

        return False

    print("\n----------------------------------------")
    print("Imagem:", os.path.basename(caminhoImagem))
    print("----------------------------------------")

    #detectar_placas
    listaDePossiveisPlacas = DetectarPlacas.DetectarPlacasInScene(
        imgCenaOriginal
    )

    #detectar_caracteres
    listaDePossiveisPlacas = DetectarCaracteres.DetectarCaracteresNasPlacas(
        listaDePossiveisPlacas
    )

    if len(listaDePossiveisPlacas) == 0:

        print("\nNenhuma placa foi encontrada\n")

    else:

        #ordenar_placas
        listaDePossiveisPlacas.sort(
            key=lambda possivelPlaca: len(possivelPlaca.strCaracteres),
            reverse=True
        )

        licPlaca = listaDePossiveisPlacas[0]

        if licPlaca.imgPlaca is not None:
            cv2.imshow("Placa detectada", licPlaca.imgPlaca)

        if licPlaca.imgThreshold is not None:
            cv2.imshow("Threshold", licPlaca.imgThreshold)

        if len(licPlaca.strCaracteres) == 0:

            print("\nNenhum caractere foi encontrado\n")

        else:

            desenharRetanguloVermelhoAoRedorDaPlaca(
                imgCenaOriginal,
                licPlaca
            )

            print(
                "\nPlaca lida da imagem = "
                + licPlaca.strCaracteres
                + "\n"
            )

            escreverCaracteresDaPlacaNaImagem(
                imgCenaOriginal,
                licPlaca
            )

            #salvar_resultado
            cv2.imwrite(
                "imgCenaOriginal.png",
                imgCenaOriginal
            )

    #instrucoes
    cv2.putText(
        imgCenaOriginal,
        "N: aleatoria | A: escolher imagem | Q/ESC: sair",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        ESCALA_VERDE,
        2
    )

    cv2.imshow(
        "Reconhecimento de Placa",
        imgCenaOriginal
    )

    return True


#programa_principal
def main():

    #treinar_knn
    blnKNNTrainingSuccessful = (
        DetectarCaracteres.loadKNNDataAndTrainKNN()
    )

    if blnKNNTrainingSuccessful == False:

        print(
            "\nErro: treinamento KNN não foi realizado com sucesso\n"
        )

        return

    imagens = listarImagens()

    if len(imagens) == 0:

        print(
            "\nErro: nenhuma imagem encontrada na pasta 'imagens'\n"
        )

        return

    #primeira_imagem
    imagemAtual = escolherImagemAleatoria()

    print("\nSistema de Reconhecimento de Placas")
    print("----------------------------------------")
    print("N - Nova imagem aleatória")
    print("A - Abrir imagem do computador")
    print("Q - Sair")
    print("ESC - Sair")
    print("----------------------------------------")

    processarImagem(imagemAtual)

    #loop_principal
    while True:

        tecla = cv2.waitKey(0) & 0xFF

        #nova_aleatoria
        if tecla == ord("n") or tecla == ord("N"):

            novaImagem = escolherImagemAleatoria(imagemAtual)

            if novaImagem is not None:

                imagemAtual = novaImagem

                processarImagem(imagemAtual)

        #abrir_imagem
        elif tecla == ord("a") or tecla == ord("A"):

            novaImagem = selecionarImagem()

            if novaImagem is not None:

                imagemAtual = novaImagem

                processarImagem(imagemAtual)

        #sair
        elif tecla == ord("q") or tecla == ord("Q") or tecla == 27:

            break

    cv2.destroyAllWindows()


#desenhar_retangulo
def desenharRetanguloVermelhoAoRedorDaPlaca(
        imgCenaOriginal,
        licPlaca):

    p2fRectPoints = cv2.boxPoints(
        licPlaca.rrLocationOfPlacaInScene
    )

    #converter_coordenadas
    p2fRectPoints = np.int32(p2fRectPoints)

    cv2.line(
        imgCenaOriginal,
        tuple(p2fRectPoints[0]),
        tuple(p2fRectPoints[1]),
        ESCALA_VERMELHO,
        2
    )

    cv2.line(
        imgCenaOriginal,
        tuple(p2fRectPoints[1]),
        tuple(p2fRectPoints[2]),
        ESCALA_VERMELHO,
        2
    )

    cv2.line(
        imgCenaOriginal,
        tuple(p2fRectPoints[2]),
        tuple(p2fRectPoints[3]),
        ESCALA_VERMELHO,
        2
    )

    cv2.line(
        imgCenaOriginal,
        tuple(p2fRectPoints[3]),
        tuple(p2fRectPoints[0]),
        ESCALA_VERMELHO,
        2
    )


#escrever_caracteres
def escreverCaracteresDaPlacaNaImagem(
        imgCenaOriginal,
        licPlaca):

    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = (
        imgCenaOriginal.shape
    )

    PlacaHeight, PlacaWidth, PlacaNumChannels = (
        licPlaca.imgPlaca.shape
    )

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX

    fltFontScale = float(PlacaHeight) / 30.0

    intFontThickness = int(
        round(fltFontScale * 1.5)
    )

    textSize, baseline = cv2.getTextSize(
        licPlaca.strCaracteres,
        intFontFace,
        fltFontScale,
        intFontThickness
    )

    (
        (intPlacaCenterX, intPlacaCenterY),
        (intPlacaWidth, intPlacaHeight),
        fltCorrectionAngleInDeg
    ) = licPlaca.rrLocationOfPlacaInScene

    intPlacaCenterX = int(intPlacaCenterX)
    intPlacaCenterY = int(intPlacaCenterY)

    ptCenterOfTextAreaX = int(
        intPlacaCenterX
    )

    #posicao_texto
    if intPlacaCenterY < (sceneHeight * 0.75):

        ptCenterOfTextAreaY = (
            int(round(intPlacaCenterY))
            + int(round(PlacaHeight * 1.6))
        )

    else:

        ptCenterOfTextAreaY = (
            int(round(intPlacaCenterY))
            - int(round(PlacaHeight * 1.6))
        )

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX
        - (textSizeWidth / 2)
    )

    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY
        + (textSizeHeight / 2)
    )

    cv2.putText(
        imgCenaOriginal,
        licPlaca.strCaracteres,
        (
            ptLowerLeftTextOriginX,
            ptLowerLeftTextOriginY
        ),
        intFontFace,
        fltFontScale,
        ESCALA_AMARELO,
        intFontThickness
    )


#iniciar_programa
if __name__ == "__main__":
    main()