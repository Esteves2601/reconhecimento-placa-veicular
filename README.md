# Reconhecimento de Placas Veiculares

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![EasyOCR](https://img.shields.io/badge/EasyOCR-PT%2BEN-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?style=flat-square)

Sistema desktop de reconhecimento de placas veiculares com interface
moderna em preto e vermelho. Projeto acadêmico de Processamento Digital
de Imagens (UniAteneu).

## Origem

Fork de [franklindias/reconhecimento-placa-veicular](https://github.com/franklindias/reconhecimento-placa-veicular)
(2016). A base original é o pipeline clássico de PDI: detecção da placa
na cena (`DetectarPlacas`), segmentação e leitura de caracteres por KNN
(`DetectarCaracteres`, 180 amostras), com seleção de imagem via diálogo
Tkinter. Esses arquivos foram mantidos intactos e são usados como
biblioteca.

## O que mudou neste fork

**Motor de decisão (`Motor.py`, novo):**
- Árvore de decisão em 7 regras ordenadas por força de evidência
- Leitor neural EasyOCR (PT+EN) em 4 variantes do recorte
- Validação por padrão regional: Brasil (antigo e Mercosul), Argentina,
  Uruguai, Paraguai e internacional
- Consenso posicional entre leitores e porteiro de certeza (55/80):
  sem evidência, o sistema declara "não encontrada" em vez de inventar
- Leitura "provável" exibida e sinalizada quando retida pela porteira
- Parada antecipada ao confirmar leitura regional (economia de varreduras)

**Interface (`Interface.py`, nova):**
- App desktop em CustomTkinter (tema grafite + vermelho): abas de
  imagem, câmera ao vivo e busca de fotos, seletor de região e de rigor
- Painéis de análise (binarização) e recorte da placa, cartão de
  resultado com certeza e log de decisões

**Medido, não prometido:** baterias de 100 casos com gabarito
(`test_100.py`, `test_suite20.py`) — 65/100 o motor completo, 45/100 o
pipeline original.

## Como funciona

Leitores independentes votam em cada placa e um porteiro de certeza
só exibe o resultado com evidência suficiente — o sistema prefere dizer
"não encontrada" a inventar uma leitura:

- **KNN original** (k=1, 180 amostras reais) sobre a imagem binarizada
- **EasyOCR** (neural, PT+EN) em 4 variantes do recorte
- **Consenso posicional** e validação por padrão regional
  (Brasil antigo e Mercosul, Argentina, Uruguai, Paraguai e internacional)

## Interface

- `Interface.py` — app principal (CustomTkinter): imagem, câmera ao vivo,
  busca de fotos na web, seletor de região, painéis de análise e recorte
- `app_gui.py` — versão alternativa em Tkinter padrão
- `app_web.py` — interface web local (Flask) com PWA

## Como executar

```bat
RODAR_PLACAS.bat
```

Ou manual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-app.txt
python Interface.py
```

Na primeira execução com IA, os modelos do EasyOCR (~100 MB) são
baixados automaticamente.

## Precisão medida (bateria de 100 casos, 16 fotos × 5 + 4 variações)

| Modo | Acertos |
|---|---|
| Motor completo (KNN + IA) | 65/100 |
| Pipeline KNN original | 45/100 |

Erros típicos: 1 caractere em fonte estilizada (`L0LWATT`,
`NITESIY`). Placas Mercosul reais com fonte padrão passam na conta
(regional + confirmação do original + conclusão = 90 pontos, limiar 55).

## Estrutura

| Arquivo | Papel |
|---|---|
| `Main.py`, `DetectarPlacas.py`, `DetectarCaracteres.py`, `Preprocesso.py` | Pipeline original (intocado) |
| `Motor.py` | Engine: decisão, consenso, certeza, IA |
| `Interface.py` | App desktop principal |
| `app_gui.py` / `app_web.py` | Interfaces alternativa e web |
| `imagens/` | Fotos de teste com gabarito |
| `test_100.py`, `test_suite20.py` | Baterias de teste |

## Notas

- Leitura "provável" abaixo do limiar é exibida sinalizada, nunca como
  certeza.
- Sem modelos da IA, o sistema opera só com KNN (mais rápido, menos
  preciso em fontes fora do padrão).
