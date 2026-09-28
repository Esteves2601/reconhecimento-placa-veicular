# Reconhecimento de Placas — aplicativo

Segue o padrão dos apps irmãos (`detector_objetos`, `buscador_cep`):
GUI fina + motor separado + lançador `.bat`.

## Arquivos do app (novos; o sistema original não foi alterado)

| Arquivo | Papel |
|---|---|
| `app_gui.py` | Janela desktop em tkinter (igual `app_gui.py` do detector) |
| `Motor.py` | Motor: usa o sistema original como biblioteca + IA + decisão |
| `RODAR_PLACAS.bat` | Lançador com checagem e auto-instalação (igual `RODAR_CEP.bat`) |
| `requirements-app.txt` | Dependências do app |
| `icone_placa.ico` | Ícone da janela/atalho |
| `app_web.py` | Servidor web/PWA para usar no celular (igual `app_web.py` do detector) |

## Como rodar

Duplo-clique em `RODAR_PLACAS.bat` (instala dependências sozinho na
primeira vez), ou `py app_gui.py`. Atalho sem terminal: `pyw app_gui.py`.

## No celular (sem instalar APK)

```text
py app_web.py [--port 5000]
```

No celular, no mesmo Wi-Fi: `http://<IP-DO-PC>:5000` → menu do navegador
→ “Adicionar à tela inicial”. Vira um app de tela cheia, sem aviso de
“app perigoso”, pois não há instalação de APK.

## Regras de decisão (árvore, em ordem)

R1. padrão regional válido → exibe • R2. troca 1↔I/0↔O válida → exibe •
R3. voto mais votado válido → exibe • R4. consenso com acordo → exibe •
R5. KNN amplo (base 16k, k=7) com corroboração → exibe •
R6. genérica (IA confiante ou corroborada) → exibe •
R7. sem evidência: retém (“NÃO ENCONTRADA”) em vez de exibir lixo.

O modo combinado cruza Motor + pipeline original e só exibe acima da
porteira de certeza (Normal 55 / Rigoroso 80).
