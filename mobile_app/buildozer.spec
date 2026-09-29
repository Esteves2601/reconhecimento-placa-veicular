[app]

# (str) Titulo do aplicativo
title = Reconhecimento de Placas

# (str) Nome do pacote (minusculas, sem espacos)
package.name = placasveicular

# (str) Dominio do pacote (gera org.reconhecimento.placas; NAO mudar depois
# de publicar: trocar o pacote orfa quem ja instalou)
package.domain = org.reconhecimento

# (str) Versao do app
version = 1.0

# (int) Versionamento interno: SEMPRE crescente (1, 2, 3...). Reinstalar por
# cima com versionCode igual ou menor = "app nao instalado". Pacote novo:
# comeca em 1.
android.versionCode = 1

# (str) Nome de versao exibido
android.versionName = 1.0

# (str) Diretorio do codigo-fonte (o app e autocontido aqui; os arquivos do
# Motor sao COPIADOS para ca no build, ver README.md)
source.dir = .

# (list) Extensoes incluidas (txt+npy = base KNN; sem eles o app nao detecta)
source.include_exts = py,png,jpg,jpeg,txt,npy

# (list) Arquivos fora do APK
source.exclude_patterns = README.md,.gitignore,*.spec

# (list) Requisitos Python (receitas p4a). Sem tkinter/customtkinter: o
# main.py usa stub de tkinter porque essas UIs nao existem no Android.
requirements = python3,kivy,plyer,numpy,opencv,requests

# (str) Orientacao de tela
orientation = portrait

# (bool) Tela cheia
fullscreen = 0

# (list) Permissoes: MINIMO necessario. CAMERA para fotografar; a galeria
# usa o seletor do sistema (SAF) e NAO precisa de permissao de midia.
# NUNCA adicionar READ_EXTERNAL_STORAGE/READ_MEDIA_* (restritas, geram
# aviso extra no Play Protect).
android.permissions = CAMERA

# (int) API Android: 34 (exigencia da Play desde ago/2024; api 33 ou menor
# = aviso/bloqueio). NDK travado para build reproduzivel.
android.api = 34
android.minapi = 24
android.ndk = 25b
android.ndk_api = 21

# (str) Branch python-for-android (master = toolchain atual)
p4a.branch = master

# Aceita licencas do SDK sem prompt (build nao interativo no Colab)
android.accept_sdk_license = True

# --- ASSINATURA RELEASE (os 90%: chave propria, sem pagar a Play) ---
# Na hora do build release, descomente e preencha com o keystore gerado
# (dev-release.p12, alias appkey). NUNCA commitar senhas neste arquivo.
#android.release_artifact = aab
#android.release_keystore = /caminho/para/dev-release.p12
#android.release_keystore_passwd =
#android.release_keyalias = appkey
#android.release_keyalias_passwd =

[buildozer]

# (int) Nivel de log (2 = info)
log_level = 2

# (str) Buildozer nao baixa o SDK sozinho no Colab sem isso em alguns setups
warn_on_root = 1
