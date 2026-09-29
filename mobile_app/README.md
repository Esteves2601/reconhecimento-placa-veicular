# App Android do Reconhecimento de Placas (90% sem pagar a Play)

UI Kivy (preto/vermelho) que usa o `Motor.py` **original como biblioteca**.
Nada da raiz é modificado: no build, os arquivos do motor são **copiados**
para cá (ignorado pelo git, ver `.gitignore`).

## Limitações honestas desta versão mobile

- Somente caminho KNN + consenso (`Motor.combinar`): EasyOCR não tem
  receita no python-for-android, então sem IA neural no celular.
- tkinter não existe no Android: `main.py` registra um stub antes de
  importar o Motor (originais intactos).

## 1. Copiar o motor (toda vez que a raiz mudar)

Na raiz do projeto:

```bash
cp Motor.py Main.py DetectarPlacas.py DetectarCaracteres.py Preprocesso.py \
   PossivelPlaca.py PossivelCaractere.py classifications.txt \
   flattened_images.txt base_kNN_ampla.npy classes_kNN_ampla.npy mobile_app/
```

## 2. Build (Colab/Linux — buildozer não roda no Windows)

```bash
cd mobile_app
buildozer android debug      # valida: api 34, so permissao CAMERA
# release: descomente as linhas android.release_* no buildozer.spec,
# preencha com o dev-release.p12 (alias appkey), e rode:
buildozer android release
```

## 3. Publicar e instalar

Distribuição por link HTTPS + SHA-256 e instalação limpa (desinstalar
versões antigas primeiro): procedimento completo em
`detector_objetos/GUIA_APK_LIMPO.md` (vale igual para este app).
