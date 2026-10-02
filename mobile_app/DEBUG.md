# Códigos de diagnóstico do app (para conversa usuário ↔ dev)

Toda falha exibe um código entre colchetes no detalhe. Ao reportar,
informe: **código + versão do rodapé (vX.Y) + origem da foto**
(câmera, galeria ou aleatória).

| Código | Significado | O que fazer |
|---|---|---|
| `T00` | Treino KNN ainda rodando | Aguardar virar `PRONTO` |
| `K00` | Base KNN não carregou | Reiniciar o app; se repetir, base fora do APK |
| `D01` | Galeria/câmera/aleatória falhou no aparelho | Problema no seletor ou sem fotos; informar modelo Android |
| `L01` | Arquivo de imagem ilegível | Foto corrompida ou formato estranho |
| `T02` | Tempo esgotado (45 s) | Foto gigante ou aparelho muito lento |
| `E10` | Nada parecido com placa (certeza 10) | Foto longe/escura/borrada ou sem placa visível |
| `E40`–`E54` | Leu quase tudo, retido pela porteira | Placa difícil (1 caractere duvidoso); tentar mais perto |
| `X99-*` | Erro inesperado (`*` = tipo) | Informar o código completo |

Sucesso mostra a placa + `certeza` + `via` (leitor que acertou).

## Linha do rodapé (auto-diagnóstico no aparelho)

`v1.7 · ml:A · auto:E10 · dados:3c7ec9a8` — informe ela junto com o código.

- `ml:A` = KNN do aparelho funciona (teste sintético passou).
  `ml:ERRO-*` = cv2.ml quebrado no aparelho: nenhum leitor KNN
  funcionará, o problema é na toolchain, não nas fotos.
- `auto:<placa>` = deteção real na foto de exemplo passou.
  `auto:E<certeza>` = falhou com aquela certeza (mesma tabela acima).
  `auto:--` = o treino da etapa 2 não terminou (treino travado).
- `dados:3c7ec9a8` = arquivos de treino íntegros (MD5 do repo).
  Qualquer outro valor (ou `SEM-ARQ`) = empacotamento corrompido:
  os leitores treinam com dados errados e nada funciona.
