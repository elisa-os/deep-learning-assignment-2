# Relatório — Parte 1 (baseline por quadro)

**Reproduzir:** `uv run pa2 1` (≈90 s) e `uv run pytest`. Saídas em `outputs/parte1_*`.
**Dados:** só o pacote de anotações (~10 MB). O detector torchvision **ainda não foi rodado**
(precisa das imagens, ~5,5 GB, e de GPU); ver §6.

## 1. Os vídeos e o split

Split por **vídeo**: os três detectores públicos compartilham vídeo e GT, então separar por
detector vazaria o mesmo vídeo entre treino e validação. O `test/` do MOT17 não traz GT, logo
validação e teste saem de `train/`.

| vídeo | split | câmera | res. / fps | ped/quadro | visib. média | ids GT |
|---|---|---|---|---|---|---|
| 05 | treino | móvel | 640×480 / 14 | 8,3 | 0,52 | 133 |
| 11 | treino | móvel | 1080p / 30 | 10,5 | 0,64 | 75 |
| 10 | treino | móvel | 1080p / 30 | 19,6 | 0,69 | 57 |
| 02 | treino | parada | 1080p / 30 | 31,0 | 0,40 | 62 |
| 04 | treino | parada | 1080p / 30 | 45,3 | 0,62 | 83 |
| 09 | **val** | parada | 1080p / 30 | 10,1 | 0,59 | 26 |
| 13 | **val** | móvel | 1080p / 25 | 15,5 | 0,67 | 110 |

Critério: a validação tem uma câmera parada e uma móvel, densidades e fps diferentes, e o
vídeo 13 tem a maior rotatividade de identidades. A câmera de cada vídeo vem do artigo do
MOT16/17 (não foi estimada das imagens).

## 2. Convenções de avaliação (iguais às do benchmark)
- GT avaliado: `class == 1` e `conf == 1`. O GT do MOT17 é **contínuo**: o pedestre ocluído
  continua em todos os quadros da sua vida, com `visibility` baixa (não há lacunas no `gt.txt`).
- Detecções/tracks que casam com classes distratoras (2, 7, 8, 12: pessoa em veículo, pessoa
  estática, distrator, reflexo) e não com um pedestre são removidas antes de medir.
- IDF1, ID switches e fragmentações: `pa2/metrics/tracking.py` (Parte 0). AP/mAP por quadro:
  `pa2/metrics/detection.py` (101 pontos de recall; mAP = média sobre IoU 0,50:0,95; empates de
  score tratados como um único limiar, já que FRCNN e SDP saturam em 1,0). Limiar de IoU das
  métricas: 0,5. Médias entre vídeos são simples (sem ponderar por tamanho).
- Verificado nos dados reais (`tests/test_mot17.py`): GT como predição ⇒ IDF1 = 1, 0 switches,
  0 fragmentações; GT como detecções ⇒ AP50 = 1; tracks sobre distratores não mudam o IDF1.

## 3. Detector público escolhido: **SDP**
Comparação só nos vídeos de **treino**, com a mesma associação (Hungarian, IoU 0,3,
`max_age` 5, `min_hits` 1) e, para cada detector, o limiar de score que maximiza o F1 somado:

| detector | limiar | AP50 | mAP | precisão | recall | F1 | IDF1 | ids prev./verd. | switches/id |
|---|---|---|---|---|---|---|---|---|---|
| DPM | −0,295 | 0,414 | 0,211 | 0,737 | 0,410 | 0,527 | 0,295 | 8,61 | 3,81 |
| FRCNN | 0,050 | 0,529 | 0,391 | 0,957 | 0,527 | 0,680 | 0,486 | 3,05 | 2,41 |
| **SDP** | 0,400 | **0,670** | **0,437** | **0,966** | **0,694** | **0,808** | **0,535** | 3,88 | 3,99 |

O SDP é o melhor em AP50, mAP, precisão, recall, F1 e IDF1. Ressalva: é também o que gera mais
ID switches por identidade, porque detecta mais objetos pequenos e parcialmente ocluídos
(mais tracks nascendo e morrendo). A fonte de detecções do resto do PA é o SDP, score ≥ 0,4.

## 4. Regra de associação e gestão de tracks (`pa2/association/tracker.py`)
- **Associação:** IoU entre a *última caixa observada* de cada track e as detecções do quadro;
  casamento `greedy` (pares por IoU decrescente) ou `hungarian` (maximiza a soma de IoU); só
  vale IoU ≥ limiar fixo. Não há modelo de movimento: a caixa da track não se move enquanto
  espera.
- **Nascimento:** detecção sem par vira track tentativa; ganha id novo ao acumular `min_hits`
  observações consecutivas (tentativa que falha um quadro é descartada). Com `min_hits = 3` as
  duas primeiras observações de cada track não aparecem na saída.
- **Morte:** track confirmada sem observação por mais de `max_age` quadros morre; ids nunca
  são reaproveitados. Só as tracks observadas no quadro entram na saída (sem extrapolar caixas).

Variantes avaliadas (guloso/Hungarian × IoU {0,3; 0,5} × `max_age` {1; 5; 30} × `min_hits`
{1; 3} = 24, no SDP), média dos 5 vídeos de treino (`parte1_association_variants.csv`):

| fator | IDF1 treino | switches treino (soma média/vídeo) |
|---|---|---|
| guloso / Hungarian | 0,527 / 0,526 | 336 / 334 |
| IoU 0,3 / 0,5 | 0,540 / 0,513 | 272 / 397 |
| `max_age` 1 / 5 / 30 | 0,509 / 0,527 / 0,544 | 324 / 291 / 389 |
| `min_hits` 1 / 3 | 0,521 / 0,532 | 426 / 244 |

"Regras diferentes dão números diferentes": o IoU mais baixo e `min_hits = 3` ajudam em todas as
métricas; `max_age` maior sobe o IDF1 mas também os switches; guloso vs. Hungarian é indiferente
(a diferença é menor que o ruído entre vídeos).

**Escolha** (regra fixa no código, no treino): maior IDF1; entre as variantes a até 0,005 do
máximo, a de menos ID switches. Resultado: **guloso, IoU ≥ 0,3, `max_age` 30, `min_hits` 3**
(IDF1 treino 0,564; validação 0,488). As 5 primeiras variantes do ranking diferem pouco entre si.

## 5. Resultado por vídeo e o descolamento (`parte1_descolamento.png`)

| vídeo | split | mAP | AP50 | F1 det. | **IDF1** | ID sw | frag | ids prev./verd. | sw/id | erro de contagem |
|---|---|---|---|---|---|---|---|---|---|---|
| 05 | treino | 0,437 | 0,642 | 0,770 | 0,603 | 150 | 216 | 137/133 | 1,13 | +4 |
| 11 | treino | 0,519 | 0,740 | 0,835 | 0,634 | 99 | 199 | 115/75 | 1,32 | +40 |
| 09 | val | 0,457 | 0,643 | 0,784 | 0,527 | 57 | 124 | 36/26 | 2,19 | +10 |
| 13 | val | 0,313 | 0,547 | 0,696 | 0,449 | 408 | 336 | 265/110 | 3,71 | +155 |
| 10 | treino | 0,422 | 0,727 | 0,822 | 0,441 | 459 | 441 | 194/57 | 8,05 | +137 |
| 02 | treino | 0,283 | 0,479 | 0,632 | 0,433 | 231 | 671 | 128/62 | 3,73 | +66 |
| 04 | treino | 0,521 | 0,762 | 0,863 | 0,711 | 274 | 761 | 114/83 | 3,30 | +31 |

Médias — treino: mAP 0,437, IDF1 0,564, 1,88 ids previstos por id verdadeiro, 3,5 switches/id;
validação: mAP 0,385, IDF1 0,488, 1,90, 2,95.

O que o gráfico mostra: a **detecção** (mAP, AP50) varia pouco entre vídeos, mas as
**identidades** se desfazem de forma desigual. O F1 da detecção funciona como teto do IDF1
(mesmo com identidades perfeitas ele não passa de ~0,63–0,86 nestes vídeos); o IDF1 fica bem
abaixo dele, e a diferença é a perda atribuível à associação. No vídeo 10 o detector é bom
(F1 0,82) e o IDF1 é 0,44, com 3,4 ids previstos por id verdadeiro e 8 switches por id.

Eixo de dificuldade: ordenei por **densidade** (pedestres/quadro). A correlação de Spearman
com n = 7 é só indicativa: densidade ↔ switches/id = 0,71; densidade ↔ IDF1 = −0,14; câmera
móvel ↔ IDF1 = 0,00. Ou seja, a densidade explica razoavelmente os switches, mas **não ordena
o IDF1**: o pior caso de identidade (vídeo 10) tem densidade média e câmera móvel. Com 7
vídeos não dá para separar bem densidade, movimento de câmera e oclusão; o gráfico rotula
câmera e densidade em cada vídeo para isso ficar visível.

## 6. Pendências e limitações
- **Detector torchvision não rodado.** O código está pronto e testado com modelo falso
  (`tests/test_torchvision_detector.py`): classe `person`, NMS interno desligado e substituído
  pelo nosso, cache em `outputs/detections/torchvision/`. Os pesos COCO carregam e uma
  inferência numa imagem sintética funcionou. Falta baixar as imagens do MOT17 (~5,5 GB) e,
  em CPU, o custo é de segundos por quadro; com GPU, minutos para os ~5300 quadros de treino.
  Ligue com `use_torchvision_detector: true`; a etapa gera `parte1_per_sequence_torchvision.csv`
  e `parte1_descolamento_torchvision.png`.
- Fine-tune do detector: não feito (opcional no enunciado).
- A escolha do detector e da associação usa só o treino, mas são 5 vídeos e as diferenças
  entre as melhores variantes são pequenas.
- O mAP aqui é de detecção por quadro sobre todas as detecções públicas; precisão/recall/F1
  são no limiar de operação (score ≥ 0,4).
- `min_hits = 3` descarta as duas primeiras observações de cada track (cria FN), o que as
  métricas de detecção da Parte 1 não capturam.
