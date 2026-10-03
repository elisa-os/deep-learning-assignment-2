# Relatório — Parte 5 (teste de estresse: queda de taxa de quadros)

**Reproduzir:** `uv run pa2 5` (≈6 min em CPU; retomável: reaproveita `parte5_curva.csv` e `parte5_multidt.csv`) e `uv run pytest`.
Saídas em `outputs/final_parte5/`. **Modelo:** o final (`outputs/checkpoints/final_motion_rnn.pt`: GRU 64, teacher forcing,
seed 42, treinado **só com Δt = 1**). **Detecções:** SDP congelado, mesma regra de associação das Partes 1 a 4 (guloso, IoU 0,3,
`max_age` 30, `min_hits` 3). Split: treino 02/04/05/10/11, validação 09/13 (resultado principal; o modelo final foi escolhido com
esses vídeos, então a validação é levemente favorável a ele).

## 1. Desenho
- **Subamostragem 1/k** (k = 2 e 5, além do original). Ficam os quadros com `(f−1−fase) mod k = 0`, renumerados; GT (pedestres e
  distratores) e detecções são cortados nos mesmos quadros (`pa2/stress/subsample.py`). **Cada taxa é avaliada nas k fases** e
  reportada como média ± desvio entre fases (as fases não são amostras independentes: são o mesmo vídeo com outro offset).
- **Sem retreinar.** Métodos, todos com a mesma regra de associação: caixa parada (Parte 1), velocidade constante (baseline
  simplificado: média móvel da velocidade, **não é Kalman**), **RNN final com Δt = 1** (a rede como foi treinada) e **RNN final
  alimentada com Δt = k** (o recurso Δt, que existe na rede mas valeu sempre 1 no treino).
- **Regra de morte casada no tempo** (secundária): `max_age = round(30/k)`, porque com `max_age = 30` em 1/5 a track espera 150
  quadros originais. As conclusões não mudam (§3).
- **Exploratório, fora da regra "sem retreinar":** o mesmo GRU treinado com Δt ∈ {1, 2, 5} (3 seeds), comparado nas mesmas
  taxas com os 3 checkpoints de Δt = 1 da Parte 3 (`teacher_forcing_s{42,123,456}`).
- **Verificação:** em k = 1 o pipeline reproduz exatamente a Parte 2 (validação: 0,488 / 0,534 / 0,576 para caixa parada /
  velocidade constante / RNN), e há teste de que k = 1 é idêntico à avaliação original.

## 2. Resultado principal (regra de associação inalterada)
IDF1 médio, **vídeos de validação (09 e 13)**, média ± desvio entre fases; entre parênteses, IDF1 relativo ao vídeo original:

| método | original (30 fps) | 1/2 | 1/5 |
|---|---|---|---|
| caixa parada | 0,488 | 0,436 ± 0,023 (0,89) | 0,314 ± 0,005 (0,64) |
| velocidade constante | 0,534 | 0,536 ± 0,013 (1,00) | 0,456 ± 0,003 (0,85) |
| **RNN final, Δt = 1** | **0,576** | **0,552 ± 0,015** (0,96) | 0,436 ± 0,015 (0,76) |
| RNN final, Δt = k alimentado | 0,576 | 0,541 ± 0,001 (0,94) | 0,375 ± 0,010 (0,65) |

Nos 7 vídeos (treino é in-sample para a RNN): caixa parada 0,543 / 0,509 / 0,417; velocidade constante 0,547 / 0,549 / 0,481;
RNN Δt = 1 0,595 / 0,578 / 0,474; RNN Δt = k 0,595 / 0,573 / 0,449. Com `max_age` casado (validação): 0,488 / 0,440 / 0,342
(parada), 0,534 / 0,537 / 0,455 (vel. const.), 0,576 / 0,555 / 0,440 (RNN Δt = 1), 0,576 / 0,548 / 0,388 (RNN Δt = k).

**A média esconde o que importa: por tipo de câmera** (`parte5_idf1_camera.png`, 7 vídeos, regra inalterada):

| subconjunto | método | original | 1/2 | 1/5 |
|---|---|---|---|---|
| câmera **parada** (02, 04, 09) | caixa parada | 0,557 | 0,557 | 0,508 |
| | velocidade constante | 0,529 | 0,544 | 0,565 |
| | RNN Δt = 1 | 0,612 | 0,623 | **0,593** |
| câmera **móvel** (05, 10, 11, 13) | caixa parada | 0,532 | 0,473 | 0,348 |
| | velocidade constante | 0,561 | 0,553 | **0,418** |
| | RNN Δt = 1 | 0,582 | 0,545 | 0,384 |

Com câmera parada a queda de taxa quase não machuca (a RNN perde 3% em 1/5; a velocidade constante até **melhora**); com
câmera móvel a RNN perde 34%. Por vídeo (`parte5_por_video.png`), em 1/5 a RNN fica **atrás da velocidade constante nos 4 de 4
vídeos de câmera móvel** (05: 0,424 × 0,456; 10: 0,270 × 0,307; 11: 0,532 × 0,557; 13: 0,310 × 0,354) e à frente nos 3 de 3 de
câmera parada (02, 04 e 09, este por pouco: 0,562 × 0,559). No vídeo original ela estava à frente da velocidade constante e da caixa parada em 7 de 7 vídeos.

## 3. Hipóteses escritas antes de rodar × resultado
- **H1** (todos perdem IDF1 com a queda de taxa): **parcialmente confirmada.** Caixa parada e RNN perdem; a velocidade constante
  é plana em 1/2 e perde 15% em 1/5; e nos vídeos de câmera parada o IDF1 praticamente não cai.
- **H2** (a vantagem da RNN encolhe ou some em 1/5): **depende do comparador.** Contra a **caixa parada refutada** (a vantagem
  *cresce*: +0,09 → +0,12 na validação, porque a caixa parada degrada mais). Contra a **velocidade constante confirmada**: de
  +0,04 no original para −0,02 em 1/5 na validação e −0,01 nos 7 vídeos (0,474 × 0,481), e isso só na câmera móvel (§2).
- **H3** (alimentar Δt = k a um modelo que só viu Δt = 1 não recupera): **refutada num sentido mais forte: piora.** Em 1/5 o IDF1
  de validação vai de 0,436 para 0,375 (−0,06; desvio entre fases ≈ 0,01) e nos 7 vídeos de 0,474 para 0,449.
- **H4** (o retreino multi-Δt recupera parte da perda): **não sustentada fora da amostra** (§5).
- **Mecanismo proposto** (a rede encolhe a velocidade para filtrar ruído em Δt = 1 e isso passa a subestimar em Δt maior):
  **parcialmente sustentado** (§4); não explica a diferença entre tipos de câmera.

## 4. Por que um modelo aprendido em Δt fixo quebra (diagnósticos em trajetórias do GT)
`parte5_diagnosticos.png`; CSVs `parte5_diag_*.csv`.
1. **Geometria (independe do modelo).** IoU entre as caixas do mesmo objeto em amostras consecutivas: fração < 0,3 de 0,5% (k = 1) →
   2,7% (k = 2) → **11,4%** (k = 5); mediana 0,957 → 0,922 → 0,829. A associação por IoU deixa de alcançar o objeto.
2. **O deslocamento previsto encolhe com Δt.** Regressão pela origem do deslocamento previsto sobre o verdadeiro em 1 passo
   (validação): RNN com Δt = 1: **0,81 → 0,72 → 0,47** (k = 1, 2, 5); velocidade constante: 0,94 → 0,93 → 0,86. Isto é, a rede
   subestima o movimento cada vez mais quando Δt cresce, e a velocidade constante (que extrapola o deslocamento por amostra)
   é quase invariante a Δt.
3. **Esse encolhimento é só em parte "descasamento de Δt".** O mesmo GRU treinado com Δt ∈ {1, 2, 5} tem inclinação 0,81 / 0,78 /
   **0,63** (média de 3 seeds, validação), contra 0,80 / 0,71 / 0,46 dos treinados com Δt = 1: o treino multi-Δt recupera uns
   0,17 dos ~0,35 perdidos, mas a inclinação continua bem abaixo de 1. O resto é provavelmente intrínseco (movimento menos
   previsível com amostras mais espaçadas e ruído do detector: o melhor preditor também encolhe); não testei essa explicação.
4. **O recurso Δt, sem treino, é um sinal fora de distribuição.** Trocar Δt de 1 para k muda a previsão do modelo final em mediana
   9,5% (k = 2) e **29%** (k = 5), com a mesma entrada, mas nunca foi usado: a mudança é ruído, e prejudica (H3).
5. **A queda se concentra na câmera móvel.** Não medi a causa. Hipótese: com câmera móvel o deslocamento aparente é grande e
   comum a todos os objetos, e nenhum dos métodos enxerga o movimento da câmera; em 1/5 ele é 5× maior por passo e o portão de
   IoU falha. Na câmera parada, onde os pedestres andam devagar, amostras mais espaçadas até melhoram a relação sinal/ruído
   da velocidade (a velocidade constante sobe).

## 5. Alimentar Δt resolveria?
- **Sem retreinar: não, piora** (H3; −0,06 de IDF1 em 1/5 na validação).
- **Com retreino (EXPLORATÓRIO, fora da regra "sem retreinar"; 3 seeds de cada lado, `parte5_multidt.png`):**

| vídeos | taxa | Δt = 1 no treino | Δt ∈ {1,2,5} no treino | diferença por seed |
|---|---|---|---|---|
| **validação (fora da amostra)** | original | 0,579 ± 0,003 | 0,572 ± 0,013 | −0,007 |
| | 1/2 | 0,560 ± 0,009 | 0,571 ± 0,012 | +0,011 (+0,007, +0,013, +0,013) |
| | 1/5 | 0,437 ± 0,008 | **0,435 ± 0,006** | **−0,002** (−0,004, −0,004, +0,003) |
| treino (in-sample) | 1/5 | 0,488 ± 0,004 | 0,526 ± 0,004 | **+0,038** (+0,036, +0,048, +0,032) |

  O retreino melhora o IDF1 de 1/5 **só nos vídeos em que foi treinado** (+0,038, 3 de 3 seeds) e **não** nos de validação
  (−0,002). Em 1/2 o ganho de validação é pequeno (+0,011, mesmo sinal nas 3 seeds, dentro de ~1 desvio). Ou seja: na minha
  avaliação, alimentar Δt *com treino* não resolve a degradação fora da amostra, mesmo recuperando parte da inclinação (§4.3).
  A validação tem só 2 vídeos (um parado, um móvel), então isso é evidência fraca de ausência de ganho, não prova.

## 6. Limitações
- **IDF1 não é perfeitamente comparável entre taxas:** o vídeo subamostrado tem menos quadros e oclusões mais curtas em quadros,
  e os parâmetros de associação são em quadros (daí a variante casada, que muda pouco).
- **Validação = 1 vídeo de câmera parada + 1 de câmera móvel**; o resultado por tipo de câmera usa os 7 vídeos, onde 5 são
  in-sample para a RNN (o modelo de movimento é treinado só em trajetórias do GT, não vê as detecções).
- **A velocidade constante é a versão simplificada** (não é um filtro de Kalman, não foi tunada fora da regra comum). Que ela
  ganhe da RNN em 1/5 com câmera móvel vale para esse baseline, não necessariamente para um Kalman bem ajustado.
- Taxas efetivas: o rótulo "30 fps" é de referência; o vídeo 05 tem 14 fps (→ 2,8 fps em 1/5) e o 13, 25 fps.
- As barras são entre fases (variação de offset), **não** entre seeds de treino do modelo final (há 1 seed); a comparação
  multi-Δt tem 3 seeds de cada lado.
- Mecanismos dos itens §4.3 (parte intrínseca) e §4.5 (câmera móvel) são hipóteses, não testadas.

## 7. Para a apresentação (resumo em quatro linhas)
1. Em 1/5 da taxa, a RNN final perde ~24% de IDF1 na validação; a caixa parada, 36%; a velocidade constante, 15%.
2. A perda está na **câmera móvel** (RNN −34%); com câmera parada ela é de 3%.
3. O modelo quebra porque encolhe o deslocamento cada vez mais com Δt (inclinação 0,81 → 0,47), e a velocidade constante,
   que extrapola o deslocamento observado, quase não sofre.
4. Alimentar Δt sem treino **piora** (−0,06); treinar com Δt variado melhora só dentro da amostra (+0,038) e não fora (−0,002).

## 8. EXTRA (03/10): o outro teste de estresse — qualidade do detector (sem retreinar)
Feito **depois** da queda de taxa de quadros, como extra (`pa2/part5_detector.py`, `uv run pa2 5b`; saídas `parte5b_*` em `outputs/final_parte5/`).
**Hipóteses escritas antes de rodar** (docstring do módulo):
- H1: o IDF1 cai com todas as degradações, e cai mais com falsos positivos e descarte do que com ruído.
- H2: a RNN **absorve** parte do descarte (a caixa prevista sustenta a track nos quadros sem detecção), mas não os falsos positivos.
- H3: o ruído alto prejudica a RNN mais que a caixa parada (a rede foi treinada com ruído moderado).

**Desenho:** sobre as detecções SDP (já sem distratores), 3 intensidades combinadas e os 3 fatores isolados na intensidade forte; 3 seeds de degradação; mesmo modelo final,
mesma associação. Descarte 10/25/40%; ruído **relativo ao tamanho da caixa** (desvio de 3/6/10%: left/top somam N(0, σ)·w/h, w e h multiplicam exp(N(0, σ))); falsos positivos
0,5/1,5/3 por quadro (copiam o tamanho de uma detecção real, posição uniforme, score uniforme em [0,4; 1], ou seja, passam pelo limiar). *Calibração:* eu havia pensado em ruído de 20%,
mas num teste rápido no vídeo 09 isso derrubava o AP50 de 0,64 a 0,17 (colapso do detector, não um teste útil); fiquei com 10% (AP50 0,57). O mAP e o F1 do detector são medidos nas detecções degradadas.

**Resultado (vídeos de validação, média ± desvio de 3 seeds; IDF1 relativo ao original entre parênteses):**

| condição | mAP | F1 det. | caixa parada | velocidade constante | **RNN final** |
|---|---|---|---|---|---|
| original | 0,385 | 0,740 | 0,488 | 0,534 | **0,576** |
| leve | 0,311 | 0,676 | 0,432 (0,88) | 0,470 (0,88) | **0,525** (0,91) |
| média | 0,200 | 0,563 | 0,296 (0,61) | 0,258 (0,48) | **0,407** (0,71) |
| forte | 0,096 | 0,410 | 0,175 (0,36) | 0,079 (0,15) | **0,207** (0,36) |
| só descarte (40%) | 0,234 | 0,529 | 0,291 (0,60) | 0,313 (0,59) | **0,374** (0,65) |
| só ruído (10%) | 0,160 | 0,672 | 0,274 (0,56) | 0,175 (0,33) | **0,371** (0,64) |
| só falsos positivos (3/q) | 0,382 | 0,647 | 0,479 (0,98) | 0,518 (0,97) | **0,558** (0,97) |

- **H1: refutada na parte "mais com falsos positivos".** Os falsos positivos quase **não** machucam (IDF1 −3% nos três métodos): aleatórios no espaço, raramente aparecem 3 quadros seguidos no mesmo lugar, então
  o `min_hits = 3` os filtra. O que mais machuca é o **ruído** nas caixas (RNN −36%, caixa parada −44%, velocidade constante −67%) e o **descarte** (−35%, −40%, −41%).
  Atenção: o **mAP não vê os falsos positivos** (0,385 → 0,382), porque o SDP tem scores saturados em ~1 e os falsos positivos ficam ranqueados abaixo dos verdadeiros; só a precisão no limiar de operação cai (0,99 → 0,68 no vídeo 09). Por isso reporto também o F1.
- **H2: parcialmente confirmada.** A RNN perde **menos** que os outros no descarte (−35% × −40% da caixa parada e −41% da velocidade constante) e na condição média (−29% × −40% e −52%); na condição forte
  ela empata em termos relativos com a caixa parada (−64% nos dois; em valor absoluto 0,207 × 0,175). Ela **absorve** parte da falha, não a amplifica, e não a elimina. Sobre os falsos positivos, a previsão é "absorvida" por outro motivo: o `min_hits` já os filtra.
- **H3: refutada.** A RNN **não** é a mais frágil ao ruído: perde menos que a caixa parada (−36% × −44%); a **velocidade constante** é a que quebra (−67%: o ruído vira velocidade espúria).
- **Resposta à pergunta do enunciado:** o modelo temporal **absorve parte, não amplifica**. A vantagem de IDF1 da RNN sobre a caixa parada é de +0,09 (original), +0,09 (leve) e +0,11 (média), e cai a **+0,03 na condição forte**; sobre a velocidade constante ela cresce com a degradação (+0,04 → +0,13 na forte). O IDF1 de todos acompanha o mAP/F1 do detector (painel mAP × IDF1 da figura): o modelo temporal desloca a curva para cima nas degradações moderadas e perde o efeito quando o detector entrega quase nada.
- **Ressalvas:** os falsos positivos aleatórios são um caso fácil (um detector real erra em lugares plausíveis e persiste por vários quadros); degradação sintética sobre o SDP, não um detector pior de verdade (veja o Faster R-CNN COCO na Parte 1); 3 seeds; só validação (09 e 13) no texto, os 7 vídeos em `parte5b_detector_resumo_todos.csv`.

