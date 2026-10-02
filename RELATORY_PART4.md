# Relatório — Parte 4 (horizonte de memória, galeria de falhas, correção)

**Reproduzir:** `uv run pa2 4` (≈40 min em CPU do zero; a etapa D é retomável e usa `parte4_correcao.csv` se existir) e `uv run pytest`. Saídas em `outputs/final_parte4/`.
**Verificado (revisão de 01/10):** etapa A reproduzida do zero com CSV idêntico; cálculo do gradiente conferido contra
diferenças finitas; números deste relatório recomputados a partir dos CSVs. A etapa B (oclusões injetadas) **não foi
reproduzida do zero** (o vídeo 04 leva > 1 h); só recomputada a partir de `parte4_oclusoes_injetadas.csv`.
**Modelo:** o final (`outputs/checkpoints/final_motion_rnn.pt`: GRU 64, teacher forcing, seed 42; ver
`RELATORY_PART2.md` §6). **Detecções:** SDP congelado, mesma regra de associação das Partes 1 a 3.
**Sem as imagens do MOT17** (só anotações): as figuras desenham as caixas sobre fundo vazio.

## 1. Horizonte de memória analítico (`parte4_gradiente.png`, `parte4_gradiente.csv`)
`||∂L_t/∂h_{t−k}||` por autograd na janela desenrolada de 48 passos (perda smooth-L1 do treino no último
passo), mediana de 400 janelas dos vídeos de validação (09 e 13), com o ruído de detector do treino. Duas
situações: **com observações** (inferência normal) e **às cegas** (depois de 8 quadros observados, a rede só
recebe as próprias previsões, que é o que acontece numa oclusão).

| modelo | k em que cai a 1/2 | a 1/10 | a 1/100 | razão em k = 31 |
|---|---|---|---|---|
| **final (teacher forcing)** | 3 | **13** | 39 | 0,018 |
| receita da Parte 2 (com buracos) | 5 | 20 | 46 | 0,039 |
| scheduled sampling | 3 | 18 | 47 | 0,042 |
| free-running | 3 | 11 | 32 | 0,010 |

- **Com observações o gradiente some**, como nos slides: 5× em 8 passos, 10× em 13, 20× em 20, ~60× na borda da
  janela de BPTT (T = 32). Isto é, o sinal de supervisão vindo de mais de ~13 quadros atrás é menos de 10% do de
  um quadro atrás. Treinar com buracos simulados alarga o horizonte (13 → 20), o que bate com a intuição; o
  free-running tem o mais curto.
- **Às cegas o gradiente não some, cresce**: 8× em k = 31 no modelo final (3,9× a 10,5× nos outros). Leitura: o
  caminho previsão → próxima entrada transforma o estado em algo que se integra (um erro de velocidade em
  k passos atrás vira erro de posição proporcional a k), então o efeito de um erro cedo no buraco **aumenta**. É
  uma hipótese de mecanismo, não testada; o que está medido é o crescimento. Ele não significa "memória útil".
  (A queda da curva às cegas depois de k ≈ 40 é efeito de borda: esses estados são de antes do início do buraco,
  nos 8 quadros de contexto observados.) O cálculo foi conferido contra diferenças finitas em dupla precisão
  (mesmos valores até 6 dígitos, nos dois modos).

## 2. Horizonte empírico (`parte4_sobrevivencia.png`, `parte4_sobrevivencia.csv`)
**Experimento controlado.** Em cada vídeo (7, treino e validação juntos) escolho uma identidade com detecção em
5 quadros antes e depois, **removo as suas detecções por N quadros** e olho a detecção que reaparece: recebe o
**mesmo id** da track de antes (`kept`), o id de **outra track** que já existia (`swapped`) ou um **id novo**
(`reborn`)? `max_age = 90`, maior que o maior N, então o que limita é a previsão e não a regra de morte. N = 0 é
o controle (deve dar ~1). ≈330 tentativas por N (4 execuções × até 12 oclusões não sobrepostas × 7 vídeos).
Mesmas tentativas para os três métodos.

| N (quadros) | 0 | 5 | 10 | 15 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|---|
| caixa parada (Parte 1) | 0,98 | 0,79 | 0,60 | 0,51 | 0,44 | 0,38 | 0,35 | 0,26 |
| velocidade constante | 1,00 | 0,83 | 0,55 | 0,37 | 0,22 | 0,12 | 0,09 | 0,03 |
| **RNN final** | 0,99 | **0,91** | **0,76** | **0,62** | **0,51** | 0,43 | 0,32 | 0,23 |

(IC 95% de Wilson de ±0,05 a ±0,06 nos pontos da RNN em N de 10 a 30.) **N50** (N em que metade volta com o
mesmo id, `parte4_n50_bootstrap.csv`): **RNN 20,9 [18,6; 24,1]**, caixa parada 15,8 [13,3; 18,5], velocidade
constante 11,4 [10,0; 12,4] (IC 95% por bootstrap das tentativas). A vantagem da RNN é real mas moderada: +5,2
quadros sobre a caixa parada [+1,3; +8,9] e +9,6 sobre a velocidade constante [+7,0; +12,8]. Os intervalos cobrem
só a amostragem das oclusões (tentativas do mesmo vídeo são correlacionadas, então são otimistas) e não a
variação entre seeds de treino. Acima de ~40 quadros a RNN e a caixa parada
são indistinguíveis (a velocidade constante colapsa: extrapola em linha reta por tempo demais). Quando falha,
a RNN quase sempre **renasce com id novo** (N = 20: 43% `reborn`, 5% `swapped`): o modo de falha de oclusão
longa é **fragmentação**, não roubo de identidade. (`swapped` só conta ids de tracks observadas no quadro
anterior à oclusão; uma track vizinha que estivesse esperando conta como `reborn`, então os 5% são um piso.)

**Comparação com o dataset** (`parte4_horizonte_empirico.json`). Dois jeitos de medir a duração das oclusões:
- *buracos de detecção*: quadros seguidos, dentro da vida de uma identidade, sem nenhuma detecção do SDP com
  IoU ≥ 0,5 (o que o rastreador realmente enfrenta): n = 4105, **mediana 2, p90 12**, 48% de 1 quadro;
  **5% passam do N50 da RNN (21)** e 3% passam de 30;
- *visibilidade < 0,2 no GT* (oclusão forte): n = 1021, mediana 9, **26% passam de 21**.
O detector enxerga pessoas parcialmente ocluídas, então os buracos de detecção são bem mais curtos que as
oclusões do GT. Para o rastreador, a memória de ~21 quadros cobre ~95% dos buracos de detecção; já para as
oclusões fortes de verdade, mais de 1/4 estão além do horizonte do estado.

**Analítico × empírico.** Os dois apontam a mesma ordem de grandeza (gradiente a 1/10 em k = 13; N50 = 21 quadros),
mas medem coisas diferentes (um é quanto sinal chega ao estado no treino, o outro é por quanto tempo a previsão
fica boa); não os apresento como a mesma quantidade.

## 3. Galeria de falhas (vídeos de validação; `parte4_falha_*.png`, `parte4_trechos.csv`, `parte4_falhas.json`)
Dos 461 buracos de rastreamento nos vídeos 09 e 13, 209 terminam sem manter o id. Três trechos escolhidos por
**regras fixas** (código em `pa2/analysis/gallery.py::pick_failures`), cada um mostrando quadros com GT (contínuo, colorido pelo id do GT; a
identidade em foco, grossa e rotulada), tracks coloridas por id da track (tracejado), a caixa que a **recorrência** previu (magenta,
pontilhada) e, embaixo, centro x, centro y e IoU previsão × GT ao longo do buraco.

**Falha 1 — oclusão longa, câmera parada** (09, GT 3, 18 quadros, visibilidade 0,14, T25 → T35; `…oclusao_longa.png`).
A caixa da recorrência deriva ~200 px para a **esquerda** enquanto a pessoa anda ~3,7 px/quadro para a
**direita**; o IoU cai abaixo de 0,3 já no 3º quadro do buraco e chega a 0 no fim, então a detecção que volta
não passa pelo portão e nasce uma track nova (a antiga continua viva, fantasma). *Diagnóstico:* não é falta de
horizonte (18 < N50 = 21); é o **sinal da velocidade estimada no início do buraco**. Antes do buraco a pessoa
estava quase parada (1,6 px/quadro) e sua visibilidade no GT é baixa (0,14 em média no buraco); caixas de
pessoas encobertas são ruidosas (o ruído medido do SDP é de ~6% da largura em cx), e a rede leu um
movimento para a esquerda e o **integrou** às cegas, como o gradiente cego crescente sugere (hipótese: não
vejo o quadro, então não confirmo que a caixa detectada estava deslocada).

**Falha 2 — buraco curto, câmera móvel** (13, GT 35, 4 quadros, visibilidade 0,83, T66 → T71; `…camera_movel.png`).
Não é oclusão (a pessoa está 83% visível): o detector falhou 4 quadros. A pessoa é **pequena (12×31 px) e anda
4,5 px/quadro, 0,39 larguras por quadro**; a caixa prevista quase não anda (−1 px/quadro contra −4,7) e como a
caixa é estreita o IoU vai a 0 em 1 quadro. *Diagnóstico:* movimento rápido **relativo à largura** é raro no
treino (0,8% dos passos passam de 0,3 larguras/quadro; p99 = 0,28) e caixas com menos de 20 px são 2,5%; a rede
encolhe a velocidade nessa cauda, e a câmera móvel é a fonte do movimento, que o modelo não vê.

**Falha 3 — troca entre duas pessoas** (09, GT 14, 6 quadros, T26 → T23, que era de outra identidade, GT 6;
`…troca_entre_pessoas.png`). Aqui a **recorrência ainda estava dentro do portão**: o IoU previsão × GT é 0,47 no 1º quadro depois do buraco
(> 0,3), embora já caindo (0,17 três quadros depois). Mas há
três pessoas lado a lado com caixas quase sobrepostas (105×257 px) e a associação por IoU, gulosa, deu a
detecção à track vizinha (T23), cuja caixa tinha IoU maior. *Diagnóstico:* **não é um problema de memória do
estado**; a geometria sozinha não distingue pessoas sobrepostas. É o caso para o qual a Trilha B (aparência) foi
feita.

## 4. Correção (`parte4_correcao.png`, `parte4_correcao.csv`, `parte4_extrapolacao.csv`)
**Diagnóstico escolhido:** o da falha 1 e da curva de gradiente cego: às cegas, a rede integra o erro de
velocidade do estado e se afasta. **Mudança que ele sugere:** amortecer o deslocamento extrapolado nos passos sem
observação (fator por passo; 1 = antes). Só no rastreador, sem retreinar.

**Hipótese registrada antes de rodar:** (i) o amortecimento aumenta o IoU às cegas e o N50 em buracos longos;
(ii) piora um pouco os curtos; (iii) muda pouco o IDF1 (90% dos buracos reais têm ≤ 12 quadros). Fator escolhido
pelo IDF1 médio de **treino** entre 1,0 / 0,95 / 0,9 / 0,8 / 0,7.

| fator | IoU cego k=5 val | k=10 val | k=30 val | N50 (7 vídeos) | IDF1 treino | IDF1 val |
|---|---|---|---|---|---|---|
| **1,0 (antes)** | 0,651 | **0,453** | 0,150 | **22,8** | 0,602 | 0,576 |
| 0,95 | 0,642 | 0,440 | 0,154 | 22,2 | 0,605 | 0,578 |
| 0,9 | 0,632 | 0,423 | 0,160 | 22,3 | 0,601 | 0,577 |
| 0,8 | 0,607 | 0,395 | 0,165 | 21,7 | 0,605 | 0,566 |
| 0,7 | 0,583 | 0,371 | 0,151 | 21,6 | 0,608 | 0,565 |

**Não funcionou.** (i) **não confirmada**: nenhum fator aumenta o N50 (1,0: 22,8; os demais, 21,6 a 22,3) e o IoU
às cegas na validação **cai** de k = 5 a k = 20 com todo fator < 1; só em k = 30 há um ganho pequeno (0,150 → 0,165
com 0,8) e em k = 20 do treino +0,01. Ressalva: as diferenças de N50 de ~1 quadro **estão dentro do ruído** (o IC 95%
do N50 é de ≈ ±2,5, §2; e o mesmo modelo sem correção dá 20,9 na §2 e 22,8 aqui, porque esta tabela usa outra grade
de N e outras tentativas). Por isso a evidência contra a correção é o IoU às cegas e a ausência de ganho no N50, não
uma "queda" do N50. (ii) confirmada (k = 5 val: 0,651 → 0,583). (iii) confirmada: o IDF1 varia ±0,003 no treino e
cai 0,01 na validação com 0,8 e 0,7. A regra registrada escolheria 0,7 (IDF1 de treino 0,608 contra 0,602, uma
diferença de ruído), que é pior na validação e não ganha em N50; **não adotei a correção**.

**O que isso revela sobre o diagnóstico.** A rede **já encolhe** o movimento às cegas (`parte4_extrapolacao.csv`):
regressão do deslocamento previsto sobre o verdadeiro dá inclinação 0,63 / 0,52 / 0,36 (treino, k = 5 / 10 / 20)
e 0,79 / 0,71 / 0,54 (validação), com correlação de 0,95 na validação. Isto é, na média ela **subestima**; o erro
da falha 1 (sentido errado) é uma cauda (4% a 15% dos casos em que a pessoa anda, na validação; 13% a 26% no treino), não o comportamento típico.
O diagnóstico partiu de um caso e da curva de gradiente, mas "o estado exagera a velocidade" não descreve o
modelo: amortecer encolhe ainda mais o que já está encolhido. **Acompanhamento exploratório, pós-hoc** (escolhido
depois de ver a inclinação < 1, portanto fora da hipótese registrada): amplificar (fatores 1,15 e 1,3) dá um
resultado **misto**, não nulo: com 1,15 o IoU às cegas na validação sobe um pouco em k = 5 e 10 (0,651 → 0,664 e
0,453 → 0,464) mas cai em k = 30 (0,150 → 0,121), o N50 vai a 20,6 (dentro do ruído) e o IDF1 não muda (0,607 / 0,577);
com 1,3 tudo piora (N50 17,7). Nenhum fator melhora o rastreamento, então o fator 1,0 é mantido. O que sobra da falha 1 é um problema de **estimar a velocidade no
início do buraco a partir de caixas ruidosas**, que escalar a extrapolação não resolve; as correções que o
atacariam (suavizar a velocidade de entrada, ou o portão por incerteza do opcional da Parte 2) ficaram como
próximo passo, **não testadas**.

## 5. Limitações
- Sem imagens, as figuras são caixas sobre fundo vazio; o diagnóstico de "pessoa parcialmente encoberta" na falha 1
  vem da visibilidade do GT, não de ver o quadro.
- As falhas foram escolhidas por regras fixas entre os vídeos de validação; cada uma é **um** caso, ilustrando uma
  causa possível, e as causas (sinal da velocidade, cauda de velocidade relativa, sobreposição) não foram
  quantificadas em todos os 209 buracos. As inclinações < 1 da §4 indicam que o caso da falha 1 é a cauda, não o típico.
- O experimento de oclusões injetadas usa 7 vídeos com modelo treinado em 5 deles (treino é in-sample para a RNN),
  e as oclusões removem a detecção de uma pessoa sem mexer na cena em torno; interações entre tentativas da mesma
  execução (não sobrepostas no tempo, mas no mesmo vídeo) existem.
- 1 modelo, 1 seed no modelo final; os intervalos são de amostragem das oclusões, não de treino.
