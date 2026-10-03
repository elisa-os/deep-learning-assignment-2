# Relatório — Parte 3 (Ablação, Eixo 2: o regime de treino)

**Reproduzir:** `uv run pa2 3` (≈25 min em CPU; retomável) e `uv run pytest`. Saídas em
`outputs/parte3_ablation/` (`parte3_summary.csv`, `parte3_runs.csv`, `parte3_reconnection.csv`, 4 figuras,
`runs/*.json`, `checkpoints/`). Resultados por seed em `parte3_runs.csv`.

## 1. Desenho
Mesmo modelo da Parte 2 (GRU 64, 19 mil parâmetros) e mesma detecção congelada (SDP). Só o regime de treino muda.
3 seeds (42, 123, 456), média ± desvio **amostral** (ddof = 1). As seeds são pareadas entre regimes (mesma
inicialização e mesma sequência de janelas; o ruído de observação não é idêntico entre regimes: o sorteio do scheduled sampling/free-running consome o mesmo gerador), então comparo também os sinais por seed.

| regime | alimentação no treino | clipping |
|---|---|---|
| `teacher_forcing` | sempre a observação | on / off |
| `scheduled_sampling` | prob. de observação decai linearmente de 1 a 0 | on / off |
| `free_running` | só as próprias previsões (depois do 1º quadro) | on / off |
| `part2_recipe` | observação + buracos simulados (p = 0,5, até 20 quadros) | on |

7 configurações × 3 seeds = 21 treinos de 20 épocas × 100 passos. Os três regimes puros usam `gap_prob = 0` para
isolar o efeito (com buracos o teacher forcing deixa de ser puro). **Protocolo fixo:** mesmas épocas, lr, batch e
ruído; checkpoint da **última** época (sem escolher a época pela validação); validação durante o treino com o
mesmo protocolo para todos. A regressão foi conferida: o checkpoint da Parte 2 continua dando exatamente a mesma
perda e IoU de validação depois da extensão do código de treino.

O que se mede (por run): (a) rastreamento nos 7 vídeos com a regra da Parte 2 (**validação = vídeos 09 e 13 é o
resultado principal**; os de treino são in-sample para a RNN); (b) IoU após k quadros às cegas, depois de 8
quadros observados com ruído de detector (`parte3_blind_rollout.png`); (c) "deriva": o mesmo modelo, nas mesmas
janelas, alimentado com observações × só com as próprias previsões; (d) reconexão depois de buracos de
rastreamento; (e) estabilidade do treino.

## 2. Resultados (média ± desvio sobre 3 seeds)

| regime | IDF1 val | ID sw/id val | ids prev./verd. val | IDF1 7 vídeos | IoU obs. | IoU às cegas k=10 (val) | k=30 (val) |
|---|---|---|---|---|---|---|---|
| teacher_forcing | **0,579 ± 0,003** | 1,64 ± 0,10 | 1,52 ± 0,04 | 0,597 ± 0,004 | 0,825 ± 0,001 | **0,464 ± 0,017** | **0,166 ± 0,015** |
| scheduled_sampling | 0,576 ± 0,007 | 1,66 ± 0,11 | 1,54 ± 0,01 | 0,593 ± 0,005 | 0,815 ± 0,001 | 0,418 ± 0,010 | 0,143 ± 0,003 |
| free_running | 0,472 ± 0,014 | 2,95 ± 0,15 | 1,83 ± 0,02 | 0,538 ± 0,006 | 0,724 ± 0,005 | 0,257 ± 0,007 | 0,091 ± 0,011 |
| free_running, sem clip | 0,492 ± 0,014 | 3,04 ± 0,11 | 1,83 ± 0,02 | 0,543 ± 0,006 | 0,724 ± 0,003 | 0,257 ± 0,008 | 0,091 ± 0,013 |
| teacher_forcing, sem clip | 0,579 ± 0,003 | 1,64 ± 0,10 | 1,52 ± 0,04 | 0,597 ± 0,004 | 0,825 ± 0,001 | 0,464 ± 0,017 | 0,166 ± 0,015 |
| scheduled_sampling, sem clip | 0,579 ± 0,008 | 1,60 ± 0,20 | 1,54 ± 0,02 | 0,594 ± 0,004 | 0,816 ± 0,000 | 0,425 ± 0,007 | 0,144 ± 0,006 |
| part2_recipe | 0,573 ± 0,010 | 1,81 ± 0,19 | 1,52 ± 0,02 | 0,592 ± 0,003 | 0,816 ± 0,001 | 0,410 ± 0,019 | 0,135 ± 0,006 |

(Referência: a Parte 2 entregue, com a mesma receita mas escolhendo a época pela validação, deu IDF1 val 0,562.)

**Estabilidade do treino** (`parte3_grad_norms.png`):

| regime | norma do gradiente (média / máx. por época) | passos com norma > 1 | passos com perda ou gradiente não finitos | perda de validação final |
|---|---|---|---|---|
| teacher_forcing | 0,09 / 0,3 | 0 % | 0 | 0,49 ± 0,02 |
| scheduled_sampling | 0,31 / 2,3 | 5 % | 0 | 0,52 ± 0,00 |
| free_running | 0,94 / 2,6 | 38 % | 0 | 0,91 ± 0,02 |
| free_running, sem clip | 0,93 / 2,3 | 36 % (que seriam clipados) | 0 | 0,91 ± 0,02 |
| part2_recipe | 0,28 / 1,9 | 0,4 % | 0 | 0,54 ± 0,02 |

## 3. Leitura dos dados (só o que os números sustentam)

**Free-running é claramente pior.** IDF1 val −0,10 em relação ao teacher forcing, com o mesmo sinal nas 3 seeds
(−0,112, −0,094, −0,116); ~80 % mais ID switches por identidade; 1,83 contra 1,52 ids previstos por id verdadeiro;
a fração de buracos que terminam em id novo vai de 0,30 para 0,47. Ele também é pior no IoU **com** observação
(0,72 contra 0,83). Ressalva importante de interpretação: esse regime nunca vê a flag "observada = 1" depois do
primeiro passo, mas na inferência ela é quase sempre 1. Parte do estrago é esse descasamento e a perda de acesso
ao histórico, e não só "viés de exposição".

**Teacher forcing, scheduled sampling e a receita da Parte 2 não se distinguem no IDF1** (0,579 / 0,576 / 0,573,
diferenças menores que um desvio). Mas no **rollout às cegas** o teacher forcing puro é o melhor na validação:
IoU em k = 10 de 0,464 contra 0,418 (scheduled sampling) e 0,410 (receita da Parte 2), com a receita abaixo do
teacher forcing em 3 de 3 seeds (0,400/0,397/0,431 contra 0,453/0,456/0,483). Isso contraria o motivo da
receita da Parte 2: treinar com buracos simulados (o modelo se alimentando das próprias previsões) **não
melhorou nada** aqui; o IDF1 foi igual ou um pouco menor (−0,014, −0,002, −0,003 por seed) e os ID switches por
id subiram (1,81 contra 1,64, dentro do ruído).

**Nas trajetórias de treino o quadro é outro:** para k ≥ 15, scheduled sampling e receita ficam acima do
teacher forcing (IoU em k = 30: 0,51 e 0,50 contra 0,44), o que é o efeito que o enunciado antecipa. Essa
vantagem não aparece na validação (lá é o contrário). Ou seja, expor o modelo às próprias previsões ajuda
**dentro** dos vídeos de treino e não generaliza para os vídeos 09 e 13; com 2 vídeos de validação não dá para
dizer se é um efeito real de generalização ou particularidade desses vídeos.

**Gradient clipping: sem ele, nada de ruim acontece neste setup.**
- Zero divergências, zero passos com perda ou gradiente não finitos, nos 21 treinos.
- No teacher forcing o clipping nunca atua (norma máxima 0,3 contra limite 1,0): os runs com e sem clipping são
  **idênticos bit a bit**, o que também confirma que a configuração "sem clipping" está de fato desligada.
- No scheduled sampling atua em ~5 % dos passos, sem efeito mensurável (0,576 contra 0,579).
- No free-running a norma média é ~0,94 e o clipping atuaria em ~37 % dos passos (37,6 % com clipping ligado; 35,7 % sem); ainda assim os runs sem
  clipping não são piores: IDF1 val 0,492 contra 0,472 (+0,02, mesmo sinal nas 3 seeds: +0,012, +0,008, +0,040,
  mas com n = 3 e desvio de 0,014 isso **não é distinguível de ruído**), IoU igual.
- A norma do gradiente cresce com o uso das próprias previsões: ~0,1 no teacher forcing, ~1,0 no free-running; no
  scheduled sampling ela sobe de 0,1 a ~1,1 à medida que a prob. de observação cai. É coerente com o gradiente
  atravessando passos alimentados por previsões, mas não testei o mecanismo.
- Por que não explode, é hipótese não testada: perda smooth-L1 (gradiente limitado por passo), GRU com portas,
  `exp(o/10)` suave na saída, Adam com lr 2e-3 e janela de 32 passos.

## 4. Hipóteses escritas antes de rodar × resultado
- **H1** (teacher forcing é pior às cegas e tem mais ID switches depois de buracos): **refutada na validação**
  (é o melhor ou empata com os outros em todos os k, e é o de menos buracos que terminam em id novo, 0,30), **parcialmente confirmada só nas
  trajetórias de treino** em k ≥ 15.
- **H2** (free-running melhora às cegas mas piora com observação e no rastreamento): **metade confirmada.** Piora
  com observação e no rastreamento, mas **não melhora** às cegas na validação (é o pior); nas trajetórias de treino fica abaixo dos outros em k = 10
  (0,66 contra 0,69–0,71) e perto do scheduled sampling em k = 30 (0,49 contra 0,51).
- **H3** (sem clipping o free-running explode; no teacher forcing é indiferente): **refutada** para a explosão
  (nada diverge); confirmada a indiferença no teacher forcing.

## 5. Medida que não funcionou
A "deriva" medida como IoU só com as próprias previsões a partir do 1º quadro (`parte3_shift.png`, painel da
direita) deu ~0,22 **em todos os regimes**, inclusive no free-running, que é treinado exatamente assim. Com um
único quadro observado não há velocidade para extrapolar, então o limite é de informação e não de regime de
treino. A medida informativa é o rollout às cegas **depois de 8 quadros observados**
(`parte3_blind_rollout.png`). Mantive a primeira no relatório como resultado negativo.

## 6. Limitações
- 3 seeds e 2 vídeos de validação: pouco poder. Diferenças menores que ~1 desvio são "não distinguíveis".
- Uma célula (GRU 64), um lr, 20 épocas, uma regra de associação.
- O free-running definido assim nunca vê observação depois do 1º passo, então não é um bom proxy do regime
  de inferência (em que as detecções existem quase sempre). Uma variante que treinasse alimentando a previsão
  *com a flag "observada = 1"* não foi testada.
- Reprodutibilidade: dentro desta rodada, mesma seed e mesmas threads reproduzem bit a bit (teacher forcing
  com e sem clipping dão runs idênticos). O retreino da Parte 2 na minha máquina não reproduziu o checkpoint;
  atribuo a diferença de threads/máquina, mas não testei.
- Mecanismos (por que o clipping não importa; por que o free-running piora) estão marcados como hipóteses.

## 7. Consequências para as próximas partes
- **Modelo final.** Pela regra definida antes de rodar (maior IDF1 médio de validação, checkpoint da seed 42):
  `teacher_forcing` (0,5795; empata, dentro do ruído, com scheduled sampling sem clipping 0,5790 e com a
  receita da Parte 2 em 0,5733). Checkpoint: `outputs/parte3_ablation/checkpoints/teacher_forcing_s42.pt`.
  **Decisão pendente da dupla:** usar esse modelo (e não o da Parte 2) nas Partes 4 e 5 e no notebook.
- **Parte 4.** Há checkpoints de regimes que diferem no rollout às cegas (teacher forcing, receita, free-running)
  para a análise empírica de sobrevivência à oclusão; a curva de gradiente ∂L/∂h (analítica) fica para o GRU.
- A resposta da Parte 2 sobre "o estado não atravessa buracos longos" continua valendo: o IoU às cegas cai a
  ~0,15 em k = 30 em todos os regimes não degenerados, e treinar com buracos simulados não mudou isso.

## 8. EXTRA (03/10): Eixo 1 — célula recorrente × janela de BPTT
Rodado **depois** da entrega do Eixo 2, como extra (o enunciado pede um eixo; este não substitui o Eixo 2). **Não registrei hipóteses numéricas antes de rodar**; a
expectativa teórica (aula) era "a RNN simples quebra primeiro e as células com portas seguram mais passos".
**Desenho** (`pa2/ablation_cells.py`, `uv run pa2 eixo1`, saídas em `outputs/extra_eixo1/`): RNN × GRU × LSTM com **orçamento de parâmetros igual**
(RNN h = 93, 18.883 parâmetros; GRU h = 64, 19.012; LSTM h = 57, 19.270), T ∈ {4, 8, 16, 32}, 3 seeds = 36 treinos; resto igual ao modelo final
(teacher forcing, clipping ligado, 20 épocas × 100 passos, última época; validação e rastreamento sempre com janelas T = 32).

| célula | T = 4 | T = 8 | T = 16 | T = 32 |
|---|---|---|---|---|
| RNN simples (IDF1 val) | 0,533 ± 0,005 | 0,537 ± 0,003 | 0,566 ± 0,005 | 0,566 ± 0,017 |
| GRU | 0,521 ± 0,011 | 0,546 ± 0,006 | 0,552 ± 0,010 | **0,579 ± 0,002** |
| LSTM | 0,538 ± 0,011 | 0,540 ± 0,006 | 0,568 ± 0,004 | 0,566 ± 0,007 |

IoU às cegas em k = 10 (validação): RNN 0,335 / 0,419 / 0,478 / 0,454; GRU 0,350 / 0,452 / 0,478 / 0,464; LSTM 0,370 / 0,426 / 0,431 / 0,414 (T = 4/8/16/32).
- **Janela:** em todas as células o IDF1 sobe de T = 4 para T = 16/32 (+0,03 a +0,06): janelas muito curtas não ensinam o filtro de ruído.
- **Célula:** a diferença entre células é **pequena** (≤ 0,03 de IDF1, da ordem de 1 a 2 desvios). A GRU com T = 32 é a melhor (0,579), mas a RNN simples
  empata ou ganha da GRU em T = 16 (0,566 × 0,552). **Neste setup a RNN simples não "quebra"**: a pergunta "onde a RNN simples quebra?" tem como resposta
  "nas janelas curtas, junto com as outras; não de forma específica da célula". Isso contraria a expectativa teórica, e não testei por quê.
- **Curva de gradiente** (`eixo1_gradiente.png`, T = 32, média de 3 seeds, mesma medida da Parte 4): a norma cai a 1/10 em k ≈ **15** na GRU, **8,7** na RNN simples e
  **7,3** na LSTM. Ou seja, a GRU preserva o gradiente por mais passos (como a teoria diz), mas isso **não vira diferença clara de IDF1** nem de IoU às cegas.
  (A LSTM tem o estado [h, c] de 114 números; a norma é do estado inteiro, então não é diretamente comparável com a das outras.) Às cegas o gradiente **cresce** nas três
  células (razão em k = 31: GRU 7,0; RNN 4,7; LSTM 3,1), como na Parte 4.
- **Estabilidade:** 0 divergências nos 36 treinos.
- **Ressalvas:** 3 seeds e 2 vídeos de validação; diferenças menores que ~1 desvio não são distinguíveis; um lr; a célula de comparação com h diferente muda também a
  capacidade da cabeça (64 → h na camada oculta da cabeça).

