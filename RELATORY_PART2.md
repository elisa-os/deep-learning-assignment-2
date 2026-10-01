# Relatório — Parte 2 (Trilha A: RNN como modelo de movimento)

**Reproduzir:** `uv run pa2 2` (≈6 min em CPU) e `uv run pytest`. Saídas em `outputs/parte2_*`.
**Dados:** só as anotações do MOT17. Por isso a Trilha A (a B precisa dos recortes das imagens).
**Fonte de detecções (congelada):** SDP, score ≥ 0,4 — a mesma da Parte 1.

## 1. O que a recorrência carrega
Um estado por track (GRU, 64 unidades, 19 mil parâmetros). A cada passo a célula recebe a
caixa "alimentada" — a detecção casada ou, sem observação, a própria previsão — e emite o
movimento até o quadro seguinte. A associação (IoU, limiar, `min_hits`, `max_age`, guloso) é
a da Parte 1; só muda *onde a track espera o objeto*: caixa prevista em vez da última vista.
Sob oclusão o estado roda para frente alimentado só com as próprias previsões, até a track
casar de novo ou passar de `max_age`.
- Parametrização invariante à escala: Δcx/w, Δcy/h, Δlog w, Δlog h (×10). Features: esse
  deslocamento da entrada anterior, log(h/H), log(w/h), posição normalizada, Δt e a flag
  "observada". A confiança não entra (os scores do SDP/FRCNN saturam em 1,0).

## 2. A perda e de onde vem o gradiente
Smooth-L1 sobre o erro da caixa prevista contra o GT, em unidades da caixa. O GT do MOT17 é
contínuo, então há alvo mesmo quando a rede está "cega". Treino em janelas de T = 32 quadros
(BPTT truncado) das trajetórias do GT dos 5 vídeos de treino, com:
- **ruído de detector** nas entradas, medido SDP × GT: desvio de [cx/w, cy/h, log w, log h] =
  [0,06; 0,025; 0,08; 0,045] — comparável ao deslocamento por quadro, então a rede precisa filtrar;
- **buracos de observação** de 1 a 20 quadros em 50% das janelas, para treinar o modo
  "alimentar a própria previsão" que a inferência usa sob oclusão.
A validação (vídeos 09 e 13) escolhe a melhor época. Adam, cosseno, clipping de gradiente ligado.

## 3. Resultados (mesma regra de associação para os três, `outputs/parte2_per_sequence.csv`)
| método | split | IDF1 | ID sw. | ids prev./verd. | sw/id |
|---|---|---|---|---|---|
| Parte 1 (caixa parada) | treino | 0,564 | 243 | 1,88 | 3,51 |
| velocidade constante (baseline) | treino | 0,552 | 205 | 2,57 | 2,96 |
| **RNN** | treino | **0,590** | 203 | 1,86 | 2,93 |
| Parte 1 | validação | 0,488 | 233 | 1,90 | 2,95 |
| velocidade constante | validação | 0,534 | 109 | 2,04 | 1,71 |
| **RNN** | validação | **0,562** | 120 | **1,51** | 1,82 |

Com cada método usando o seu melhor (IoU, `max_age`) escolhido só no treino: validação 0,494 /
0,544 / **0,564** (Parte 1 / vel. constante / RNN). A conclusão não muda.

Movimento isolado (`parte2_gap_rollout.png`): IoU da caixa prevista após k quadros às cegas
(validação) — k = 5: 0,40 parada, 0,51 vel. constante, **0,66 RNN**; k = 10: 0,24 / 0,29 / **0,48**;
k = 30: 0,08 / 0,05 / 0,13. A RNN ganha em todos os horizontes; depois de ~15 quadros todas
degradam para perto de zero.

## 4. Em que aspecto a RNN melhora o fracasso da Parte 1 (`parte2_reconexao.png`)
Um **buraco** é uma sequência de quadros em que uma identidade verdadeira fica sem track
(`pa2/metrics/reconnection.py`; `kept + switched` bate com a contagem de fragmentações da
métrica). Desfecho: mesmo id depois (`kept`), id novo (`switched`) ou nunca volta (`lost`).
62% dos buracos têm visibilidade média < 0,5 no GT (oclusão); o resto é detecção perdida.
Mesma regra de associação, 7 vídeos, buracos agrupados por duração:

| duração do buraco | n (RNN) | mesmo id: parada | vel. const. | **RNN** | id novo: parada | **RNN** |
|---|---|---|---|---|---|---|
| 1–5 quadros | 2158 | 0,713 | 0,655 | **0,766** | 0,220 | 0,165 |
| 6–15 | 383 | 0,456 | 0,271 | **0,475** | 0,391 | 0,366 |
| 16–30 | 173 | 0,273 | 0,083 | **0,324** | 0,552 | 0,509 |
| 31+ | 199 | 0,059 | 0,023 | 0,050 | 0,659 | 0,668 |

Total de buracos que terminam em id novo: 881 (parada), 976 (vel. const.), **716 (RNN)**, −19%.

O que os dados sustentam:
- O ganho está nos buracos **curtos (1–5 quadros)**, que são a maioria, e em parte nos de
  16–30. Em 6–15 a diferença é pequena (+2 pontos) e **acima de 30 quadros não há ganho**
  (0,05 contra 0,06): a memória da RNN não atravessa oclusões longas. Isso é coerente com
  buracos de treino de até 20 quadros e janela de BPTT de 32, mas esta análise **não prova**
  essa causa; é a hipótese que a Parte 4 vai testar.
- O ganho aparece nos vídeos de **câmera móvel** (fração que mantém o id: 0,59 RNN × 0,49
  parada), e some nos de **câmera parada** (empate, 0,82 × 0,82). Plausível: com a câmera
  andando, a caixa parada erra mais e um modelo de movimento ajuda; mas a RNN não vê a câmera,
  então o que ela aprende é um movimento médio na imagem. Não testei essa explicação.
- A velocidade constante é **pior que a caixa parada** em buracos longos (0,08 contra 0,27
  em 16–30): extrapolar em linha reta por muitos quadros sai do objeto. A RNN não tem esse
  problema (o IoU de rollout dela cai devagar), o que é o ponto a favor dela contra o
  baseline "difícil de bater".
- Parte da diferença é de **quantidade de buracos** e não de reconexão: a RNN tem menos
  buracos curtos que a caixa parada (2158 contra 2293), porque a caixa prevista casa mais
  vezes com a detecção ruidosa.

Ressalvas: sem seeds (as barras de erro vêm na Parte 3); a RNN não ganha em IDF1 no vídeo 02
(empate) e no 10 empata com a velocidade constante; a velocidade constante tem menos ID
switches que a RNN na validação (109 contra 120) e perde dela sobretudo em ids duplicados.
A rede não compensa o movimento da câmera. O ruído do treino é gaussiano e independente por
quadro, o do detector real é correlacionado no tempo. A melhor época foi a 6 de 30 e a perda
de validação oscila (só 2 vídeos).

## 5. Pergunta da apresentação (sem implementar): o que quebra na fronteira entre janelas?
Vale para o modelo desta Parte 2; **não foi implementado nem testado**, porque o rastreador
aqui roda online, quadro a quadro, sobre o vídeo inteiro e só o *treino* usa janelas.
Com inferência em janelas de T quadros, quebram:
1. **O estado recorrente.** Cada track começa a janela com h = 0: perde a velocidade e o
   contexto acumulados, e a previsão dos primeiros quadros é pior (na rede, o primeiro passo
   sem histórico vale "fica parado").
2. **Os ids.** Cada janela numera as suas tracks; o objeto que cruza a fronteira ganha id novo
   (um ID switch por objeto vivo na fronteira) e quem estava ocluído na fronteira some, sem
   estado para continuar.
3. **O nascimento.** Com `min_hits = 3`, as primeiras observações de cada track da nova janela
   são descartadas.

O que a representação permite para costurar: (a) **passar o estado adiante**: guardar, no fim
de cada janela, (h, última caixa, quadros sem observação) de cada track viva e iniciar a
janela seguinte com eles (inferência com estado, como em BPTT truncado "stateful"); a perda
de gradiente na fronteira só afeta o treino, não a inferência; (b) **janelas com sobreposição**
de pelo menos `max_age` quadros, com a RNN rodando o estado das tracks que terminam para
dentro da próxima janela e um casamento (Hungarian por IoU entre caixa prevista e caixa
observada) na região comum; (c) para quem estava ocluído na fronteira, continuar rodando o
estado sem observação, que é exatamente o que o rastreador já faz. O limite é o da seção 4:
só geometria, sem aparência, o estado não sobrevive a mais de ~30 quadros e em multidão a
costura fica ambígua; uma memória de aparência (Trilha B) permitiria costurar por similaridade
de embeddings mesmo depois de buracos longos.
