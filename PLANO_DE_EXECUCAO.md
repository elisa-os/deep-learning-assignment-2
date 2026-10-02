# Plano de Execução — PA2: Identidade ao longo do tempo

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Alunos:** Bruno Ferreira & Elisa Soares  
**Entrega:** 02/10, 23h59  
**Status:** Em construção

---

## Visão geral

PA2 tem a mesma arquitetura de projeto que o PA1: `uv` para ambiente, `pyproject.toml` com CLI scripts, `config.yaml` por parte, `config.py` com dataclasses, `utils/` com seed/device/export/visualize, e separação por subpacotes. O que muda é o domínio (vídeo sequencial em vez de imagem estática) e o que se mede (IDF1/ID switches em vez de mAP).

---

## Fase 0 — Estrutura do projeto  *(plano inicial; a estrutura real e atual está no `README.md`)*

Criar `deep-learning-assignment-2/` com a mesma estrutura de pastas do PA1:

```
deep-learning-assignment-2/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── README.md
├── AI_LOG.md            # criado na Fase 7
├── PLANO_DE_EXECUCAO.md   # este arquivo
└── pa2/
    ├── __init__.py
    ├── main.py          # pipeline principal (Fase 1+)
    ├── config.py        # dataclasses (copiar do PA1 + adaptar)
    ├── config.yaml      # configuração por parte (Fase 1)
    ├── metrics/
    │   ├── __init__.py
    │   └── tracking.py  # IDF1, ID switches, fragmentações (Fase 1, Parte 0)
    └── utils/
        ├── __init__.py
        ├── seed.py      # copiar do PA1
        ├── device.py    # copiar do PA1
        ├── export.py    # adaptar do PA1 (colunas de tracking)
        └── visualize.py # adaptar do PA1 (plot de trajetórias)
```

Arquivos criados nas fases seguintes:
```
pa2/
├── synthetic_video/     # Fase 1 — gerador + simulador de detector (Parte 0)
│   ├── __init__.py
│   └── synthetic.py
├── mot17/               # Fase 2 — loader MOT17 (Parte 1)
│   ├── __init__.py
│   └── loader.py
├── association/         # Fase 2 — matching e gestão de tracks (Parte 1)
│   ├── __init__.py
│   └── matching.py
├── models/              # Fase 3 — RNN de movimento ou aparência (Parte 2)
│   ├── __init__.py
│   └── motion_rnn.py   # (ou appearance_rnn.py se for Trilha B)
├── ablation.py          # Fase 4 — runner de ablações (Parte 3)
├── stress/              # Fase 6 — testes de estresse (Parte 5)
│   ├── __init__.py
│   └── stress_test.py
└── inferencia.ipynb     # Fase 7 — notebook de inferência
```

---

## O que copiar/adaptar diretamente do PA1

| Arquivo | Status | Adaptação |
|---|---|---|
| `pyproject.toml` | Copiar estrutura | Mudar nome/descrição, adicionar scripts do PA2 |
| `config.py` (dataclasses) | Copiar | Adicionar campos para sequência, janela T, tipo de rastreador |
| `config.yaml` | Copiar estrutura | Seções por parte do PA2 |
| `utils/seed.py` | Copiar direto | Nenhuma |
| `utils/device.py` | Copiar direto | Nenhuma |
| `utils/export.py` | Copiar e adaptar | Colunas diferentes (IDF1, ID sw, fragmentações) |
| `utils/visualize.py` | Copiar e adaptar | Plot de trajetórias, galeria de falhas em vídeo |

**Não reutilizar:** `metrics/instance.py` (métricas diferentes), `data/` (domínio MOT17 diferente), `models/unet.py`, `losses/`, `postprocessing/`.

---

## Fase 1 — Parte 0: Testes sintéticos

**Gerador de vídeos sintéticos (`pa2/synthetic_video/synthetic.py`):**
- Vídeos 128×128, 30-60 quadros
- 5-15 elipses em movimento, tamanhos variados
- Ruído e contraste variáveis
- Parâmetros expostos: n_objetos, velocidade_tipica, duracao_oclusao
- **Requisito verificável:** elipses desenhadas com ordem de profundidade, de modo que uma passe atrás da outra e realmente desapareça (não é só ocultação superficial)
- Figura com uma trajetória que some por N quadros e volta

**Simulador de detector (`pa2/synthetic_video/synthetic.py`):**
- Recebe as caixas verdadeiras e as estraga de propósito
- Descarta p% das detecções
- Adiciona ruído nas coordenadas
- Injeta falsos positivos
- Valida a Parte 1 inteira no sintético antes de baixar o MOT17

**Métrica e testes (`pa2/metrics/tracking.py`):**
- Implementar IDF1 e contagem de ID switches
- Três casos construídos à mão:
  - (a) predição = ground truth → IDF1 = 1 e zero switches
  - (b) duas identidades trocadas a partir do quadro k → número exato de switches esperado
  - (c) uma track partida em duas no meio → efeito esperado em IDF1 (não o mesmo de (b))

**Baseline no piso fácil:**
- Rodar associação ingênua da Parte 1 com poucas elipses, lentas e sem oclusão
- IDF1 tem que ficar muito perto de 1
- Depois girar os botões do gerador (mais objetos, mais rápidos, oclusão mais longa) e mostrar onde o baseline começa a quebrar
- Esse gráfico é o ensaio da Parte 1

---

## Fase 2 — Download MOT17 + Parte 1 baseline

1. **Download:** pacote só de anotações (~10 MB) de https://motchallenge.net/data/MOT17/
2. **`pa2/mot17/loader.py`:** ler dog.txt (frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility), carregar detecções públicas (DPM/Faster R-CNN/SDP), split por sequência (pelo menos 1 sequência inteira de validação que o modelo nunca vê)
3. **Escolher detector público padrão:** DPM, FRCNN ou SDP — um como fonte padrão do resto do PA, justificar no README
4. **Detector pré-treinado torchvision:** classe person do COCO, modo inferência
5. **Associação ingênua (`pa2/association/matching.py`):** IoU entre detecções de t e t-1, matching guloso ou Hungarian, limiar fixo, ID novo quando nada casa, track morta após k quadros sem observação
6. **Métricas (`pa2/metrics/tracking.py`):** IDF1 (atribuição global 1-para-1 entre identidades previstas e verdadeiras na sequência inteira), ID switches e fragmentações contadas explicitamente, erro de contagem de identidades únicas no vídeo
7. **Gráfico obrigatório (2 painéis):**
   - Em cima: mAP por quadro e IDF1
   - Embaixo: número de identidades previstas dividido pelo de verdadeiras e ID switches por identidade verdadeira
   - Sequências ordenadas por eixo de dificuldade (densidade, movimento de câmera, duração de oclusão)
8. **Documentar:** regra de associação e gestão de nascimento/morte de tracks no README

---

## Fase 3 — Parte 2: Memória temporal (escolher UMA trilha)

**Decisão tomada: Trilha A** (só há as anotações do MOT17 em `data/`; a Trilha B precisa dos recortes das imagens).

**Status da Parte 2: feita, exceto o opcional abaixo.** Ver `RELATORY_PART2.md`.

**Trilha A — RNN como modelo de movimento (recomendada):**
- LSTM/GRU por track, recebe última observação (caixa, confiança, Δt) e prevê caixa do próximo quadro
- Associação: IoU entre caixa prevista e caixa observada
- Sob oclusão: o estado roda para frente sem observação
- Perda L1/smooth-L1 sobre caixa, treinada em trajetórias do GT
- Opcional: prever incerteza e usar log-verossimilhança gaussiana (portão de associação adaptativo)

**Trilha B — RNN como memória de aparência:**
- Cada detecção vira embedding D-dimensional (recorte por encoder pequeno pré-treinado)
- Agregador recorrente mantém estado de aparência da track, atualizado a cada observação
- Associação: similaridade de cosseno, com portão geométrico de IoU por cima
- Perda contrastiva ou triplet sobre identidades do GT

**Plano se for Trilha A:**
1. `pa2/models/motion_rnn.py`: LSTM/GRU que recebe caixa + confiança + Δt e prevê próxima caixa
2. `pa2/association/matching.py`: matching entre caixa prevista (pela RNN) e caixa observada, IoU, limiar
3. Gestão de tracks com estado recorrente por track (estado roda forward sob oclusão)
4. Treinar em trajetórias do ground truth das sequências de treino
5. Comparar lado a lado com baseline da Parte 1, mesmas métricas, mesmas sequências
6. Responder na apresentação: o que quebra na fronteira entre janelas de T quadros e o que a representação permitiria fazer para costurar as identidades (rascunho na seção 5 do `RELATORY_PART2.md`)

**Feito:** itens 1–5 (modelo, rastreador com movimento, estado por track sob oclusão, treino no GT, comparação lado a lado com a Parte 1 e com velocidade constante), análise de reconexão depois de buracos e rascunho do item 6.

**Falta (opcional do enunciado): incerteza e portão de associação adaptativo.**
- [ ] Cabeça da rede prevendo também a log-variância da caixa (`predict_uncertainty: true`; hoje levanta `NotImplementedError` em `MotionRNN`)
- [ ] Perda: log-verossimilhança gaussiana no lugar do smooth-L1
- [ ] Portão na associação: aceitar o par só se a distância de Mahalanobis (caixa prevista × detecção) estiver abaixo de um limiar, no lugar do IoU fixo; o portão se alarga sozinho sob oclusão
- [ ] Comparar com a versão sem incerteza nas mesmas sequências; entra nas métricas da Parte 2 e pode alimentar a Parte 4 (o portão largo ajuda ou atrapalha nos buracos longos?)

---

## Fase 4 — Parte 3: Ablação — Eixo 2 (regime de treino), 3 seeds  *(FEITA — ver `RELATORY_PART3.md`)*

**Eixo escolhido: 2** (já declarado no README). Eixo 1 custaria 36 treinos (3 células × 4 janelas T × 3 seeds)
e exigiria igualar parâmetros entre RNN/LSTM/GRU; só compensaria pela curva de gradiente comparativa da Parte 4,
que é bônus ("se fizeram o Eixo 1"). Para a Parte 4 basta a curva do GRU.

### Hipóteses (escritas antes de rodar)
- H1 (exposure bias): treinar só com observações (TF) dá o melhor erro "com observação" mas piora nas previsões
  às cegas (k grande), e mais ID switches depois de buracos.
- H2: free-running melhora o rollout às cegas, mas piora o passo com observação (nunca viu a flag `observada = 1`
  depois do 1º passo) e o tracking; scheduled sampling fica no meio.
- H3: sem clipping, o free-running é o mais propenso a explodir (32 passos alimentando a própria previsão
  através de `w·exp(o/S)`); no TF o clipping quase não importa (smooth-L1 tem gradiente limitado).
Se os dados contrariarem, o relatório diz isso.

### Desenho
Fator A (política de entrada) × fator B (clipping), todos com **`gap_prob = 0`** para isolar o efeito (com buracos
simulados o TF deixa de ser puro: a receita da Parte 2 já expõe o modelo às próprias previsões):

| config | tf | clipping | obs. |
|---|---|---|---|
| TF | 1.0 | on / off | teacher forcing puro |
| SS | 1.0 → 0.0 linear ao longo das épocas | on / off | scheduled sampling |
| FR | 0.0 | on / off | free-running |
| **ref. Parte 2** | 1.0 + buracos (p = 0,5, até 20) | on | receita atual; não é TF puro |

= 7 configs × 3 seeds (42, 123, 456) = **21 treinos**. Fixos: GRU 64, T = 32, lr 2e-3, batch 64, ruído de observação
da Parte 2, **mesmo nº de épocas e checkpoint da ÚLTIMA época** (sem seleção por validação).

### O que medir
1. **Deriva (distribution shift):** IoU em janelas de validação com observação (inferência normal) vs só com as
   próprias previsões; e IoU após k quadros às cegas, k ∈ {1,2,5,10,15,20,30} (reuso de `gap_rollout_iou`).
2. **Tracking** (regra da Parte 2: guloso, IoU 0,3, `max_age` 30, `min_hits` 3; `min_conf` da Parte 1): IDF1, ID
   switches, ids previstos/verdadeiros nos vídeos de validação (09, 13); os 7 vídeos como secundário (treino é
   in-sample). Opcional: tabela de reconexão por duração de buraco.
3. **Estabilidade do treino:** norma do gradiente (média, máx., fração de passos com norma > clip), nº de passos
   não finitos, "divergiu" (pesos NaN) por seed.
4. Agregação: média ± desvio amostral (ddof = 1) sobre 3 seeds + CSV por seed. Diferença < 1 desvio = "não
   distinguível" (n = 3, 2 vídeos de validação: pouco poder). Saídas: barras com erro, curvas IoU×k com banda,
   normas de gradiente por época (clip on/off).

### Armadilhas encontradas no código atual (corrigir antes de rodar)
- `eval_windows` usa `cfg.tf_ratio` → a validação de FR/SS mediria outra coisa. Validação deve ter protocolo único.
- O melhor epoch é escolhido pela perda de validação, que teria significados diferentes por regime → usar a última.
- Passos com perda não finita são *pulados* em silêncio (`continue`) e o contador `nbad` só vai para o log; sem
  clipping, gradiente não finito com perda finita atualiza os pesos com NaN. Registrar, não esconder.
- Um modelo com NaN faz `linear_sum_assignment` levantar erro no rastreador → proteger e reportar "divergiu".
- Com clipping ligado, norma infinita também dá NaN (coeficiente NaN): medir os dois lados.
- O treino não é determinístico bit a bit (observado na Parte 2): seeds controlam a inicialização/amostragem, não
  garantem rerun idêntico. Reportar por seed.
- FR nunca vê a flag `observada = 1` após o 1º passo, mas na inferência ela é 1 quase sempre: parte do efeito
  medido é esse descasamento, não só "exposure bias". Interpretar com cuidado (a medida 1 separa os dois casos).

### Implementação (ordem)
1. `models/motion_rnn.py`: `TrainSettings` ganha `tf_schedule` (`constant|linear`), `tf_end`, `select`
   (`best|last`), protocolo de validação fixo (tf = 1); histórico guarda norma máx., fração clipada, passos não
   finitos, flag de divergência. **Padrões = comportamento da Parte 2** (teste de regressão: `--eval-only` do
   checkpoint atual continua dando os mesmos CSVs).
2. `pa2/ablation.py` (`run_ablation`, já chamado por `main.py`): laço config × seed, checkpoint em
   `outputs/parte3_ablation/checkpoints/`, retomável (pula o que já existe), `--jobs` opcional para rodar seeds em
   paralelo (CPU de 12 núcleos); reaproveita `Source`, `track_and_evaluate`, `gap_rollout_iou` de `part1.py`/`part2.py`.
3. `config.yaml` `parte3`: GRU, batch 64, lr 2e-3, `regimes` reescritos conforme a tabela acima, seeds.
4. Testes: agenda de `tf`, `tf = 0` usa só a própria previsão e flag 0, regime sem buracos, guarda de NaN,
   agregador média ± desvio, parsing do config.
5. Smoke (1 seed, 2 épocas) → medir tempo → rodada completa em segundo plano → figuras → `RELATORY_PART3.md`,
   README e `AI_LOG.md`.

### Custo estimado (CPU)
~6 s/época (30 épocas = 171 s medidos na Parte 2) → 20 épocas ≈ 2 min/treino + ~1 min de avaliação = **~1 h
sequencial, ~25 min com 3 processos**. O modelo é pequeno: GPU não ajuda.

### Decisão para a Parte 4/5
O "modelo final" (checkpoint do notebook e base dos testes de estresse) deve ser definido aqui, por regra: o regime
de maior IDF1 médio de validação; checkpoint da seed 42 (sem escolher a melhor seed).

---

## Fase 5 — Parte 4: Galeria de falhas + horizonte de memória  *(FEITA — ver `RELATORY_PART4.md`)*

1. Selecionar 3 trechos onde o modelo final erra feio
2. Para cada um:
   - Figura: tira de quadros com GT e predição coloridos por identidade + mapa intermediário relevante (caixa prevista pela recorrência ou matriz de similaridade de embeddings)
   - Diagnóstico escrito: "esse objeto fica ocluído por 34 quadros; minha janela de BPTT é de 10 e a norma do gradiente cai 20× em 8 passos, então o modelo nunca recebeu sinal de supervisão que atravessasse esse buraco"
3. **Horizonte de memória analítico:** norma de ∂L_t/∂h_{t−k} em função de k. Curva de gradiente que some no modelo e nos dados. Se fizeram Eixo 1 da Parte 3, comparar curva da RNN simples com a do modelo com portas.
4. **Horizonte de memória empírico:** quantos quadros o estado sobrevive a uma oclusão antes de track morrer ou trocar de ID. Comparar com distribuição de duração de oclusão do dataset.
5. **Correção:** escolher um diagnóstico, implementar a mudança que ele sugere, mostrar antes/depois. Se não funcionar, explicar o que isso revela sobre o diagnóstico estar errado.

---

## Fase 6 — Parte 5: Teste de estresse — queda de taxa de quadros  *(FEITA — ver `RELATORY_PART5.md`)*

**Decidido (01/10):** queda de taxa de quadros; curva principal estritamente **sem retreinar**, em cima do modelo final;
retreino multi-Δt como **exploratório rotulado** (fora da regra). Implementação: `pa2/stress/` (subamostragem e
diagnósticos), `pa2/part5.py`, seção `parte5` do `config.yaml`; saída em `outputs/final_parte5/`.

**Hipóteses registradas ANTES de rodar:**
- H1: todos os métodos perdem IDF1 com a queda de taxa.
- H2: a vantagem da RNN sobre a caixa parada encolhe ou some em 1/5 (o modelo foi aprendido em Δt = 1).
- H3: alimentar Δt = k a um modelo que só viu Δt = 1 não recupera nada (o recurso nunca variou no treino).
- H4: o retreino multi-Δt recupera parte da perda.
- Mecanismo proposto (a testar nos diagnósticos): em Δt = 1 o deslocamento por quadro é pequeno frente ao ruído do
  detector, então a rede aprendeu a *encolher* a velocidade; em Δt maior esse encolhimento passa a subestimar.

**Desenho:** subamostra a 1/2 e 1/5 em todas as fases (offsets); métodos: caixa parada, velocidade constante, RNN final com
Δt = 1, RNN final com Δt = k alimentado; regra de associação inalterada (principal) e `max_age` casado no tempo
(`round(30/k)`, secundária); diagnósticos em trajetórias do GT (IoU entre amostras consecutivas, inclinação do deslocamento
previsto, sensibilidade ao recurso Δt).

---

## Fase 7 — Entregáveis finais (target: até 02/10)

1. **`inferencia.ipynb`** *(FEITO, na raiz; lógica em `pa2/inference.py`)*: recebe a pasta de uma sequência qualquer (formato MOTChallenge), devolve vídeo com identidades coloridas de forma consistente e contagem de objetos únicos. Roda sem retreinar. Limites: sem imagens o fundo é vazio; o caminho torchvision (sem `det/det.txt`) está coberto por teste com modelo falso, mas nunca rodou em imagens reais.
2. **`README.md`:**
   - Ambiente (`uv sync`)
   - Download dos dados (MOT17, link, tamanho, pacote de anotações vs. frames)
   - **Um comando que treina**
   - **Um comando que avalia**
   - Arquivos entregues listados
3. **`pa2/metrics/tracking.py`:** implementação própria de IDF1, ID switches e fragmentações
4. **`AI_LOG.md`:** registrar uso de IA (episódios, decisões, problemas resolvidos)
5. **Checkpoint:** pesos do modelo temporal (link se for grande)
6. Commit final e push

---

## Backlog — ajustes adiados (não bloqueiam as partes seguintes)

Origem: revisão da Parte 2 (01/10). Fazer se sobrar tempo, na ordem.

**Parte 3 (achados que merecem seguimento)**
- [ ] A receita da Parte 2 (treinar com buracos simulados) **não melhorou** nada frente ao teacher forcing puro
      (IDF1 val 0,573 × 0,579; IoU às cegas k = 10: 0,410 × 0,464, 3/3 seeds). Decidir se o modelo final das
      Partes 4/5 e do notebook é o da Parte 2 ou o `teacher_forcing_s42` da Parte 3 (a regra pré-definida aponta
      para o segundo) e ajustar o texto da Parte 2 se mudar.
- [ ] Opcional: variante de free-running que alimenta a previsão com a flag "observada = 1" (o free-running atual
      mistura "viés de exposição" com descasamento da flag).
- [ ] Opcional: teste de estresse do clipping (lr ×5 e/ou célula RNN simples, que deve ser mais instável) para
      mostrar *quando* ele passa a importar; hoje "não importa" vale só para este setup.
- [ ] A medida "IoU só com as próprias previsões a partir do 1º quadro" não discrimina regimes; pode sair do relatório.

**Revisão da Parte 4 e do repositório (01/10) — adiado**
- [ ] Reproduzir do zero a etapa B da Parte 4 (oclusões injetadas; > 1 h, vídeo 04 é o gargalo): hoje só foi
      recomputada a partir de `parte4_oclusoes_injetadas.csv`. Rodar à noite com `uv run pa2 4`.
- [x] **Feito (revisão do repositório, 01/10):** os helpers importados entre módulos passaram a ser públicos (`Source`,
      `best_f1_threshold`, `pick_variant`, `track_and_evaluate`, `clear_match`, `frame_range`, `parse_tracks_to_frames`);
      `Context`/`build_context` deixaram de estar duplicados (`part5.py` usa o de `ablation.py`); código morto herdado do
      PA1 removido (`BoxAssociation`, `TrackManager`, `track_matches_iou`, `simple_greedy_match`, `HungarianMatcher`,
      5 funções de `utils/visualize.py`, `make_synthetic_loader`, `SyntheticVideoDataset`); lint limpo.
      **Pendente:** unificar `GreedyMatcher` (Parte 0) e `IoUTracker`.
- [ ] `outputs/` mistura convenções (`parte0_`/`parte1_`/`parte2_` soltos, `final/`, `final_parte4/`,
      `parte3_ablation/`); padronizar em `outputs/parteN/` e atualizar README e relatórios juntos.
- [ ] `data/MOT17Labels.zip` (9,7 MB) está versionado e é redundante com as pastas extraídas.
- [ ] Cosmético: títulos das figuras de falha saem sem acento ("oclusao longa") e há muito espaço vazio entre as
      tiras de quadros e os gráficos.
- [x] Feito na revisão: README (status, árvore, saídas), `CLAUDE.md`, relatório da Parte 4 (N50 com IC por bootstrap,
      refutação da correção pelo critério certo, resultado misto do fator 1,15), correção da frase sobre a seed 42 em
      `RELATORY_PART2.md` §6, mensagem final de `part4.py`, remoção do `eval_iou` sem efeito.

**Parte 2**
- [ ] Relatório: dizer que a validação (vídeos 09 e 13) escolheu a melhor época, logo não é independente; e que o
      baseline de velocidade constante é uma versão simplificada (média móvel da velocidade, β = 0,5), não um Kalman.
- [ ] Kalman de velocidade constante de verdade como baseline (o enunciado o permite); corrigir a atualização de
      velocidade logo após um buraco (hoje usa "observação − última previsão": salta de 8 para 9,5 px/q num objeto a
      8 px/q); tunar a regra de associação do baseline fora da borda da grade (melhor IoU hoje = 0,1, o mínimo).
- [ ] Reprodutibilidade: o retreino da RNN não reproduz o checkpoint (IDF1 de treino 0,590 → 0,605; melhor época 6 → 8).
      Tentar determinismo (threads/seeds) ou documentar; no README, "reproduzir exatamente" = `--eval-only`.
- [ ] Teste de `kept + switched = fragmentações` também para a RNN (hoje só para a caixa parada).
- [ ] Opcional do enunciado: incerteza + portão de associação adaptativo (ver Fase 3).

**Configuração**
- [x] `input_size` e `num_workers` removidos (nenhum código os lia). `dropout` e `num_layers` ficam no YAML, mas o modelo só aceita 1 camada e sem dropout (levanta erro se forem outros valores).
- [ ] `parte3` do YAML ainda diz LSTM (a Parte 2 usa GRU) → resolvido pela Fase 4.

**Parte 1**
- [ ] Rodar o detector torchvision (precisa das imagens do MOT17 e de GPU; `torch.cuda.is_available()` já dá `True`).

**Repositório**
- [ ] `.gitignore`: `data/` e `outputs/` foram liberados no commit `f169e2a` (51 MB de dados, binários que mudam a
      cada execução → conflitos). Decidir o que versionar; checar a licença do MOTChallenge (CC BY-NC-SA) se for público.

---

## Decisões pendentes

| # | Decisão | Recomendação | Responsável |
|---|---|---|---|
| 1 | Trilha A ou B (Parte 2) | **Decidido: Trilha A** (RNN movimento) | dupla |
| 1b | Modelo final (Partes 4 e 5, notebook) | **Decidido: `teacher_forcing_s42`** (`outputs/checkpoints/final_motion_rnn.pt`; `RELATORY_PART2.md` §6) | dupla |
| 2 | Eixo de ablação (Parte 3) | Eixo 2 (regime de treino) | dupla |
| 3 | Teste de estresse (Parte 5) | **Decidido: queda de taxa de quadros** (+ multi-Δt exploratório rotulado) | dupla |
| 4 | Sequência MOT17 de validação nunca vista | Justificar por câmera parada/móvel, densidade, ponto de vista | dupla |
| 5 | Divisão de responsabilidades entre Bruno e Elisa | Definir contratos de interface antes de começar | dupla |

---

## Regras de engenharia (do enunciado)

**Permitido:** PyTorch; `torchvision.models.detection` (Faster R-CNN, Mask R-CNN) e encoders pré-treinados; albumentations; scipy; sklearn; código de matching e métricas escritos no PA1.

**Permitido com condição:** filtro de Kalman de velocidade constante, **apenas** como baseline de comparação. Não pode ser o modelo temporal da Parte 2.

**Proibido:**
- Rastreadores prontos: SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT, Norfair, motpy, trackers do supervision, `model.track()` do ultralytics, ou qualquer implementação pronta de associação
- Métricas de rastreamento prontas de biblioteca: motmetrics, TrackEval, py-motmetrics. Implementar IDF1 e contagem de ID switches.
- `torchvision.ops.nms` — implementar o NMS
- Clonar solução pronta de MOT17. Citar e reescrever se usarem ideia de repo ou paper

**O modelo temporal, as perdas, a associação e a gestão de tracks são de autoria de vocês.**

---

## Mapeamento de arquivos do PA1 que são base para o PA2

| PA1 | PA2 | Reuso |
|---|---|---|
| `pa1/config.py` | `pa2/config.py` | Copiar + adaptar dataclasses |
| `pa1/config.yaml` | `pa2/config.yaml` | Copiar estrutura, preencher com partes do PA2 |
| `pa1/utils/seed.py` | `pa2/utils/seed.py` | Copiar direto |
| `pa1/utils/device.py` | `pa2/utils/device.py` | Copiar direto |
| `pa1/utils/export.py` | `pa2/utils/export.py` | Adaptar colunas (tracking metrics) |
| `pa1/utils/visualize.py` | `pa2/utils/visualize.py` | Adaptar (trajetórias, vídeo) |
| `pa1/metrics/instance.py` | `pa2/metrics/tracking.py` | Não reutilizar (métricas diferentes) |
| `pa1/data/synthetic.py` | `pa2/synthetic_video/synthetic.py` | Adaptar conceito (vídeo com oclusão) |
| `pa1/main.py` | `pa2/main.py` | Adaptar (loop sequencial, não imagem estática) |
