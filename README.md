# PA2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Professor:** Dario Oliveira | **Monitor:** Erick Brito  
**Alunos:** Bruno Ferreira & Elisa Soares  
**Entrega:** 02/10, 23h59  
**Apresentação:** 02/10 — horários a definir  

> Enunciado completo em [`PA2.md`](PA2.md) (transcrição do `PA2.pdf`).

---

## Entregáveis (o que o enunciado pede)

### Ambiente

O projeto usa [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências e comandos:

```bash
uv sync            # cria o .venv e instala tudo (PyTorch, torchvision, scipy, pandas, matplotlib, ...)
uv run pytest      # 159 testes (≈15 s)
```

Em **Linux e Windows** o PyTorch vem do índice CUDA 11.8 (`2.7.1+cu118`, download de ~3 GB com as bibliotecas CUDA; roda também em CPU, que é o que a maior parte do projeto usa). Em **macOS** o `uv` usa o PyTorch do PyPI, na mesma versão (`2.7.1`; não testado em macOS). O pacote `pa2` é configurado via `module-root` no `pyproject.toml`.

Comandos do CLI (uma parte por vez; cada uma grava em `outputs/` e imprime o que mediu):

```bash
uv run pa2 0     # Parte 0 — testes sintéticos (gerador + simulador + métricas + baseline)      ≈20 s
uv run pa2 1     # Parte 1 — baseline por quadro (detecções públicas, associação por IoU)       ≈90 s
uv run pa2 2     # Parte 2 — RNN como modelo de movimento (Trilha A; modelo inicial)            ≈6 min
uv run pa2 3     # Parte 3 — ablação do regime de treino (Eixo 2; 21 treinos)                   ≈25 min
uv run pa2 4     # Parte 4 — horizonte de memória, galeria de falhas, correção                  ≈40 min
uv run pa2 5     # Parte 5 — teste de estresse (queda de taxa de quadros)                       ≈6 min
```

Opções úteis: `--output-dir <pasta>` (não sobrescreve os resultados versionados), `--eval-only --checkpoint <ckpt>`, `--epochs N`; na Parte 3, `--seeds` e `--regimes`.

### Dados

**Já estão no repositório**, em `data/MOT17/`: o pacote de anotações do MOT17 (~10 MB: `gt/gt.txt`, `det/det.txt` e `seqinfo.ini`), então **nada precisa ser baixado** para reproduzir as Partes 0 a 5. Estrutura:

```
data/MOT17/
  train/MOT17-02-SDP/{gt/gt.txt, det/det.txt, seqinfo.ini}    # também -DPM e -FRCNN; vídeos 02, 04, 05, 09, 10, 11, 13
  test/MOT17-01-SDP/{det/det.txt, seqinfo.ini}                # o test/ não tem GT
```

As **imagens** (`img1/`) não estão aqui: precisariam do `MOT17.zip` completo (~5,5 GB, https://motchallenge.net/data/MOT17/) e só servem para o detector do torchvision (Parte 1) e para o vídeo do notebook com fundo real. O pacote de anotações é o `MOT17Labels.zip` do mesmo site. O split de treino e validação é por vídeo e está definido em `pa2/mot17/loader.py` (`DEFAULT_SPLIT`; `sequence_split` do `config.yaml`, se preenchido, tem prioridade).

### Um comando que treina

Treina o **modelo final** (GRU de 64 unidades, teacher forcing puro, seed 42, 20 épocas, checkpoint da última época), que é o `teacher_forcing` da ablação da Parte 3, **sem sobrescrever** o checkpoint entregue:

```bash
uv run pa2 3 --regimes teacher_forcing --seeds 42 --output-dir outputs/retreino      # ≈2 min em CPU
# grava outputs/retreino/parte3_ablation/checkpoints/teacher_forcing_s42.pt
```

O treino usa as trajetórias do ground truth dos vídeos de treino (`pa2/config.yaml` → `parte3`). O retreino reproduz o regime mas **não bit a bit** o checkpoint entregue (threads/máquina; diferença máxima de peso medida ≈ 0,006; a conclusão da Parte 5 se mantém no retreino). Para usar o retreinado no lugar do entregue: `cp outputs/retreino/parte3_ablation/checkpoints/teacher_forcing_s42.pt outputs/checkpoints/final_motion_rnn.pt`. `uv run pa2 2` treina o modelo inicial da Parte 2 (com buracos simulados e época escolhida pela validação), que não é o final.

### Um comando que avalia

Avalia o modelo final sem retreinar:

```bash
uv run pa2 2 --eval-only --checkpoint outputs/checkpoints/final_motion_rnn.pt --output-dir outputs/final
```

Carrega os pesos e roda, nos 7 vídeos com GT (treino e validação), a comparação Parte 1 × velocidade constante × RNN e a análise de reconexão depois de buracos. Gera em `outputs/final/` (reproduz byte a byte o que está versionado):
- `parte2_summary.json` — configuração e médias por split
- `parte2_per_sequence.csv` / `parte2_per_sequence_tuned.csv` — métricas por vídeo (IDF1, ID switches, fragmentações, contagem)
- `parte2_comparacao.png` — Parte 1 × velocidade constante × RNN, vídeos ordenados por densidade
- `parte2_reconnection.csv`, `parte2_reconexao.png` — o que acontece com a identidade depois de um buraco

A Parte 0 não tem checkpoint (é só teste sintético): `uv run pa2 0` roda e valida tudo.

### Arquivos entregues junto com o repositório

Além do README.md, o repositório entrega:

- **`AI_LOG.md`** — log de uso de IA neste assignment, conforme exigido pela política de uso de IA do enunciado (seção 5 do PA2.pdf). Descreve episódios em que IA foi usada e como os problemas foram resolvidos.
- **`inferencia.ipynb`** — notebook de inferência: recebe a **pasta de uma sequência** (formato MOTChallenge) e devolve o vídeo com as identidades coloridas de forma consistente (mesmo id, mesma cor), as tracks e a contagem de objetos únicos, **sem retreinar**. Usa o checkpoint do modelo final (`outputs/checkpoints/final_motion_rnn.pt`). A lógica está em `pa2/inference.py` (testada); o notebook é só a casca. Para executar: `uv run jupyter nbconvert --to notebook --execute inferencia.ipynb --inplace` (ou abra no VS Code / `uv run --with jupyterlab jupyter lab inferencia.ipynb`); para outra sequência, troque `SEQUENCE_DIR` na primeira célula. Saídas em `outputs/inferencia/` (o `.mp4` completo não é versionado).
- **`outputs/checkpoints/final_motion_rnn.pt`** — pesos do modelo temporal final (GRU 64, teacher forcing, seed 42; decisão em `RELATORY_PART2.md` §6). É o artefato que o `inferencia.ipynb` e o comando de avaliação usam. O modelo inicial da Parte 2 continua em `parte2_motion_rnn.pt`.

---

## Detalhamento por Parte

Cada parte abaixo lista onde o código está, como reproduzir, o que gera e o status.

### Parte 0 — Testes sintéticos

Parte 0 valida o pipeline completo antes de tocar em dados reais, usando vídeos sintéticos de elipses em movimento com oclusão real. É um teste de sanidade: gera sequências em segundos e mostra que o gerador, simulador e métricas de tracking funcionam.

**Artefatos implementados:**
1. **Gerador:** `pa2/synthetic_video/synthetic.py` → `generate_synthetic_sequence()` gera vídeos 128×128 de 30 a 60 quadros, com 5 a 15 elipses em movimento, tamanhos variados, ruído e contraste variáveis. Expõe como parâmetros: número de objetos, velocidade típica e duração da oclusão. As elipses são desenhadas com ordem de profundidade (depth), de modo que uma passe atrás da outra e realmente desapareça — é oclusão real, não apenas ocultação superficial.
2. **Simulador de detector:** `SimulatedDetector` recebe as caixas verdadeiras e as estraga de propósito: descarta p% delas, adiciona ruído gaussiano nas coordenadas, injeta falsos positivos (Poisson). Com isso é possível validar a Parte 1 inteira no sintético antes de baixar o MOT17.
3. **Métrica e seus testes:** `pa2/metrics/tracking.py` implementa IDF1 (em caixas, atribuição global por Hungarian), ID switches e fragmentações (matching quadro a quadro estilo CLEAR-MOT). Os 3 casos construídos à mão estão em `pa2/metrics/cases.py` (valores esperados derivados à mão, também verificados em `tests/test_metrics.py`):
   - (a) predição = ground truth ⇒ IDF1 = 1, 0 switches, 0 fragmentações
   - (b) ids 1 e 2 trocados a partir do quadro 16 ⇒ IDF1 = 2/3, 2 switches
   - (c) track partida no quadro 10 e sem detecção nos quadros 12–14 ⇒ IDF1 = 156/177 ≈ 0,881 (≠ de (b)), 1 switch, 1 fragmentação
4. **Baseline no piso fácil:** `_check_baseline_easy()` roda o matching guloso (IoU 0,3, `max_age` 5) com 3 elipses lentas sem oclusão e detector perfeito em 5 seeds; exige IDF1 ≥ 0,99. Depois, `_run_parameter_sweep()` gira um botão por vez (`n_objects`, `velocity_scale`, `occlusion_duration`) a partir de uma referência (6 objetos, velocidade 1, sem oclusão), com 10 seeds por ponto, e mostra IDF1, razão ids previstos/verdadeiros e ID switches por identidade.

- **Onde:** `pa2/synthetic_video/synthetic.py` (gerador + simulador de detector), `pa2/metrics/tracking.py` (métricas), `pa2/association/matching.py` (`GreedyMatcher`), `pa2/part0.py` (pipeline da Parte 0)
- **Como reproduzir:** `uv run pa2 0`
- **Saídas:**
  - `outputs/parte0_occlusion_demo.png` — tira de quadros do alvo escondido por N quadros + curva de visibilidade
  - `outputs/parte0_baseline_easy.png` — frame com GT e predições do baseline
  - `outputs/parte0_parameter_sweep.png` — IDF1 e razão ids previstos/verdadeiros vs. oclusão, velocidade e nº de objetos
  - `outputs/parte0_sweep_results.json` — dados brutos da varredura
  - `outputs/metrics/parte0_baseline_metrics.csv` — métricas do baseline
- **Status:** Implementado e executado. Métricas validadas nos 3 casos de teste.

### Parte 1 — Baseline por quadro

Detecções congeladas + associação ingênua por IoU; nada é treinado. Duas fontes de detecção: as públicas do MOT17 (`det/det.txt`; escolhida: **SDP**) e o Faster R-CNN pré-treinado do torchvision (classe `person`, só inferência, NMS próprio). Associação: IoU entre a última caixa de cada track e as detecções do quadro, guloso ou Hungarian, limiar fixo, id novo quando nada casa, track morta após `max_age` quadros sem observação (regras completas em `pa2/association/tracker.py` e em `RELATORY_PART1.md`).

- **Onde:** `pa2/mot17/` (loader, avaliação com distratores), `pa2/association/tracker.py` (rastreador), `pa2/detection/` (NMS próprio, detector torchvision), `pa2/metrics/` (IDF1/switches/fragmentações e AP/mAP), `pa2/part1.py` (pipeline)
- **Como reproduzir:** `uv run pa2 1` (≈90 s, só com o pacote de anotações de ~10 MB em `data/MOT17/`). Testes: `uv run pytest`.
- **Detector torchvision:** precisa das imagens (pacote completo de ~5,5 GB em `data/MOT17/`) e, na prática, de GPU. Ligue com `use_torchvision_detector: true` em `parte1` do `config.yaml`; as detecções ficam em cache em `outputs/detections/torchvision/`.
- **Saídas** (`outputs/`):
  - `parte1_sequences.csv` — estatísticas dos vídeos (densidade, câmera, visibilidade)
  - `parte1_detector_comparison.csv` — DPM vs FRCNN vs SDP (só treino) e a escolha
  - `parte1_association_variants.csv` — 24 variantes da regra de associação
  - `parte1_per_sequence_public.csv` — IDF1, ID switches, fragmentações, erro de contagem, AP/mAP por vídeo
  - `parte1_descolamento.png` — gráfico obrigatório (mAP e IDF1 / razão de ids e switches por id)
  - `parte1_summary.json` — configuração escolhida
- **Split por vídeo** (nunca por quadro nem por detector): treino 02, 04, 05, 10, 11; validação 09 (câmera parada, esparso) e 13 (câmera móvel, alta rotatividade de identidades). O `test/` do MOT17 não tem GT.
- **Status:** Implementado com as detecções públicas. **Pendente:** rodar o detector torchvision (precisa das imagens/GPU).

### Parte 2 — Trilha A: RNN como modelo de movimento

Escolha da dupla: **Trilha A** (RNN como modelo de movimento). A fonte de detecções fica congelada a partir daqui, o que muda é o que acontece entre os quadros. Um estado recorrente por track. A cada quadro, o LSTM/GRU recebe a última observação (caixa, opcionalmente confiança e Δt) e prevê a caixa do quadro seguinte; a associação usa IoU entre caixa prevista e caixa observada. Sob oclusão, o estado roda para frente sem observação, e a track sobrevive ou não. Perda L1/smooth-L1 sobre a caixa, treinada em trajetórias do ground truth.

- **Onde:** `pa2/models/motion_rnn.py` (modelo + treino), `pa2/association/motion.py` (modelos de movimento intercambiáveis: caixa parada, velocidade constante, RNN), `pa2/association/motion_tracker.py` (mesma gestão de tracks da Parte 1, mas comparando com a caixa *prevista*), `pa2/mot17/trajectories.py` (trajetórias do GT → janelas de treino), `pa2/part2.py` (pipeline)
- **Como reproduzir:** `uv run pa2 2` (≈6 min em CPU, só com as anotações de `data/MOT17/`). `uv run pa2 2 --eval-only` reavalia o checkpoint sem treinar. Relatório: `RELATORY_PART2.md`.
- **Saídas** (em `outputs/`): `checkpoints/parte2_motion_rnn.pt`, `parte2_treino.png`, `parte2_gap_rollout.png` (IoU após k quadros sem observação), `parte2_comparacao.png` (Parte 1 × velocidade constante × RNN, mesmas sequências), `parte2_per_sequence*.csv`, `parte2_summary.json`
- **Status:** Implementado e avaliado nas detecções públicas (SDP), com análise de reconexão depois de buracos e resposta sobre janelas em `RELATORY_PART2.md`. Pendente: incerteza/portão adaptativo (opcional).

### Parte 3 — Ablação (Eixo 2: regime de treino)

Escolhemos o **Eixo 2** (teacher forcing → scheduled sampling → free-running). Na inferência o modelo se alimenta das próprias previsões (e, sob oclusão, só delas); se nunca viu isso no treino, a distribuição muda debaixo dele. Medimos essa deriva e incluímos gradient clipping ligado/desligado. 3 seeds (42, 123, 456), média ± desvio amostral. O modelo é o da Parte 2 (GRU 64) e a detecção, a mesma (SDP congelado); só o regime de treino varia.

7 configurações (21 treinos): `teacher_forcing`, `scheduled_sampling` (probabilidade de observação decai linearmente de 1 a 0), `free_running`, cada um com clipping ligado e desligado, mais `part2_recipe` (teacher forcing + buracos de observação simulados, a receita da Parte 2 re-treinada sob o mesmo protocolo). Os três regimes puros usam `gap_prob: 0` para isolar o efeito. Todos treinam as mesmas épocas e o checkpoint é o da **última** época (sem escolher a época pela validação).

- **Onde:** `pa2/ablation.py` (runner, avaliação, agregação e figuras); `pa2/models/motion_rnn.py` (agenda de teacher forcing, estatísticas de gradiente, detecção de divergência, `eval_shift`); regimes em `pa2/config.yaml` → `parte3.ablation.regimes`
- **Como reproduzir:** `uv run pa2 3` (≈25 min em CPU; retomável: pula o que já existe em `outputs/parte3_ablation/runs/`). Para dividir em processos: `uv run pa2 3 --seeds 42`, `--seeds 123`, `--seeds 456`, e depois `uv run pa2 3 --aggregate-only`. `--regimes nome1 nome2` roda só alguns regimes.
- **O que mede:** deriva (IoU com observações × só com as próprias previsões), IoU após k quadros às cegas, rastreamento nos 7 vídeos (validação = 09 e 13 é o principal), reconexão depois de buracos, e estabilidade do treino (norma do gradiente, passos não finitos, divergência).
- **Saídas** (em `outputs/parte3_ablation/`): `parte3_summary.csv` (média ± desvio por regime), `parte3_runs.csv` (por seed), `parte3_reconnection.csv`, `parte3_tracking.png`, `parte3_shift.png`, `parte3_blind_rollout.png`, `parte3_grad_norms.png`, `runs/*.json` (brutos), `checkpoints/`
- **Relatório:** `RELATORY_PART3.md`
- **Status:** Implementado e executado (21 treinos). Resumo: free-running é claramente pior; teacher forcing, scheduled sampling e a receita da Parte 2 não se distinguem no IDF1; sem clipping não há instabilidade neste setup. Detalhes e limitações em `RELATORY_PART3.md`.

### Parte 4 — Galeria de falhas e horizonte de memória

Três trechos em que o modelo final erra feio, cada um com a figura (tira de quadros com ground truth e predição coloridos por identidade, mais o mapa intermediário relevante — caixa prevista pela recorrência) e um diagnóstico escrito. Obrigatório: medir o horizonte de memória efetivo do modelo, das duas formas:
1. Analítica: a norma de ∂L_t/∂h_{t−k} em função de k
2. Empírica: quantos quadros o estado sobrevive a uma oclusão antes de a track morrer ou trocar de ID

E fazer uma correção: escolher um dos diagnósticos, implementar a mudança que ele sugere, mostrar o antes/depois.

- **Onde:** `pa2/part4.py` (pipeline), `pa2/analysis/memory.py` (gradiente, oclusões injetadas, duração dos buracos), `pa2/analysis/gallery.py` (escolha e figuras das falhas), `pa2/analysis/runner.py` (rastreador com caixas previstas guardadas), `pa2/metrics/reconnection.py`
- **Como reproduzir:** `uv run pa2 4` (usa `outputs/checkpoints/final_motion_rnn.pt`; ≈40 min em CPU do zero, a etapa da correção é retomável). Relatório: `RELATORY_PART4.md`.
- **Saídas** (em `outputs/final_parte4/`):
  - `parte4_gradiente.png|csv|json` — horizonte analítico ||∂L_t/∂h_{t−k}||, com observações e às cegas, e comparação com outros regimes
  - `parte4_sobrevivencia.png|csv`, `parte4_horizonte_empirico.json`, `parte4_oclusoes_injetadas.csv`, `parte4_buracos_*.csv` — horizonte empírico e duração das oclusões do dataset
  - `parte4_falha_oclusao_longa.png`, `parte4_falha_buraco_curto_camera_movel.png`, `parte4_falha_troca_entre_pessoas.png`, `parte4_falhas.json`, `parte4_trechos.csv` — galeria de falhas (caixas sobre fundo vazio: o pacote de anotações não traz imagens)
  - `parte4_correcao.png|csv`, `parte4_extrapolacao.csv` — correção testada (amortecer a extrapolação às cegas): **não melhorou**; o relatório explica o que isso revela sobre o diagnóstico
- **Status:** Implementado e executado. Resultado principal: o estado sobrevive a ~21 quadros de oclusão (N50) contra 16 da caixa parada; 5% dos buracos reais de detecção passam disso.

### Parte 5 — Teste de estresse (queda de taxa de quadros)

Escolhemos o teste de **queda de taxa de quadros**: o vídeo é subamostrado a 1/2 e 1/5 (em todas as fases) e o **modelo final roda sem retreinar**; curva de degradação do IDF1, mais a resposta a "por que quebra quando Δt muda?" e "alimentar Δt resolveria?". Um retreino multi-Δt entra como experimento **exploratório e rotulado** (fora da regra "sem retreinar").

- **Onde:** `pa2/part5.py` (pipeline), `pa2/stress/subsample.py` (subamostragem do vídeo), `pa2/stress/diagnostics.py` (diagnósticos em trajetórias do GT)
- **Como reproduzir:** `uv run pa2 5` (≈6 min em CPU, só com as anotações; retomável). Relatório: `RELATORY_PART5.md`.
- **Saídas** (`outputs/final_parte5/`): `parte5_idf1_fixa.png` e `parte5_idf1_casada.png` (curva principal e variante com `max_age` casado), `parte5_idf1_camera.png` (por tipo de câmera), `parte5_por_video.png`, `parte5_estrutura.png`, `parte5_diagnosticos.png`, `parte5_multidt.png` (exploratório), `parte5_resumo.csv`/`.json`, `parte5_curva.csv` (todas as células), `parte5_diag_*.csv`, `parte5_multidt*.csv`, `checkpoints/` (modelos multi-Δt)
- **Status:** Implementado e executado. Resumo: em 1/5 a RNN perde ~24% de IDF1 na validação (caixa parada 36%, velocidade constante 15%); a perda está nos vídeos de **câmera móvel**; alimentar Δt sem treino piora; treinar com Δt variado melhora só dentro da amostra.

---

## Arquivos do repositório

```
deep-learning-assignment-2/
├── pyproject.toml              # Dependências e comandos uv
├── uv.lock                     # Lockfile determinístico
├── .gitignore                  # Ignora .venv, caches e notebooks (data/ e outputs/ estão versionados)
├── README.md                   # Este arquivo
├── PLANO_DE_EXECUCAO.md        # Plano de execução, decisões e backlog
├── RELATORY_PART0.md ... RELATORY_PART4.md   # Relatório de cada parte: desenho, resultados, limitações
├── PA2.md / PA2.pdf            # Enunciado (transcrição em .md e original)
├── AI_LOG.md                   # Log de uso de IA (entregável)
├── CLAUDE.md                   # Instruções para agentes (aponta para PA2.md e os relatórios)
├── metrics.py                  # Entregável: re-exporta as métricas de pa2/metrics/tracking.py
├── inferencia.ipynb            # Entregável: inferência sobre uma sequência qualquer (usa pa2/inference.py)
├── tests/                      # pytest: métricas, gerador, detecção/NMS, rastreadores, RNN, ablação, Parte 4
├── data/MOT17/                 # Anotações do MOT17 (train/ com GT e det.txt; test/ só det.txt)
├── outputs/                    # Resultados versionados (ver "Onde estão as saídas" abaixo)
│
└── pa2/
    ├── main.py                 # Ponto de entrada: `uv run pa2 <parte>`
    ├── config.py / config.yaml # Dataclasses e configuração por parte (parte0..parte4)
    ├── part0.py                # Parte 0: testes sintéticos
    ├── part1.py                # Parte 1: baseline por quadro (detector público, associação por IoU)
    ├── part2.py                # Parte 2: RNN como modelo de movimento (Trilha A)
    ├── ablation.py             # Parte 3: ablação do regime de treino (Eixo 2)
    ├── part4.py                # Parte 4: horizonte de memória, galeria de falhas, correção
    ├── part5.py                # Parte 5: teste de estresse (queda de taxa de quadros)
    ├── inference.py            # Inferência sobre uma pasta de sequência (tracks, contagem, vídeo); usada pelo notebook
    │
    ├── utils/                  # seed, device, exportação de métricas, gráficos (herdados do PA1)
    ├── synthetic_video/        # Gerador de vídeos sintéticos + simulador de detector (Parte 0)
    ├── metrics/
    │   ├── tracking.py         # IDF1, ID switches, fragmentações (implementação própria)
    │   ├── detection.py        # AP/mAP por quadro e remoção de detecções sobre distratores
    │   ├── reconnection.py     # Desfecho da identidade depois de um buraco de rastreamento
    │   └── cases.py            # Casos (a)(b)(c) feitos à mão, com valores esperados
    ├── association/
    │   ├── matching.py         # GreedyMatcher (Parte 0)
    │   ├── tracker.py          # IoUTracker: baseline ingênuo das Partes 1+
    │   ├── motion.py           # Modelos de movimento: caixa parada, velocidade constante, RNN
    │   └── motion_tracker.py   # MotionTracker: mesma gestão de tracks, compara com a caixa prevista
    ├── mot17/                  # loader.py, evaluate.py (convenções do benchmark), trajectories.py (janelas de treino)
    ├── detection/              # nms.py (próprio) e torchvision_person.py (detector pré-treinado, inferência)
    ├── models/motion_rnn.py    # MotionRNN (RNN/GRU/LSTM), treino, validação, checkpoints
    ├── analysis/               # Parte 4: memory.py (gradiente, oclusões injetadas), gallery.py, runner.py
    └── stress/                 # Parte 5: subsample.py (subamostragem 1/k do vídeo), diagnostics.py
```

**Notebook de inferência:** `inferencia.ipynb` (na raiz), apoiado em `pa2/inference.py`.

**Onde estão as saídas (`outputs/`):**
- `parte0_*`, `parte1_*`, `metrics/`: Partes 0 e 1.
- `parte2_*` e `checkpoints/parte2_motion_rnn.pt`: Parte 2 com o **modelo inicial** (treino com buracos simulados).
- `checkpoints/final_motion_rnn.pt`: **modelo final** (teacher forcing, seed 42; `RELATORY_PART2.md` §6).
- `final/`: reavaliação da Parte 2 com o modelo final (mesmos arquivos de `parte2_*`).
- `parte3_ablation/`: ablação (resumo em CSV/PNG; `runs/` brutos; `checkpoints/` dos 21 treinos).
- `final_parte4/`: Parte 4 (modelo final).
- `final_parte5/`: Parte 5 (teste de estresse; `checkpoints/` = modelos multi-Δt exploratórios).

---

## Configuração via `pa2/config.yaml`

O comportamento do pipeline é controlado por `pa2/config.yaml`, que define seções separadas por parte (`parte0` a `parte5`). Cada seção especifica: dados sintéticos ou reais, diretório dos dados, parâmetros do gerador (Parte 0), parâmetros do matcher, parâmetros da RNN (Parte 2+), janela T, teacher forcing ratio, e configurações de ablação.

Exemplo da seção `parte0`:

```yaml
parte0:
  synthetic: true
  n_sequences: 20
  n_frames: 45
  frame_size: 128
  n_objects_min: 5
  n_objects_max: 12
  occlusion_duration_max: 15
  velocity_scale: 1.0
  epochs: 5
  lr: 1.0e-3
  detector_drop_rate: 0.0
  detector_noise: 0.0
  detector_fp_rate: 0.0
```

---

## Regras de engenharia (do enunciado)

**Permitido:** PyTorch; `torchvision.models.detection` (Faster R-CNN, Mask R-CNN) e encoders pré-treinados; albumentations; scipy; sklearn; o código de matching e de métricas que escrevemos no PA1.

**Permitido com condição:** filtro de Kalman de velocidade constante, **apenas** como baseline de comparação. Ele é um baseline honesto e frequentemente difícil de bater; o que não pode é ele ser o modelo temporal da Parte 2. Lá, quem carrega o estado é a rede recorrente.

**Proibido:**
- Rastreadores prontos: SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT, Norfair, motpy, trackers do supervision, `model.track()` do ultralytics, ou qualquer implementação pronta de associação
- Métricas de rastreamento prontas de biblioteca — motmetrics, TrackEval, py-motmetrics. **Implementamos** IDF1 e a contagem de ID switches
- `torchvision.ops.nms` — **implementamos** o NMS
- Clonar uma solução pronta de MOT17. Se usarem ideia de repo ou paper, citem e reescrevam

**O modelo temporal, as perdas, a associação e a gestão de tracks são de autoria de vocês.**

---

## Status atual

| Parte | Status |
|-------|--------|
| Parte 0 — Testes sintéticos | **Concluída e revisada** — gerador com oclusão por profundidade, simulador testado, IDF1/ID switches/fragmentações validados nos 3 casos à mão (`uv run pytest`), baseline fácil com IDF1 ≈ 1 e varredura dos botões. |
| Parte 1 — Baseline por quadro | **Feita com detecções públicas** (SDP) — falta rodar o detector torchvision (imagens + GPU) |
| Parte 2 — Trilha A (RNN movimento) | **Feita** — GRU de movimento + análise de reconexão (`RELATORY_PART2.md`); opcional pendente: incerteza/portão adaptativo |
| Parte 3 — Ablação (Eixo 2) | **Feita** — 7 regimes × 3 seeds, resultados em `RELATORY_PART3.md` |
| Parte 4 — Galeria de falhas + horizonte de memória | **Feita** — gradiente (conferido por diferenças finitas), oclusões injetadas, 3 falhas, correção negativa (`RELATORY_PART4.md`) |
| Parte 5 — Teste de estresse (queda de taxa de quadros) | **Feita** — curva de IDF1, diagnósticos e multi-Δt exploratório (`RELATORY_PART5.md`) |
| Entregáveis | **Prontos:** README (um comando que treina e um que avalia), `metrics.py`, `AI_LOG.md`, `inferencia.ipynb`, checkpoint (`outputs/checkpoints/final_motion_rnn.pt`). Pendente (opcional): detector torchvision da Parte 1 |

---

## Decisões tomadas

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Trilha da Parte 2 | **Trilha A** (RNN movimento) | Mais simples de começar: LSTM/GRU prevê próxima caixa, perda L1 sobre trajetórias do GT, não requer encoder de imagem adicional |
| Eixo de ablação da Parte 3 | **Eixo 2** (regime de treino) | Mais relevante para o problema: na inferência o modelo se alimenta das próprias previsões; se nunca viu isso no treino, há distribution shift |
| Teste de estresse da Parte 5 | **Queda de taxa de quadros** | Conecta diretamente com a pergunta da Parte 2 sobre janela de T quadros e Δt variável |

---

## Notas

- **Aula dia 30/09/2026:** a Parte 0 foi executada e validada; depois foram feitas e documentadas as Partes 1 a 5 e o notebook de inferência (cada uma com o seu `RELATORY_PARTEn.md`).
- **MOT17:** o pacote de anotações já está em `data/MOT17/`; o split por vídeo está em `pa2/mot17/loader.py`. As imagens (~5,5 GB) só são necessárias para o detector do torchvision da Parte 1.
- **Reprodutibilidade:** as Partes 0, 1, 2 (`--eval-only`) e 5 foram reexecutadas do zero numa revisão e reproduzem os arquivos versionados; o treino é determinístico só na mesma máquina e com as mesmas threads (ver "Um comando que treina").

---

*Última atualização: 01/10/2026 — Partes 0 a 5 e `inferencia.ipynb` concluídos.*
