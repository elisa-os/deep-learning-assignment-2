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

O projeto usa [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências e comandos. Sincronize o ambiente com:

```bash
cd deep-learning-assignment-2
uv sync
```

Isso cria o ambiente virtual isolado (`.venv`) e instala todas as dependências listadas em `pyproject.toml`, incluindo PyTorch, torchvision, scikit-image, scipy, pandas, matplotlib, albumentations e tqdm. O pacote `pa2` está configurado via `module-root` no `pyproject.toml`; não é necessário uma pasta `src/`.

Comandos disponíveis no CLI:

```bash
uv run pa2 0     # Parte 0 — testes sintéticos (gerador + simulador + métricas + baseline)
uv run pa2 1     # Parte 1 — baseline por quadro (MOT17 ou sintético)
uv run pa2 2     # Parte 2 — RNN memória temporal (Trilha A)
uv run pa2 3     # Parte 3 — ablações (Eixo 2: regime de treino)
```

Ou use os comandos específicos:

```bash
uv run pa2 0  # Parte 0 — teste sintético completo
uv run pa2 3  # Parte 3 — ablações
```

### Download dos dados

O dataset usado é **MOT17 / MOTChallenge** — sequências de pedestres em rua e ambiente interno, com caixas e identidades anotadas quadro a quadro.

**Como baixar:**

Opção 1 — pacote só de anotações (recomendado para começar, ~10 MB, sem conta):
```
https://motchallenge.net/data/MOT17/
```

Baixe o arquivo `MOT17.zip` (ou `MOT17.zip` apenas com anotações) e extraia para:
```
data/MOT17/
```

Ou seja, a estrutura esperada é:
```
data/MOT17/
  train/
    MOT17-02/
      det/
        det.txt        # detecções públicas (DPM/Faster R-CNN/SDP)
      seqinfo.ini
    MOT17-04/
      ...
  test/
    ...
```

O pipeline detecta automaticamente a estrutura acima; não é necessário renomear ou reorganizar.

Após baixar, edite `pa2/config.yaml` → `parte1.sequence_split` e `parte2.sequence_split` para definir quais sequências vão para treino, validação e teste (split por sequência inteira, nunca por quadro).

### Um comando que treina

Treina o **modelo final** (GRU de 64 unidades, teacher forcing puro, seed 42, 20 épocas, checkpoint da última época), que é o `teacher_forcing` da ablação da Parte 3:

```bash
uv run pa2 3 --regimes teacher_forcing --seeds 42      # ≈1–2 min em CPU; grava outputs/parte3_ablation/checkpoints/teacher_forcing_s42.pt
cp outputs/parte3_ablation/checkpoints/teacher_forcing_s42.pt outputs/checkpoints/final_motion_rnn.pt
```

O treino usa as trajetórias do ground truth dos vídeos de treino (`pa2/config.yaml` → `parte3`). O retreino reproduz o regime, mas pode diferir bit a bit do checkpoint entregue (threads/máquina). `uv run pa2 2` treina o modelo inicial da Parte 2 (com buracos simulados e época escolhida pela validação), que não é o final.

Sobrescreva epochs se precisar de uma execução mais curta (ex: validação rápida):

```bash
uv run pa2 2 --epochs 5
```

### Um comando que avalia

Avalia o modelo já treinado sem retreinar, usando o checkpoint salvo:

```bash
uv run pa2 2 --eval-only --checkpoint outputs/checkpoints/final_motion_rnn.pt --output-dir outputs/final
```

Isso carrega os pesos e roda, nos 7 vídeos com GT (treino e validação), a comparação Parte 1 × velocidade constante × RNN e a análise de reconexão depois de buracos. Gera em `outputs/`:
- `parte2_summary.json` — configuração e médias por split
- `parte2_per_sequence.csv` / `parte2_per_sequence_tuned.csv` — métricas por vídeo (IDF1, ID switches, fragmentações, contagem)
- `parte2_comparacao.png` — Parte 1 × velocidade constante × RNN, vídeos ordenados por densidade
- `parte2_reconnection.csv`, `parte2_reconexao.png` — o que acontece com a identidade depois de um buraco

Para avaliar apenas uma parte específica com o checkpoint dela:

```bash
uv run pa2 0 --eval-only --checkpoint outputs/checkpoints/parte0_baseline.pt
```

### Arquivos entregues junto com o repositório

Além do README.md, o repositório entrega:

- **`AI_LOG.md`** — log de uso de IA neste assignment, conforme exigido pela política de uso de IA do enunciado (seção 5 do PA2.pdf). Descreve episódios em que IA foi usada e como os problemas foram resolvidos.
- **`pa2/inferencia.ipynb`** — notebook de inferência: recebe o caminho de uma sequência MOT17 qualquer e devolve o vídeo com as identidades coloridas de forma consistente e a contagem de objetos únicos, rodando sem retreinar. Usa o checkpoint do modelo final (`outputs/checkpoints/final_motion_rnn.pt`).
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

- **Onde:** `pa2/synthetic_video/synthetic.py` (dataset + gerador + simulador), `pa2/metrics/tracking.py` (métricas), `pa2/association/matching.py` (matching guloso e Hungarian), `pa2/part0.py` (pipeline da Parte 0)
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

- **Onde:** `pa2/part4.py` (pipeline da Parte 4)
- **Como reproduzir:** `uv run pa2 4` (requer checkpoint da Parte 2 treinado)
- **Saídas:**
  - `outputs/parte4_failure_gallery.png` — 3 trechos com falhas e diagnósticos
  - `outputs/parte4_memory_horizon.png` — curva de gradiente que some + distribuição de oclusão
  - `outputs/parte4_correction_before_after.png` — antes/depois da correção
- **Status:** A implementar

### Parte 5 — Teste de estresse (queda de taxa de quadros)

Escolhemos o teste de **queda de taxa de quadros**: avaliamos com o vídeo subamostrado a 1/2 e 1/5 da taxa original — curva de degradação do IDF1. Por que um modelo de movimento aprendido em Δt fixo quebra quando Δt muda? Alimentar Δt na recorrência resolveria? Feito sem retreinar, em cima do modelo final.

- **Onde:** `pa2/stress/stress_test.py` (avaliação de subamostragem)
- **Como reproduzir:** `uv run pa2 5` (requer checkpoint da Parte 2 treinado)
- **Saídas:**
  - `outputs/parte5_stress_test.png` — curva de degradação IDF1 vs. taxa de quadros
  - `outputs/parte5_stress_results.json` — resultados quantitativos
- **Status:** A implementar

---

## Arquivos do repositório

```
deep-learning-assignment-2/
├── pyproject.toml              # Dependências e comandos uv
├── uv.lock                     # Lockfile determinístico
├── .gitignore                  # Ignora outputs/, data/, cache, etc.
├── README.md                   # Este arquivo
├── PLANO_DE_EXECUCAO.md        # Plano de execução detalhado
├── RELATORY_PART0.md           # Relatório de execução da Parte 0
├── PA2.md / PA2.pdf            # Enunciado (transcrição em .md e original)
├── AI_LOG.md                   # Log de uso de IA (entregável)
├── metrics.py                  # Entregável: re-exporta as métricas de pa2/metrics/tracking.py
├── tests/                      # pytest: métricas, gerador, simulador de detector
├── data/MOT17/                 # Dados do MOT17 (ignorado pelo git)
│
└── pa2/
    ├── __init__.py             # Pacote principal
    ├── main.py                 # Ponto de entrada: CLI por parte
    ├── config.py               # Dataclasses de configuração
    ├── config.yaml             # Configuração por parte (parte0..parte5)
    ├── part0.py                # Pipeline da Parte 0 (testes sintéticos)
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── seed.py             # set_seed() (copiado do PA1)
    │   ├── device.py           # get_device() (copiado do PA1)
    │   ├── export.py           # PerSequenceMetricsWriter (adaptado do PA1)
    │   └── visualize.py        # Gráficos de tracking (adaptado do PA1)
    │
    ├── metrics/
    │   ├── __init__.py
    │   ├── tracking.py          # IDF1, ID switches, fragmentações (implementação própria)
    │   └── cases.py             # Casos (a)(b)(c) feitos à mão, com valores esperados
    │
    ├── synthetic_video/
    │   ├── __init__.py
    │   └── synthetic.py          # Gerador + simulador de detector (implementação própria)
    │
    ├── association/
    │   ├── __init__.py
    │   ├── matching.py           # GreedyMatcher, HungarianMatcher (Parte 0)
    │   └── tracker.py            # IoUTracker: baseline ingênuo da Parte 1
    │
    ├── mot17/                    # Loader MOT17 + avaliação com distratores
    │   ├── __init__.py
    │   ├── loader.py
    │   └── evaluate.py
    │
    ├── detection/                # NMS próprio + detector torchvision (inferência)
    │   ├── nms.py
    │   └── torchvision_person.py
    │
    ├── part1.py                  # Pipeline da Parte 1
    │
    ├── models/                   # (A implementar) RNN de movimento
    │   ├── __init__.py
    │   └── motion_rnn.py
    │
    ├── ablation.py               # (A implementar) Runner de ablações
    ├── part1.py                  # (A implementar) Pipeline da Parte 1
    ├── part2.py                  # (A implementar) Pipeline da Parte 2
    ├── part4.py                  # (A implementar) Pipeline da Parte 4
    ├── part5.py                  # (A implementar) Pipeline da Parte 5
    ├── stress/                   # (A implementar) Testes de estresse
    │   ├── __init__.py
    │   └── stress_test.py
    └── inferencia.ipynb          # (A implementar) Notebook de inferência
```

---

## Configuração via `pa2/config.yaml`

O comportamento do pipeline é controlado por `pa2/config.yaml`, que define seções separadas por parte (`parte0`, `parte1`, `parte2`, `parte3`). Cada seção especifica: dados sintéticos ou reais, diretório dos dados, parâmetros do gerador (Parte 0), parâmetros do matcher, parâmetros da RNN (Parte 2+), janela T, teacher forcing ratio, e configurações de ablação.

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
| Parte 2 — Trilha A (RNN movimento) | A implementar |
| Parte 3 — Ablação (Eixo 2) | **Feita** — 7 regimes × 3 seeds, resultados em `RELATORY_PART3.md` |
| Parte 4 — Galeria de falhas + horizonte de memória | A implementar |
| Parte 5 — Teste de estresse (queda de taxa de quadros) | A implementar |
| Entregáveis (README, AI_LOG, inferencia.ipynb, checkpoints) | A completar |

---

## Decisões tomadas

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Trilha da Parte 2 | **Trilha A** (RNN movimento) | Mais simples de starting: LSTM/GRU prevê próxima caixa, perda L1 sobre trajetórias do GT, não requer encoder de imagem adicional |
| Eixo de ablação da Parte 3 | **Eixo 2** (regime de treino) | Mais relevante para o problema: na inferência o modelo se alimenta das próprias previsões; se nunca viu isso no treino, há distribution shift |
| Teste de estresse da Parte 5 | **Queda de taxa de quadros** | Conecta diretamente com a pergunta da Parte 2 sobre janela de T quadros e Δt variável |

---

## Notas

- **Aula dia 30/09/2026:** a Parte 0 foi executada e validada. O gerador de vídeos sintéticos gera sequências com oclusão real. O simulador de detector aplica drop, ruído e FPs. As métricas (IDF1, ID switches, fragmentações) passam nos 3 casos de teste. O baseline no piso fácil tem IDF1≈1.
- **Varredura de parâmetros:** demonstra a degradação do baseline ingênuo com detector ruidoso. Os resultados são esperados — o matcher não foi otimizado para condições adversas.
- **MOT17:** o pacote só de anotações (~10 MB) deve ser baixado antes de começar a Parte 1. O split por sequência deve ser definido em `pa2/config.yaml` → `parte1.sequence_split` e `parte2.sequence_split`.
- **Divisão de trabalho:** a dupla deve definir contratos de interface antes de começar cada parte para evitar bloqueios mútuos.

---

*Última atualização: 30/09/2026 — Parte 0 concluída e documentada.*
