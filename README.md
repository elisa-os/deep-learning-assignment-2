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

Treina a solução principal (Parte 2 — Trilha A: RNN de movimento) com a configuração padrão de `pa2/config.yaml`:

```bash
cd deep-learning-assignment-2   # ou trabalhe a partir da raiz do repositório
uv run pa2 2
```

Isso treina o LSTM/GRU para prever a próxima caixa do tracking, com teacher forcing, usando as trajetórias do ground truth das sequências de treino. O treinamento usa os hiperparâmetros definidos em `pa2/config.yaml` → `parte2` (30 épocas, lr=1e-3, janela T=32).

Sobrescreva epochs se precisar de uma execução mais curta (ex: validação rápida):

```bash
uv run pa2 2 --epochs 5
```

### Um comando que avalia

Avalia o modelo já treinado sem retreinar, usando o checkpoint salvo:

```bash
uv run pa2 2 --eval-only --checkpoint outputs/checkpoints/parte2_motion_rnn.pt
```

Isso carrega os pesos, roda a inferência nas sequências de validação e teste, e gera:
- `outputs/metrics/parte2_tracking_results.json` — métricas de tracking (IDF1, ID switches, fragmentações)
- `outputs/metrics/parte2_per_sequence_tracking_metrics.csv` — métricas por sequência
- `outputs/parte2_resultados.png` — curvas de treino + gráfico IDF1 por sequência

Para avaliar apenas uma parte específica com o checkpoint dela:

```bash
uv run pa2 0 --eval-only --checkpoint outputs/checkpoints/parte0_baseline.pt
```

### Arquivos entregues junto com o repositório

Além do README.md, o repositório entrega:

- **`AI_LOG.md`** — log de uso de IA neste assignment, conforme exigido pela política de uso de IA do enunciado (seção 5 do PA2.pdf). Descreve episódios em que IA foi usada e como os problemas foram resolvidos.
- **`pa2/inferencia.ipynb`** — notebook de inferência: recebe o caminho de uma sequência MOT17 qualquer e devolve o vídeo com as identidades coloridas de forma consistente e a contagem de objetos únicos, rodando sem retreinar. Usa o checkpoint da Parte 2 (`parte2_motion_rnn.pt`).
- **`outputs/checkpoints/parte2_motion_rnn.pt`** — pesos do modelo temporal treinado (checkpoint). É o artefato que o `inferencia.ipynb` e o comando de avaliação usam.

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

- **Onde:** `pa2/models/motion_rnn.py` (MotionRNN, MotionRNNPredictor), `pa2/part2.py` (pipeline da Parte 2)
- **Como reproduzir:** `uv run pa2 2` (requer MOT17 baixado e configurado em `pa2/config.yaml` → `parte2.sequence_split`)
- **Saídas:**
  - `outputs/checkpoints/parte2_motion_rnn.pt` — checkpoint do modelo treinado
  - `outputs/metrics/parte2_tracking_results.json` — métricas de tracking
  - `outputs/metrics/parte2_per_sequence_tracking_metrics.csv` — métricas por sequência
  - `outputs/parte2_resultados.png` — curvas de treino + gráfico IDF1 por sequência
  - `outputs/parte2_qualitativo.png` — trajetórias coloridas por identidade
- **Status:** A implementar

### Parte 3 — Ablação (Eixo 2: regime de treino)

Escolhemos o **Eixo 2 — o regime de treino** (teacher forcing → scheduled sampling → free-running). Na inferência o modelo se alimenta das próprias previsões (e, sob oclusão, só delas); se ele nunca viu isso no treino, a distribuição muda debaixo dele. Medimos. Incluimos gradient clipping ligado/desligado e reportamos o que acontece sem ele. 3 seeds, média ± desvio.

- **Onde:** `pa2/ablation.py` (script automatizado que roda todas as configurações)
- **Como reproduzir:** `uv run pa2 3`
- **Saídas (em `outputs/parte3_ablation/`):**
  - `outputs/parte3_ablation/checkpoints/` — checkpoints de cada configuração × seed
  - `outputs/parte3_ablation/metrics/ablation_results.json` — resultados brutos de todas as configurações
  - `outputs/parte3_ablation/metrics/ablation_summary.csv` — tabela agregada (μ ± σ) por configuração
  - `outputs/parte3_ablation_idf1_comparison.png` — gráfico de barras: IDF1 por regime de treino
- **Status:** A implementar

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
| Parte 3 — Ablação (Eixo 2) | A implementar |
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
