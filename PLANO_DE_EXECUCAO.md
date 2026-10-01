# Plano de Execução — PA2: Identidade ao longo do tempo

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Alunos:** Bruno Ferreira & Elisa Soares  
**Entrega:** 02/10, 23h59  
**Status:** Em construção

---

## Visão geral

PA2 tem a mesma arquitetura de projeto que o PA1: `uv` para ambiente, `pyproject.toml` com CLI scripts, `config.yaml` por parte, `config.py` com dataclasses, `utils/` com seed/device/export/visualize, e separação por subpacotes. O que muda é o domínio (vídeo sequencial em vez de imagem estática) e o que se mede (IDF1/ID switches em vez de mAP).

---

## Fase 0 — Estrutura do projeto

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

## Fase 1 — Parte 0: Testes sintéticos (target: até X/2026)

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

## Fase 2 — Download MOT17 + Parte 1 baseline (target: até X/2026)

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

## Fase 3 — Parte 2: Memória temporal (escolher UMA trilha) (target: até X/2026)

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

## Fase 4 — Parte 3: Ablação (escolher 1 eixo, 3 seeds) (target: até X/2026)

**DECISAO PENDENTE: Qual eixo?**

**Eixo 2 — regime de treino (recomendado):**
- Teacher forcing → scheduled sampling → free-running
- Na inferência o modelo se alimenta das próprias previsões (e sob oclusão só delas)
- Medir efeito de distribution shift
- Incluir gradient clipping ligado/desligado e reportar o que acontece sem ele
- 3 seeds, média ± desvio

Implementar `pa2/ablation.py` que roda as 3 configurações com 3 seeds, salva checkpoints e métricas, reporta média ± desvio.

---

## Fase 5 — Parte 4: Galeria de falhas + horizonte de memória (target: até X/2026)

1. Selecionar 3 trechos onde o modelo final erra feio
2. Para cada um:
   - Figura: tira de quadros com GT e predição coloridos por identidade + mapa intermediário relevante (caixa prevista pela recorrência ou matriz de similaridade de embeddings)
   - Diagnóstico escrito: "esse objeto fica ocluído por 34 quadros; minha janela de BPTT é de 10 e a norma do gradiente cai 20× em 8 passos, então o modelo nunca recebeu sinal de supervisão que atravessasse esse buraco"
3. **Horizonte de memória analítico:** norma de ∂L_t/∂h_{t−k} em função de k. Curva de gradiente que some no modelo e nos dados. Se fizeram Eixo 1 da Parte 3, comparar curva da RNN simples com a do modelo com portas.
4. **Horizonte de memória empírico:** quantos quadros o estado sobrevive a uma oclusão antes de track morrer ou trocar de ID. Comparar com distribuição de duração de oclusão do dataset.
5. **Correção:** escolher um diagnóstico, implementar a mudança que ele sugere, mostrar antes/depois. Se não funcionar, explicar o que isso revela sobre o diagnóstico estar errado.

---

## Fase 6 — Parte 5: Teste de estresse (escolher 1) (target: até X/2026)

**DECISAO PENDENTE: Qual teste?**

**Queda de taxa de quadros (recomendado):**
- Avaliar com vídeo subamostrado a 1/2 e 1/5 da taxa original
- Curva de degradação do IDF1
- Por que modelo de movimento aprendido em Δt fixo quebra quando Δt muda?
- Alimentar Δt na recorrência resolveria?

**Qualidade do detector:**
- Degradar detecções propositalmente: descartar p%, adicionar ruído nas caixas, injetar falsos positivos
- 3 intensidades diferentes
- O modelo temporal absorve ou amplifica a falha do detector?
- Reportar mAP e IDF1 juntos

Feito sem retreinar, em cima do modelo final.

---

## Fase 7 — Entregáveis finais (target: até 02/10)

1. **`pa2/inferencia.ipynb`:** recebe caminho de uma sequência qualquer, devolve vídeo com identidades coloridas de forma consistente e contagem de objetos únicos. Roda sem retreinar.
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

## Decisões pendentes

| # | Decisão | Recomendação | Responsável |
|---|---|---|---|
| 1 | Trilha A ou B (Parte 2) | **Decidido: Trilha A** (RNN movimento) | dupla |
| 2 | Eixo de ablação (Parte 3) | Eixo 2 (regime de treino) | dupla |
| 3 | Teste de estresse (Parte 5) | Queda de taxa de quadros | dupla |
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
