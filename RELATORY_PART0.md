# Relatório — Parte 0 (testes sintéticos)

**Data:** 01/10/2026 (revisão e correção da versão de 30/09)
**Reproduzir:** `uv run pa2 0` (≈20 s) e `uv run pytest` (≈2 s). Saídas em `outputs/`.

A primeira versão da Parte 0 passava nos seus próprios checks mas tinha defeitos reais
(IDF1 por identidade em vez de por caixa, oclusão simulada por "não desenhar", detector
perfeito que via objetos escondidos). Esta versão corrige tudo isso; `part0.py` agora falha
com `assert` se qualquer requisito não for atendido.

## 1. Gerador (`pa2/synthetic_video/synthetic.py`)
- Vídeos 128×128, 30–60 quadros, 5–15 elipses. Parâmetros expostos: `n_objects`,
  `velocity_scale`, `occlusion_duration`, `noise_level`, `contrast_scale`.
- As elipses são pintadas em ordem de profundidade (maior `z` por cima). A visibilidade de
  cada objeto em cada quadro é medida em pixels (fração dos seus pixels que sobra visível).
- Oclusão roteirizada: um alvo pequeno cruza por trás de um oclusor grande e fica escondido
  por `occlusion_duration` quadros (duração medida com visibilidade < 0,05). Oclusões parciais
  extras acontecem naturalmente.
- Convenção de GT como no MOT17: caixa amodal sempre presente, campo `visibility`, `conf = 0`
  quando `visibility < 0.3` (ignorado na avaliação). Um detector perfeito só vê `conf = 1`.
- **Verificação:** `outputs/parte0_occlusion_demo.png` (alvo ID 5 some por 15 quadros e volta;
  curva de visibilidade abaixo). `tests/test_synthetic.py` confere N ∈ {5, 10, 20} × 5 seeds.

## 2. Simulador de detector
Descarta p%, ruído gaussiano nas coordenadas, falsos positivos (Poisson por quadro).
Verificado com 4000 caixas (pedido → medido): descarte 0,20 → 0,199; ruído 2,0 px → 1,98 px;
FP/quadro 0,50 → 0,46. Cada detecção traz `src_id` e `is_fp`. O mesmo teste está em
`tests/test_synthetic.py`.

## 3. Métricas (`pa2/metrics/tracking.py`, re-exportadas em `metrics.py`)
- **IDF1** em caixas: `2·IDTP / (2·IDTP + IDFP + IDFN)`, com atribuição global um-para-um
  (Hungarian) maximizando IDTP.
- **ID switches / fragmentações:** matching quadro a quadro com continuidade (estilo CLEAR-MOT);
  um switch é uma identidade GT casada com um id diferente do último; fragmentação é
  rastreado → perdido → rastreado.
- Também: erro de contagem de identidades, MOTA (opcional), MT/ML.

Três casos à mão (3 objetos parados, 30 quadros), valores esperados derivados em
`pa2/metrics/cases.py`:

| Caso | IDF1 esperado = obtido | ID sw | Frag |
|---|---|---|---|
| (a) pred = GT | 1,0000 | 0 | 0 |
| (b) ids 1↔2 trocados do quadro 16 | 120/180 = 0,6667 | 2 | 0 |
| (c) track 1 partida no quadro 10, sem detecção nos quadros 12–14 | 156/177 = 0,8814 | 1 | 1 |

(b) e (c) têm IDF1 diferentes, como o enunciado antecipa. Na versão anterior (b) dava 1,0,
o que estava errado.

## 4. Baseline no piso fácil
Associação gulosa por IoU (limiar 0,3, `max_age` 5), 3 elipses lentas, sem oclusão, detector
perfeito, 5 seeds: **IDF1 = 1,0000, 0 switches, 0 fragmentações em todas**
(`outputs/metrics/parte0_baseline_metrics.csv`). O mesmo baseline é usado na varredura.

## 5. Onde o baseline quebra (`outputs/parte0_parameter_sweep.png`)
Um botão por vez a partir de {6 objetos, velocidade 1, sem oclusão}, 10 seeds por ponto.

| Botão | IDF1 | ids prev./verd. | ID sw / id |
|---|---|---|---|
| velocidade 0,3 → 4,0 | 1,00 → 0,57 | 1,0 → 7,7 | 0,0 → 9,3 |
| oclusão 0 → ≥3 quadros | 0,98 → ≈0,90–0,95 | 1,05 → 1,2–1,7 | 0,05 → 0,2–0,7 |
| objetos 3 → 15 | 1,00 → 0,95 | 1,03 → 1,19 | 0,03 → 0,25 |

Leituras:
- A **velocidade** é o que derruba o IoU entre quadros consecutivos; acima de ~3 a associação desmonta.
- A **oclusão** quebra a identidade assim que existe (o alvo reaparece deslocado, com outro id),
  mas o efeito é praticamente um degrau em vez de crescer com N, e o IDF1 médio dilui isso
  porque só um objeto por vídeo é ocluído. O efeito aparece melhor em ID switches e na razão
  de ids. O IDF1 não é monotônico em N: com o alvo escondido por muito tempo, os quadros
  escondidos saem da avaliação (`conf = 0`) e o IDF1 só perde `min(antes, depois)` caixas.
- Com mais **objetos** a degradação é suave neste regime (detector perfeito, sem ruído).

## 6. Outros ajustes
- `AI_LOG.md` na raiz; `metrics.py` na raiz; `tests/` com pytest (`uv run pytest`).
- `output_dir` relativo ao diretório de execução (`outputs/`); dados em `data/MOT17/`
  (ignorado pelo git); scripts `pa2-part0`/`pa2-ablation` removidos do `pyproject.toml`
  (apontavam para código inexistente).

## 7. Pontos de atenção para a Parte 1
- No MOT17, filtrar o GT por `conf == 1` e `class == 1` (o `gt.txt` tem distratores).
- `test/` do MOT17 não tem GT: validação e teste saem de `train/` (7 sequências × 3 detectores
  com o mesmo GT).
- O `GreedyMatcher` ainda é a versão ingênua (último box, limiar fixo); `min_hits` ainda não
  filtra a saída. Isso é matéria da Parte 1.
