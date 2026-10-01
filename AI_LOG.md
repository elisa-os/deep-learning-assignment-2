# AI_LOG — Registro de Uso de Inteligência Artificial

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Alunos:** Bruno Ferreira & Elisa Soares  
**Assignment:** Programming Assignment 2 (PA2) — Identidade ao longo do tempo: detecção, recorrência e rastreamento

Este documento registra os episódios de utilização de ferramentas de Inteligência Artificial durante o desenvolvimento do PA2, conforme orientado na seção 5 do enunciado.

---

## Episódio 1: Planejamento e estrutura do projeto

- **Data:** 30/09/2026
- **Ferramenta:** Antigravity / LLM
- **Contexto e Motivação:**
  - Definir a estrutura do projeto PA2, copiando e adaptando a estrutura do PA1
  - Definir decisões de design (Trilha A, Eixo 2, queda de taxa de quadros)
  - Criar o plano de execução sequencial
- **Como a IA auxiliou:**
  - Analisou o PA2.pdf e o PA1 completo para mapear o que seria reutilizado e o que seria criado do zero
  - Produza o PLANO_DE_EXECUCAO.md com todas as fases, decisões pendentes e arquivos a serem criados
  - Criou a estrutura de diretórios e todos os arquivos base (`pyproject.toml`, `config.yaml`, `config.py`, `utils/`, `metrics/`, `synthetic_video/`, `association/`, `main.py`)
- **Validação / Decisões Humanas:**
  - Revisão das decisões de design (Trilha A, Eixo 2, queda de taxa de quadros) — confirmadas pela dupla
  - Revisão da estrutura de pacotes e contratos de interface para garantir que Marcela possa continuar a partir deles

---

## Episódio 2: Implementação da Parte 0 — Gerador sintético e simulador de detector

- **Data:** 30/09/2026
- **Ferramenta:** Antigravity / LLM
- **Contexto e Motivação:**
  - Implementar o gerador de vídeos sintéticos com oclusão real (ordem de profundidade)
  - Implementar o simulador de detector com drop, ruído e FPs
  - Implementar o dataset PyTorch (`SyntheticVideoDataset`)
- **Como a IA auxiliou:**
  - Implementou `generate_synthetic_sequence()` com desenho de elipses rotacionadas usando `skimage.draw.ellipse`
  - Implementou a lógica de oclusão: uma elipse é posicionada atrás de outra e só é desenhada se não estiver coberta
  - Implementou `SimulatedDetector` com 3 fontes de erro (drop, ruído gaussiano, FPs Poisson)
- **Validação / Decisões Humanas:**
  - Validação visual da oclusão: o GT da elipse em oclusão continua presente nos frames de oclusão (comportamento correto — o GT reflete a posição real, não o que é visível)
  - A oclusão visual é demonstrada pela figura `parte0_occlusion_demo.png`

---

## Episódio 3: Implementação das métricas de tracking (IDF1, ID switches, fragmentações)

- **Data:** 30/09/2026
- **Ferramenta:** Antigravity / LLM
- **Contexto e Motivação:**
  - Implementar IDF1, ID switches e fragmentações sem usar bibliotecas prontas (motmetrics, TrackEval)
  - Validar com 3 casos construídos à mão exigidos pelo enunciado
- **Como a IA auxiliou:**
  - Implementou `compute_idf1()` com matching global (Hungarian via scipy.optimize.linear_sum_assignment)
  - Implementou `count_id_switches()` com matching frame a frame (IoU guloso) — comportamento consistente com MOTChallenge
  - Implementou `count_fragmentations()` com matching frame a frame
  - Implementou `evaluate_tracking_sequence()` que retorna todas as métricas
- **Validação / Decisões Humanas:**
  - Caso (a): pred = GT → IDF1=1.0000, sw=0, frag=0 ✓
  - Caso (b): troca de IDs a partir do frame 16 → sw=2 (correto: uma troca para cada GT identity) ✓
  - Caso (c): track dividida em duas a partir do frame 10, com omissão de detecção nas frames 12-14 → IDF1=0.8571, sw=1, frag=1 ✓
  - Decisão: ID switches e fragmentações são computados com matching frame a frame, não com matching global. Isso é consistente com a definição MOTChallenge.

---

## Episódio 4: Correção do bug do matcher (detecções não-casadas não emitidas no output)

- **Data:** 30/09/2026
- **Ferramenta:** Antigravity / LLM
- **Contexto e Motivação:**
  - O baseline no piso fácil tinha IDF1=0.6667 em vez de ≈1.0
  - Debug identificou que o `GreedyMatcher.run()` só emitia no output as detecções casadas com tracks existentes — as detecções que criavam novos tracks eram silenciosamente descartadas
  - Isso gerava tracks fantasmas e fragmentação artificial (9 pred_ids únicos para 3 objetos)
- **Como a IA auxiliou:**
  - Diagnosticou o problema via análise frame a frame
  - Reescreveu o `match_frame()` para retornar `(track_id, det_idx, iou, is_new_track)` e o `run()` para emitir todas as detecções no output
  - Corrigiu também o `HungarianMatcher.run()` com a mesma lógica
- **Validação:**
  - Após correção: IDF1=1.0000, sw=0, frag=0 com detector perfeito e 3 objetos
  - 3 pred_ids únicos para 3 objetos (era 15 antes da correção)
  - 90 tracks gerados para 3 objetos × 30 frames (era 80 antes)

---

## Episódio 5: Escrita do README e documentação da Parte 0

- **Data:** 30/09/2026
- **Ferramenta:** Antigravity / LLM
- **Contexto e Motivação:**
  - Documentar a Parte 0 implementada conforme o padrão do README do PA1 (ambiente → download → treino → avaliação → detalhamento por parte)
  - Escrever AI_LOG.md conforme exigência do enunciado
- **Como a IA auxiliou:**
  - Escreveu o README.md completo com todas as seções obrigatórias
  - Escreveu o AI_LOG.md com 5 episódios documentados
  - Documentou os outputs gerados, decisões de design e estado atual de cada parte
- **Validação / Decisões Humanas:**
  - Revisão das seções de ambiente, download, comandos de treino/avaliação para garantir que estão corretas e completas
  - Revisão do AI_LOG para garantir que os episódios estão documentados de forma concisa e factual

---

## Episódio 6: Revisão crítica e correção da Parte 0

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Antes de começar a Parte 1, pedimos uma revisão da Parte 0. O relatório anterior dava a Parte 0 como concluída, mas havia sinais de erro: IDF1 = 1.0 no caso (b) (troca de identidades), baseline fácil em 0.67 e detecções iguais ao GT.
- **Como a IA auxiliou:**
  - Rodou o pipeline, abriu as figuras e leu o código. Diagnosticou: (1) o IDF1 contava identidades, não caixas, e por isso não penalizava a troca do caso (b); (2) a "oclusão" era um hack (a elipse alvo deixava de ser desenhada) e a figura mostrava o alvo visível dentro da janela de oclusão; (3) o detector "perfeito" enxergava objetos escondidos, então o sweep não media oclusão; (4) o simulador de detector nunca era testado com erro ligado; (5) o CSV de métricas tinha valores fixos (1.0); (6) arquivos fora do lugar.
  - Reescreveu `metrics/tracking.py` (IDF1 em caixas com Hungarian; ID switches e fragmentações com matching estilo CLEAR-MOT), o gerador (pintura em ordem de profundidade, visibilidade medida por pixel, oclusão roteirizada de N quadros, ruído e contraste) e `part0.py` com asserts. Criou `tests/` e `pa2/metrics/cases.py`.
- **Validação / Decisões Humanas:**
  - Os valores esperados dos casos (b) e (c) (IDF1 = 2/3 e 156/177) foram derivados à mão a partir das contagens de caixas (ver docstring de `pa2/metrics/cases.py`) e só depois comparados com a saída do código. Conferimos as figuras de oclusão e a varredura.

---

## Episódio 7: Parte 1 (baseline por quadro)

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Implementar a Parte 1 com cautela, usando só o pacote de anotações (~10 MB) e deixando o detector torchvision para quando houver imagens/GPU.
- **Como a IA auxiliou:**
  - Explorou o `gt.txt` e achou convenções que mudam a avaliação: GT contínuo (oclusão só na `visibility`), classes distratoras (2, 7, 8, 12), scores saturados em 1,0 no FRCNN/SDP e GT idêntico entre os três detectores (⇒ split por vídeo).
  - Escreveu loader, NMS, AP/mAP, rastreador por IoU, avaliação, wrapper do torchvision, `part1.py` e testes. Também achou e corrigiu um bug de configuração: a seção `association:` aninhada do `config.yaml` era ignorada em silêncio.
  - Primeiro o `part1.py` não existia (o `uv run pa2 1` falhava); só foi declarado feito depois de rodar de ponta a ponta.
- **Validação / Decisões Humanas:**
  - Testes que usam o GT real: GT como predição ⇒ IDF1 = 1; GT como detecções ⇒ AP50 = 1. Dois testes foram reescritos por serem fracos (asserção tautológica).
  - A escolha do detector (SDP) e da regra de associação seguem regras fixas no código e usam só os vídeos de treino; o detector torchvision continua pendente.

---

## Episódio 8: Parte 2 (Trilha A, RNN de movimento)

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Com só as anotações do MOT17 disponíveis, a Trilha A era a única viável (a B precisa dos recortes das imagens).
- **Como a IA auxiliou:**
  - Propôs a parametrização invariante à escala, o treino com ruído de detector medido (SDP × GT) e buracos de observação simulados, e escreveu `motion_rnn.py`, `motion.py`, `motion_tracker.py`, `part2.py` e os testes.
  - Fez o rastreador novo herdar o `IoUTracker` e verificou que, com movimento "parado", ele reproduz a Parte 1 exatamente (teste no vídeo 09 real).
- **Validação / Decisões Humanas:**
  - Os resultados da RNN foram comparados com a Parte 1 e com um baseline de velocidade constante nas mesmas sequências; a velocidade constante é um baseline forte e o relatório diz isso.
  - Pedimos uma análise do "em que aspecto a RNN melhora": a IA escreveu `metrics/reconnection.py` (desfecho de cada buraco de rastreamento) e confrontou o relatório com os números. A primeira versão do texto atribuía o ganho a "a track sobrevive à oclusão"; a análise mostrou que o ganho está em buracos curtos e na câmera móvel, e que acima de 30 quadros não há ganho. O relatório foi corrigido.

---

## Episódio 9: Parte 3 (Ablação, Eixo 2: regime de treino)

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Fazer a ablação completa (7 configurações × 3 seeds) sem abrir mão de nenhuma resposta do enunciado; as hipóteses foram escritas no plano antes de rodar.
- **Como a IA auxiliou:**
  - Antes de implementar, leu o código de treino da Parte 2 e achou armadilhas que invalidariam a comparação: a validação herdava o `tf_ratio` do treino; a época "melhor" seria escolhida por critérios diferentes em cada regime; passos com perda não finita eram pulados em silêncio; um modelo com NaN faria o rastreador falhar.
  - Estendeu `motion_rnn.py` com padrões idênticos aos da Parte 2 (verificado: a perda e o IoU de validação do checkpoint antigo ficaram exatamente iguais), escreveu `ablation.py`, 14 testes e rodou 21 treinos.
- **Validação / Decisões Humanas:**
  - Os resultados contrariaram duas das três hipóteses (teacher forcing não é pior às cegas na validação; sem clipping nada explode). O relatório registra isso, em vez de ajustar a hipótese depois.
  - Uma das medidas planejadas ("deriva" só com as próprias previsões) se mostrou inútil (igual em todos os regimes) e ficou documentada como resultado negativo.
  - Os runs com e sem clipping no teacher forcing deram resultados idênticos bit a bit, o que foi usado como checagem de que o clipping estava mesmo desligado.
