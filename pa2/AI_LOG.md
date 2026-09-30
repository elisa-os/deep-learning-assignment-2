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
