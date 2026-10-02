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
  - Revisão da estrutura de pacotes e contratos de interface para garantir que a dupla (Bruno e Elisa) possa continuar a partir deles

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
  - Caso (c): track dividida em duas a partir do frame 10, com omissão de detecção nas frames 12-14 → IDF1=156/177≈0.881, sw=1, frag=1 ✓
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

---

## Episódio 10: Escolha do modelo final

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - A ablação da Parte 3 mostrou que a receita da Parte 2 (com buracos simulados) não ganhava do teacher forcing puro. Decidimos usar o teacher forcing como modelo final, desde que não contrariasse o enunciado.
- **Como a IA auxiliou:**
  - Conferiu o enunciado e apontou três condições: o comando que treina tem de reproduzir o modelo final; a comparação da Parte 2 tem de ser refeita com ele; e o critério de escolha usou os vídeos de validação ("nunca vista"), o que ficou declarado como viés.
  - Reavaliou o modelo final (`outputs/final/`) sem sobrescrever os resultados anteriores (novo `--output-dir`).
- **Validação / Decisões Humanas:**
  - A reavaliação trouxe um resultado menos favorável: com cada método usando a sua melhor regra de associação, o modelo final empata com a velocidade constante na validação (0,541 contra 0,544). O relatório registra isso junto com o resultado a favor (0,576 com a regra comum).

---

## Episódio 11: Parte 4 (horizonte de memória, galeria, correção)

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Como a IA auxiliou:**
  - Implementou o gradiente ∂L_t/∂h_{t−k} por autograd (com observações e às cegas), o experimento de oclusões injetadas (remove as detecções de uma identidade por N quadros e vê se o id volta), a escolha por regras fixas de três falhas e as figuras.
  - A primeira escolha automática das falhas pegou dois buracos de mais de 100 quadros, em que a track morre pela regra de `max_age` e não por falta de memória; as regras foram refeitas para buracos dentro de `max_age`. Também limpou rótulos fora do recorte nas figuras.
- **Validação / Decisões Humanas:**
  - A hipótese da correção (amortecer a extrapolação às cegas) foi escrita antes de rodar. Ela **não funcionou**; a IA mediu por quê (a rede já encolhe o movimento: inclinação 0,5–0,8) e o relatório registra que o diagnóstico partiu de um caso extremo. Os fatores > 1 foram testados depois, como exploratório, e estão marcados como pós-hoc.
  - Revisamos o relatório: corrigimos uma porcentagem (80% → 90%), um mecanismo que não foi verificado (a caixa "deslocada" na falha 1) e um trecho truncado.

---

## Episódio 12: Revisão cruzada da Parte 4 e do repositório

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Antes de seguir, pedimos uma revisão independente da Parte 4 (feita por outra pessoa da dupla) e da coesão do repositório.
- **Como a IA auxiliou:**
  - Conferiu o cálculo de ∂L/∂h contra diferenças finitas em dupla precisão (código independente), reproduziu a etapa A (CSV idêntico) e recomputou os números do relatório a partir dos CSVs. A etapa B não foi reproduzida do zero (> 1 h) e isso ficou registrado.
  - Achou: uma refutação por N50 dentro do ruído (o mesmo modelo dá 20,9 e 22,8 em duas tabelas; IC 95% ≈ ±2,5), um resultado exploratório descrito como nulo quando era misto, uma frase errada sobre a seed 42 e documentação defasada no README.
- **Validação / Decisões Humanas:**
  - O bootstrap do N50 foi incorporado ao pipeline (`bootstrap_n50`, com teste), para o relatório não citar um número que o repositório não produz. As conclusões não mudaram (a correção continua não funcionando e a RNN continua melhor que a caixa parada), mas o critério que as sustenta mudou.

---

## Episódio 13: Parte 5 (teste de estresse: queda de taxa de quadros)

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Fazer a Parte 5 sem retreinar e responder as duas perguntas do enunciado (por que quebra; alimentar Δt resolveria) com dados, não só com texto. As hipóteses foram escritas no plano antes de rodar.
- **Como a IA auxiliou:**
  - Propôs avaliar cada taxa em todas as fases de subamostragem, uma variante com `max_age` casado no tempo e diagnósticos em trajetórias do GT (IoU entre amostras consecutivas, inclinação do deslocamento previsto, sensibilidade ao recurso Δt). Escreveu `pa2/stress/`, `pa2/part5.py` e os testes.
  - Quando o resultado médio da validação escondia a estrutura, acrescentou o corte por tipo de câmera como saída do pipeline (e não como conta avulsa): a perda está toda nos vídeos de câmera móvel.
  - Conferiu o relatório contra os CSVs e achou um número errado na própria frase da H2 (−0,03 nos 7 vídeos; o certo é −0,007).
- **Validação / Decisões Humanas:**
  - O pipeline reproduz exatamente os números da Parte 2 em k = 1.
  - Resultados que contrariaram as hipóteses (alimentar Δt piora; o ganho do retreino multi-Δt só aparece nos vídeos de treino) ficaram no relatório, e o retreino ficou rotulado como exploratório, fora da regra "sem retreinar".

---

## Episódio 14: Notebook de inferência (`inferencia.ipynb`)

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Entregável obrigatório: receber uma sequência qualquer e devolver o vídeo com identidades coloridas e a contagem de objetos únicos, sem retreinar. Só temos as anotações do MOT17 (sem imagens).
- **Como a IA auxiliou:**
  - Colocou a lógica num módulo testável (`pa2/inference.py`) e deixou o notebook como casca; escreveu 16 testes com sequências sintéticas (sem GT, com imagens, erros) e um teste de regressão que reproduz o IDF1 da Parte 2 no vídeo 09.
  - Antes de usar detecções brutas (sem GT), mediu o efeito do filtro por distratores que as Partes 1–5 aplicavam antes de rastrear: IDF1 médio 0,595 com e 0,596 sem. O número ficou registrado em `RELATORY_PART1.md`.
  - Executou o notebook e conferiu as imagens embutidas (quadros com cor por id; gráfico de vida dos ids).
- **Validação / Decisões Humanas:**
  - O notebook avisa que a contagem de objetos únicos superestima (ids fragmentados) e mostra também a contagem filtrada e o GT, quando existe. O caminho com o torchvision só foi testado com modelo falso: está declarado como nunca rodado em imagens reais.

---

## Episódio 15: Revisão do repositório inteiro e correções

- **Data:** 01/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Antes da entrega, uma revisão do repositório como um todo: conformidade com o enunciado, regras proibidas, se os comandos documentados funcionam, consistência dos documentos e qualidade do código.
- **Como a IA auxiliou:**
  - Executou **cada comando do README** em diretórios temporários e comparou com os arquivos versionados (Partes 0, 1, 2 `--eval-only` e 5 idênticas; a Parte 5 inclusive com o retreino multi-Δt). Isso achou um comando enganoso (`pa2 0 --eval-only --checkpoint ...`, que ignora o checkpoint) e uma seção de download com nome de arquivo e estrutura de pastas errados.
  - Achou um risco de portabilidade: o `torch` fixado no índice CUDA sem marcador de plataforma (macOS não tem wheels nesse índice) e uma chave `tool.uv.package` fora do escopo. Ao corrigir, percebeu que o lock passaria o macOS para `torch 2.14.1` (não testado) e limitou o intervalo a `<2.8`.
  - Removeu código morto herdado do PA1, unificou `Context`/`build_context`, tornou públicos os helpers importados entre módulos e zerou o lint. Como os testes não exercitam os pipelines `run_parte*`, reexecutou as Partes 0, 1, 2, 4 (etapas A e C) e 5 depois da faxina e confirmou que reproduzem os resultados versionados.
- **Validação / Decisões Humanas:**
  - Não foi reproduzido: a instalação em macOS, os 21 treinos completos da Parte 3 e as etapas B e D da Parte 4. A lacuna de conformidade da Parte 1 (detector torchvision nunca rodado em imagens reais) estava em aberto nesta data e foi fechada no Episódio 16.

---

## Episódio 16: Revisão final, galeria, material da apresentação e detector torchvision

- **Data:** 02/10/2026
- **Ferramenta:** Claude Code
- **Contexto e Motivação:**
  - Véspera/dia da apresentação: auditar o repositório contra o enunciado, corrigir o que estivesse errado, montar o material visual e fechar a única lacuna de conformidade (o detector torchvision da Parte 1, que nunca tinha rodado em imagens reais).
- **Como a IA auxiliou:**
  - Três revisões independentes (Partes 0-1, Partes 2-3, Partes 4-5 e estrutura) contra o `PA2.md`; os números citados nos relatórios foram conferidos contra os CSVs. Achados: o detector torchvision pendente, o RPN do torchvision ainda usando `batched_nms` (documentado), erros no `AI_LOG.md` (um nome que não era da dupla e um IDF1 do caso (c) desatualizado: 0,8571 → 156/177 ≈ 0,881), o campo `settings` do `parte2_summary.json` descrevendo a receita errada, e a galeria da Parte 4 com o GT em cinza (o enunciado pede GT e predição coloridos por identidade).
  - Implementou as correções (galeria com GT colorido e diagnóstico na legenda, subcomando `uv run pa2 train-final`, `settings` do JSON, poda de 15 checkpoints e do zip redundante), montou `apresentacao/` (figuras copiadas de `outputs/` e geradas por `apresentacao/montar_apresentacao.py`) e escreveu as anotações de estudo para a apresentação.
  - Depois do download do `MOT17.zip`, extraiu só os quadros SDP de treino, mediu a velocidade do detector em 40 quadros (0,1 s/quadro) antes de rodar tudo, e rodou o Faster R-CNN nos 7 vídeos. As hipóteses foram escritas em `RELATORY_PART1.md` §7 **antes** de rodar; o resultado refutou parte delas (AP50 do torchvision é maior que o do SDP em 4 de 7 vídeos, apesar de o mAP e o IDF1 serem menores em 7 de 7).
  - Descobriu que o estágio do torchvision exigia as imagens mesmo com o cache pronto e o corrigiu para ler o cache sem imagens; ao reexecutar do cache, o CSV mudou na 4ª casa decimal (o cache guarda as caixas em texto), então os números documentados são os do cache.
- **Validação / Decisões Humanas:**
  - A dupla decidiu rodar só o mínimo (o detector torchvision e o vídeo do notebook com fundo real) e não refazer as figuras de falha da Parte 4 sobre quadros reais. As Partes 0, 2, 3, 4 e 5 não leem imagem, então não foram reexecutadas.
  - Segue não reproduzido: instalação em macOS, os 21 treinos completos da Parte 3 e as etapas B e D da Parte 4.

