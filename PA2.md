# Programming Assignment 2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento

> Transcrição fiel de `PA2.pdf` para consumo por agentes. Em caso de dúvida, o PDF manda.

| | |
|---|---|
| Disciplina | Aprendizado Profundo |
| Professor | Dario Oliveira |
| Monitor | Erick Brito |
| Entrega | 02/10, 23h59 — B41254@fgv.edu.br |
| Apresentação | 02/10 — horários a definir |
| Formato | Duplas |

---

## 1. O problema

A aula de detecção começou pelo mapa das tarefas de visão computacional — classificação, localização, detecção, segmentação — e todas são definidas sobre uma imagem. YOLO e Mask R-CNN entregam, para cada quadro, um conjunto de objetos com caixa, classe e (Mask R-CNN) máscara. Mas o conjunto do quadro t não tem relação nenhuma com o do quadro t+1: a ordem das saídas vem do NMS, que ordena por score. O objeto que era o terceiro da lista vira o primeiro quando alguém passa na frente dele.

A aula de RNN deu a máquina para sequências: estado que persiste, BPTT, o problema do gradiente que some, as variantes com portas que existem justamente por causa disso. Só que os exemplos eram sequências de vetores, com comprimento e ordem bem definidos. Um vídeo é uma sequência de **conjuntos** de tamanho variável, sem ordem canônica, e o que precisa persistir não é um estado só: é **um estado por objeto**, com objetos nascendo, sumindo atrás de um poste e voltando trinta quadros depois.

Este PA é a junção. A tarefa é fazer as arquiteturas das duas aulas produzirem rótulos **identity-aware**: se um objeto aparece no quadro 3 e reaparece no quadro 40, ele tem que sair com o mesmo identificador. **Sem usar rastreador pronto** (ByteTrack, DeepSORT, OC-SORT, `model.track` do ultralytics e afins estão proibidos).

É assim que boa parte da literatura de MOT funciona, mas exige projetar três coisas que a aula não entregou prontas:

1. o que a recorrência carrega (o que é o estado de um objeto: caixa, velocidade, aparência);
2. qual perda otimiza isso (de onde vem o gradiente?);
3. como decodificar a previsão em trajetórias (atribuição, nascimento e morte de tracks).

No PA1 passou-se de class-aware para instance-aware no espaço; aqui, de instance-aware por quadro para identity-aware no tempo. Estão igualmente acopladas.

---

## 2. Dados

**MOT17 / MOTChallenge.** Sequências de pedestres em rua e ambiente interno, com caixas e identidades anotadas quadro a quadro, câmeras paradas e em movimento, densidades muito diferentes.

- Download direto, sem conta: https://motchallenge.net/data/MOT17/
- Pacote completo ~5,5 GB. Existe também o pacote só de anotações (~10 MB): **comecem por ele**. Dá para escrever e depurar a métrica e a associação inteiras antes de baixar um único quadro.
- O benchmark fornece detecções públicas (DPM, Faster R-CNN, SDP) por sequência. Isso permite o PA caber em duas semanas: detecção já vem resolvida; o trabalho é o que vem depois.
- Formato do `gt.txt`: `frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility`. O campo `visibility` é ouro para a Parte 4.

**Split.** Por **sequência**, nunca por quadro. Separar quadros aleatoriamente coloca o quadro t no treino e o t+1 na validação, e o modelo temporal é avaliado em cima de algo que praticamente já viu. Reservar **pelo menos uma sequência inteira** nunca vista, e **justificar o critério na apresentação** (câmera parada vs. móvel, densidade, ponto de vista).

---

## 3. Regras de engenharia

**Permitido:** PyTorch; `torchvision.models.detection` (Faster R-CNN, Mask R-CNN) e encoders pré-treinados; albumentations; scipy; sklearn; e o código de matching e de métricas do PA1.

**Permitido com condição:** filtro de Kalman de velocidade constante, **apenas como baseline de comparação**. É um baseline honesto e frequentemente difícil de bater; não pode ser o modelo temporal da Parte 2. Lá, quem carrega o estado é a rede recorrente.

**Proibido:**

- rastreadores prontos: SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT, Norfair, motpy, trackers do `supervision`, `model.track()` do ultralytics, ou qualquer implementação pronta de associação;
- métricas de rastreamento prontas de biblioteca — `motmetrics`, `TrackEval`, `py-motmetrics`. **Implementar IDF1 e contagem de ID switches**;
- `torchvision.ops.nms` — **implementar o NMS**;
- clonar solução pronta de MOT17. Se usarem ideia de repo ou paper, **citem e reescrevam**.

O modelo temporal, as perdas, a associação e a gestão de tracks são de autoria da dupla.

---

## Parte 0 — Testes sintéticos

Antes do MOT17, construir um ambiente controlado onde se sabe a resposta certa. Três dos quatro artefatos abaixo são reaproveitados nas partes seguintes.

1. **O gerador.** Vídeos 128×128 de 30 a 60 quadros, com 5 a 15 elipses em movimento, tamanhos variados, ruído e contraste variáveis. Expor como parâmetros pelo menos: **número de objetos, velocidade típica e duração da oclusão**.
   - *Requisito verificável:* elipses desenhadas com **ordem de profundidade**, de modo que uma passe atrás da outra e realmente desapareça (senão não é oclusão). **Mostrar uma figura com uma trajetória que some por N quadros e volta.**
2. **O simulador de detector.** Função que recebe as caixas verdadeiras e as estraga de propósito: descarta p% delas, adiciona ruído nas coordenadas, injeta falsos positivos. Permite validar a Parte 1 inteira no sintético antes de baixar o MOT17 (é exatamente um experimento da Parte 5).
3. **A métrica e seus testes.** Implementar IDF1 e contagem de ID switches e mostrar que passam em três casos construídos à mão:
   - (a) predição = ground truth ⇒ IDF1 = 1 e zero switches;
   - (b) duas identidades trocadas a partir do quadro k ⇒ o número exato de switches esperado;
   - (c) uma track partida em duas no meio ⇒ o efeito esperado em IDF1, que **não é o mesmo** de (b).
4. **O baseline no piso fácil.** Rodar a associação ingênua da Parte 1 com poucas elipses, lentas e sem oclusão. IDF1 tem que ficar muito perto de 1.

Depois, girar os botões do gerador (mais objetos, mais rápidos, oclusão mais longa) e **mostrar onde o baseline começa a quebrar**. Esse gráfico é o ensaio da Parte 1.

---

## Parte 1 — Baseline por quadro

1. **Detecções, sem treinar detector.** Duas fontes, usar as duas:
   - as detecções públicas do MOT17 (`det/det.txt`). Três detectores: DPM, FRCNN, SDP; **escolher um como fonte padrão do resto do PA. Dizer qual, e por quê**;
   - um detector pré-treinado do torchvision em modo de inferência, classe `person` do COCO.

   Fine-tune do detector é opcional e não vale ponto extra por si só — só vale se mostrarem o que isso muda no rastreamento.

2. **Associação ingênua:** IoU entre detecções do quadro t e t−1, matching guloso ou Hungarian, limiar fixo, ID novo quando nada casa, track morta depois de k quadros sem observação.

3. **Avaliar como trajetórias, com métricas próprias:**
   - **IDF1**: exige atribuição global um-para-um entre identidades previstas e verdadeiras ao longo da sequência inteira;
   - **ID switches e fragmentações**, contados explicitamente;
   - **erro de contagem de identidades únicas** no vídeo (análogo temporal do erro de contagem do PA1).

   MOTA é opcional: com detector congelado, os termos FP/FN quase não variam entre configurações e ela esconde a única coisa que muda.

4. **Documentar** a regra de associação e a gestão de nascimento/morte de tracks. Como no PA1, regras diferentes dão números diferentes.

5. **Quantificar o fracasso.** Gráfico obrigatório: o **descolamento**, em dois painéis sobre as mesmas sequências:
   - em cima: **mAP por quadro** e **IDF1**;
   - embaixo: **nº de identidades previstas / nº de verdadeiras** e **ID switches por identidade verdadeira**.

   Ordenar as sequências por um eixo de dificuldade à escolha: densidade, movimento de câmera, duração de oclusão.

---

## Parte 2 — Memória temporal (escolher UMA trilha)

A fonte de detecções fica **congelada** a partir daqui; o que muda é o que acontece entre os quadros.

### Trilha A — RNN como modelo de movimento
Um estado recorrente por track. A cada quadro, o LSTM/GRU recebe a última observação (caixa, opcionalmente confiança e Δt) e prevê a caixa do quadro seguinte; a associação usa IoU entre caixa prevista e caixa observada. Sob oclusão, o estado roda para frente sem observação, e a track sobrevive ou não. Perda L1/smooth-L1 sobre a caixa, treinada em trajetórias do ground truth; opcionalmente prever também uma incerteza e usar log-verossimilhança gaussiana, o que dá um **portão de associação adaptativo**.

### Trilha B — RNN como memória de aparência
Cada detecção vira um embedding D-dimensional (recorte da imagem por um encoder pequeno e pré-treinado). Um agregador recorrente mantém o estado de aparência da track, atualizado a cada observação; a associação é similaridade de cosseno, com um portão geométrico de IoU por cima. Perda contrastiva ou triplet sobre as identidades do ground truth.

A apresentação precisa explicar **em que aspecto** essa representação (A ou B) melhora o fracasso da Parte 1. Mostrar a comparação com as mesmas métricas **lado a lado com a baseline, nas mesmas sequências**.

**Pergunta para a apresentação (sem implementar):** um vídeo de duas horas não cabe na memória, então a inferência é feita em janelas de T quadros. No PA1, a inferência em mosaico quebrava objetos na fronteira entre tiles. **O que quebra aqui, na fronteira entre janelas, e o que a sua representação em particular permitiria fazer para costurar as identidades?**

---

## Parte 3 — Ablação

Escolher **um** dos eixos abaixo e rodar a configuração com **3 seeds**, reportando **média ± desvio**:

- **Eixo 1 — a célula recorrente.** RNN simples vs. LSTM vs. GRU, no mesmo orçamento aproximado de parâmetros, variando também o comprimento da janela de BPTT truncado (T ∈ {4, 8, 16, 32}). Pergunta: onde a RNN simples quebra, e isso bate com a história do gradiente que some da aula?
- **Eixo 2 — o regime de treino.** Teacher forcing → scheduled sampling → free-running. Na inferência o modelo se alimenta das próprias previsões (e, sob oclusão, só delas); se nunca viu isso no treino, a distribuição muda debaixo dele. Medir. Incluir **gradient clipping ligado/desligado** e reportar o que acontece sem ele.
- **Eixo 3 — o que entra na recorrência.** Só geometria, só aparência, ou os dois. Qual sustenta a identidade através de uma oclusão longa, e isso muda com a densidade da cena?
- **Eixo 4 — direção do contexto.** Causal (online, só o passado) vs. bidirecional (offline, com o futuro). Quantos pontos de IDF1 o futuro vale, e a que custo de latência? **Reportar ganho e atraso juntos.**

---

## Parte 4 — Galeria de falhas e horizonte de memória

**Três trechos** em que o modelo final erra feio, cada um com:

- a **figura**: tira de quadros com ground truth e predição coloridos por identidade, mais o mapa intermediário relevante (caixa prevista pela recorrência ou matriz de similaridade de embeddings);
- um **diagnóstico**, ex.: "esse objeto fica ocluído por 34 quadros; minha janela de BPTT é de 10 e a norma do gradiente cai 20× em 8 passos, então o modelo nunca recebeu sinal de supervisão que atravessasse esse buraco".

**Obrigatório:** medir o **horizonte de memória efetivo** do modelo, das duas formas:

1. **Analítica.** Norma de ∂L_t/∂h_{t−k} em função de k. É a curva de gradiente que some dos slides de RNN, no seu modelo e nos seus dados. Se fez o Eixo 1 da Parte 3, comparar a curva da RNN simples com a do modelo com portas, na mesma janela — os checkpoints já existem.
2. **Empírica.** Quantos quadros o estado sobrevive a uma oclusão antes de a track morrer ou trocar de ID. Comparar essa distribuição com a distribuição de duração de oclusão do dataset.

**E fazer uma correção:** escolher um dos diagnósticos, implementar a mudança que ele sugere, mostrar antes/depois. Se não funcionar, explicar o que isso revela sobre o diagnóstico estar errado.

---

## Parte 5 — Teste de estresse (escolher UM)

Ambos **sem retreinar**, em cima do modelo final.

- **Queda de taxa de quadros:** avaliar com o vídeo subamostrado a **1/2 e 1/5** da taxa original — curva de degradação do IDF1. Por que um modelo de movimento aprendido em Δt fixo quebra quando Δt muda? Alimentar Δt na recorrência resolveria?
- **Qualidade do detector:** degradar as detecções de propósito: descartar p%, adicionar ruído nas caixas, injetar falsos positivos, em **3 intensidades**. O modelo temporal absorve ou amplifica a falha do detector? Reportar **mAP e IDF1 juntos**.

---

## 4. Entregáveis

| Arquivo | Descrição |
|---|---|
| Repositório Git | Repositório do trabalho |
| `README.md` | Ambiente, download dos dados, **um comando que treina, um comando que avalia** |
| `metrics.py` | Implementação própria de IDF1, ID switches e fragmentações |
| `AI_LOG.md` | Ver seção 5 |
| `inferencia.ipynb` | Recebe o caminho de uma sequência qualquer, devolve o vídeo com as identidades coloridas de forma consistente e a contagem de objetos únicos. **Roda sem retreinar** |
| Checkpoint | Pesos do modelo temporal (link se for grande) |

**Não há relatório escrito. A avaliação é a apresentação.** O repositório dá lastro ao que for afirmado lá: toda tabela, curva, imagem e vídeo mostrado na apresentação tem que ser **reproduzível a partir dele**.

---

## 5. Política de uso de IA

Uso de IA é permitido e esperado. O que não é permitido é entregar algo que a dupla não entende.

Subir também um `AI_LOG.md` descrevendo como a IA foi usada no assignment. Pode citar alguns episódios e como recorreram às ferramentas de IA para resolvê-los.

---

## Checklist rápido (derivado, não faz parte do PDF)

- [ ] P0: gerador com oclusão por profundidade + figura de trajetória que some N quadros
- [ ] P0: simulador de detector (drop p%, ruído, FPs)
- [ ] P0: IDF1 + ID switches + 3 testes manuais (a, b, c)
- [ ] P0: baseline no piso fácil (IDF1 ≈ 1) + gráfico de quebra girando botões
- [ ] P1: det públicas (escolher DPM/FRCNN/SDP e justificar) + detector torchvision (`person`)
- [ ] P1: NMS próprio; associação ingênua; IDF1, IDsw, fragmentações, erro de contagem
- [ ] P1: gráfico de descolamento em 2 painéis
- [ ] P2: trilha A **ou** B + comparação lado a lado + resposta sobre janelas
- [ ] P3: um eixo, 3 seeds, média ± desvio
- [ ] P4: 3 falhas com figura + diagnóstico; horizonte analítico e empírico; 1 correção antes/depois
- [ ] P5: um teste de estresse sem retreinar
- [ ] Split por sequência (≥1 sequência inteira de validação, critério justificado)
- [ ] README (1 comando treina, 1 avalia), `metrics.py`, `AI_LOG.md`, `inferencia.ipynb`, checkpoint
- [ ] Proibidos: trackers prontos, motmetrics/TrackEval, `torchvision.ops.nms`
