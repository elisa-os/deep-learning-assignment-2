# Material visual da apresentação

Só as figuras essenciais, na ordem da fala. São cópias de `outputs/` ou geradas por `uv run python apresentacao/montar_apresentacao.py`
(que também recopia tudo). Nenhum número aqui é novo: vêm dos CSVs de `outputs/` e dos `RELATORY_PART*.md`.

| Pasta | Arquivos | O que mostra |
|---|---|---|
| `visao_geral/` | `00a` pipeline, `00b` arquitetura da RNN, `00c` split por vídeo | o desenho do trabalho |
| `parte0/` | `0a` oclusão por profundidade, `0b` simulador de detector, `0c` casos das métricas, `0e` onde o baseline quebra | testes sintéticos |
| `parte1/` | `1a` descolamento mAP × IDF1 (SDP), `1c` SDP × torchvision | baseline por quadro, duas fontes de detecção |
| `parte2/` | `2a` comparação, `2b` IoU às cegas, `2c` reconexão após buracos | RNN de movimento (modelo final) |
| `parte3/` | `3a` IDF1 por regime, `3b` rollout às cegas, `3c` norma do gradiente | ablação do regime de treino |
| `parte4/` | `4a` gradiente, `4b` sobrevivência, `4c`/`4d`/`4e` três falhas, `4f` correção | memória e falhas |
| `parte5/` | `5a` curva de IDF1, `5b` por câmera, `5c` diagnóstico, `5e` multi-Δt | queda de taxa de quadros |
| `inferencia/` | `6a` GIF, `6c` quadros reais | notebook de inferência (MOT17-09) |

As figuras de falha da Parte 4 são caixas sobre fundo vazio; o vídeo de inferência usa os quadros reais do MOT17-09 (as imagens não são versionadas).
O `torchvision` é o Faster R-CNN COCO sem fine-tune e só entra na comparação da Parte 1; o resto usa o SDP.
