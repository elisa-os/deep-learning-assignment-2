# CLAUDE.md

- **Fonte do enunciado:** `PA2.md` (transcrição fiel de `PA2.pdf`). Leia-o em vez do PDF; em caso de dúvida, o PDF manda.
- **Estado e decisões:** `PLANO_DE_EXECUCAO.md` (plano, decisões, backlog). **Resultados e limitações de cada parte:**
  `RELATORY_PART0.md` a `RELATORY_PART4.md`. Estrutura do repositório e comandos: `README.md`.
- Modelo final (Partes 4 e 5, notebook): `outputs/checkpoints/final_motion_rnn.pt` (decisão em `RELATORY_PART2.md` §6).
  Detecções congeladas: SDP, score ≥ limiar da Parte 1. Split por vídeo: treino 02/04/05/10/11, validação 09/13.
- Regras proibidas (ver `PA2.md` §3): trackers prontos, `motmetrics`/`TrackEval`, `torchvision.ops.nms`. IDF1, ID switches e NMS são implementação própria.
- Ambiente via `uv` (`uv sync`); testes: `uv run pytest`. `outputs/` e `data/` são versionados.
- Convenção do projeto: hipóteses escritas antes de rodar; resultados que as contrariam ficam no relatório; afirmações sobre
  números só com o número conferido contra o CSV.
