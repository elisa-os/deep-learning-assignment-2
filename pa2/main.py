"""Ponto de entrada principal do PA2.

Uso:
    uv run pa2                     # roda a parte configurada (default: parte 0)
    uv run pa2 0                   # Parte 0 — testes sintéticos
    uv run pa2 1                   # Parte 1 — baseline por quadro
    uv run pa2 2                   # Parte 2 — RNN memória temporal
    uv run pa2 3                   # Parte 3 — ablações
    uv run pa2 0 --eval-only       # avalia sem retreinar (quando checkpoint existe)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pa2.config import load_config
from pa2.utils import set_seed, get_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PA2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento",
    )
    parser.add_argument(
        "parte",
        nargs="?",
        default="0",
        help="Número da parte a executar (0-5, default: 0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pa2/config.yaml",
        help="Caminho para o arquivo de configuração (default: pa2/config.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescreve epochs da configuração",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Sobrescreve learning rate da configuração",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Avalia sem retreinar (requer checkpoint existente)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Caminho para checkpoint para avaliação",
    )
    args = parser.parse_args()

    parte = args.parte
    cfg = load_config(path=args.config, parte=parte)

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.eval_only:
        cfg.train.eval_only = True
    if args.checkpoint is not None:
        cfg.train.checkpoint = args.checkpoint

    set_seed(cfg.seed)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"PA2 — Parte {cfg.parte} ({cfg.mode_tag})")
    print(f"{'='*60}")
    print(f"  seed: {cfg.seed}")
    print(f"  device: {device}")
    print(f"  output_dir: {cfg.output_dir}")
    print(f"  epochs: {cfg.train.epochs}")
    print(f"  lr: {cfg.train.lr}")
    print(f"  eval_only: {cfg.train.eval_only}")
    print(f"  checkpoint: {cfg.train.checkpoint}")
    print(f"  data.synthetic: {cfg.data.synthetic}")
    if not cfg.data.synthetic:
        print(f"  data.data_dir: {cfg.data.data_dir}")
    print(f"{'='*60}\n")

    parte_int = cfg.parte or 0

    if parte_int == 0:
        from pa2.part0 import run_parte0
        run_parte0(cfg, device)
    elif parte_int == 1:
        from pa2.part1 import run_parte1
        run_parte1(cfg, device)
    elif parte_int == 2:
        from pa2.part2 import run_parte2
        run_parte2(cfg, device)
    elif parte_int == 3:
        from pa2.ablation import run_ablation
        run_ablation(cfg, device)
    elif parte_int == 4:
        from pa2.part4 import run_parte4
        run_parte4(cfg, device)
    elif parte_int == 5:
        from pa2.part5 import run_parte5
        run_parte5(cfg, device)
    else:
        print(f"Parte {parte_int} não implementada ainda.")
        print("Partes disponíveis: 0 (sintético), 1 (baseline), 2 (RNN), 3 (ablação), 4 (galeria), 5 (estresse)")
        sys.exit(1)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
