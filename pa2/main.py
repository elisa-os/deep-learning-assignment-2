"""Ponto de entrada principal do PA2.

Uso:
    uv run pa2 0                   # Parte 0 — testes sintéticos
    uv run pa2 1                   # Parte 1 — baseline por quadro
    uv run pa2 2                   # Parte 2 — RNN como modelo de movimento (Trilha A)
    uv run pa2 3                   # Parte 3 — ablação (Eixo 2: regime de treino)
    uv run pa2 4                   # Parte 4 — horizonte de memória, galeria de falhas, correção
    uv run pa2 5                   # Parte 5 — teste de estresse (queda de taxa de quadros)
    uv run pa2 2 --eval-only --checkpoint <ckpt> --output-dir <dir>   # reavalia sem retreinar
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
        help="Número da parte a executar (0-5, default: 0) ou `train-final` (treina o modelo final)",
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
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Sobrescreve output_dir (p.ex. para não sobrescrever resultados de outro modelo)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="(parte 3) roda só estas seeds; vários processos podem dividir as seeds")
    parser.add_argument("--regimes", type=str, nargs="+", default=None,
                        help="(parte 3) roda só estes regimes (nomes do config.yaml)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="(parte 3) só agrega os runs que já existem")
    parser.add_argument("--install", action="store_true",
                        help="(train-final) copia o checkpoint treinado para outputs/checkpoints/final_motion_rnn.pt")
    args = parser.parse_args()

    parte = args.parte
    train_final = parte == "train-final"
    if train_final:
        # modelo final = regime `teacher_forcing`, seed 42 da Parte 3 (RELATORY_PART2.md §6)
        parte, args.seeds, args.regimes = "3", [42], ["teacher_forcing"]
        args.output_dir = args.output_dir or "outputs/retreino"
    cfg = load_config(path=args.config, parte=parte)

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
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
        run_ablation(cfg, device, only_seeds=args.seeds, only_regimes=args.regimes,
                     aggregate_only=args.aggregate_only)
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

    if train_final:
        import shutil
        from pathlib import Path
        trained = Path(cfg.output_dir) / "parte3_ablation" / "checkpoints" / "teacher_forcing_s42.pt"
        target = Path("outputs/checkpoints/final_motion_rnn.pt")
        if args.install:
            shutil.copy(trained, target)
            print(f"\nCheckpoint final instalado em {target}")
        else:
            print(f"\nCheckpoint treinado: {trained} (use --install para copiá-lo para {target})")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
