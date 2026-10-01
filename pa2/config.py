"""Carregamento e validação da configuração via YAML para o PA2.

Usa dataclasses para acesso por atributo em vez de dicionário.
Suporte a configuração por parte (parte0, parte1, ...):
  - Na seção "parteN" do YAML, cada parte tem seus próprios valores.
  - load_config(path, parte=N) carrega a seção parte{N} do YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class SyntheticConfig:
    """Parâmetros do gerador sintético (Parte 0)."""
    n_sequences: int = 20
    n_frames: int = 45
    frame_size: int = 128
    n_objects_min: int = 5
    n_objects_max: int = 12
    occlusion_duration_min: int = 5
    occlusion_duration_max: int = 15
    velocity_scale: float = 1.0
    noise_level: float = 0.0
    contrast_scale: float = 1.0
    detector_drop_rate: float = 0.0
    detector_noise: float = 0.0
    detector_fp_rate: float = 0.0


@dataclass
class DataConfig:
    """Configuração de dados (MOT17 ou sintético)."""
    synthetic: bool = True
    data_dir: str | None = None
    n_sequences: int = 500
    batch_size: int = 1
    num_workers: int = 0
    sequence_split: dict[str, list[str]] | None = None  # {train: [...], val: [...], test: [...]}


@dataclass
class ModelConfig:
    """Configuração do modelo (RNN de movimento, Trilha A)."""
    rnn_type: str = "LSTM"                # LSTM | GRU
    input_size: int = 5                  # [cx, cy, w, h, conf]
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    use_delta_t: bool = True
    predict_uncertainty: bool = False


@dataclass
class AssociationConfig:
    """Configuração de associação e gestão de tracks."""
    method: str = "greedy"               # greedy | hungarian
    iou_threshold: float = 0.3
    max_age: int = 30                    # quadros sem observação antes de matar track
    min_hits: int = 3                    # hits mínimos para track válida


@dataclass
class RNNConfig:
    """Configuração específica da RNN (Parte 2+)."""
    window_T: int = 32                   # janela de BPTT truncado
    teacher_forcing_ratio: float = 1.0  # 1.0 = teacher forcing puro


@dataclass
class TrainConfig:
    """Configuração de treinamento."""
    epochs: int = 30
    lr: float = 1.0e-3
    checkpoint: str | None = None
    eval_only: bool = False
    gradient_clipping: bool = True
    grad_clip_value: float = 1.0


@dataclass
class AblationConfig:
    """Configuração de ablação (Parte 3)."""
    axis: str = "eixo2"                  # eixo2 = regime de treino
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    regimes: list[dict] = field(default_factory=list)


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "outputs"
    parte: Optional[int] = None
    mode_tag: str = "parte0"
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    # Detector público do MOT17 (Parte 1+)
    detector_source: str = "FRCNN"       # DPM | FRCNN | SDP
    use_torchvision_detector: bool = False


def load_config(
    path: str | Path = "pa2/config.yaml",
    parte: str | int | None = None,
) -> Config:
    """Lê um arquivo YAML e retorna um Config tipado.

    Se `parte` for informado, carrega a seção correspondente do YAML.
    Campos ausentes no YAML usam os defaults do dataclass.
    """
    # Normaliza parte para int ou None
    parte_int: Optional[int] = None
    if parte is not None:
        if isinstance(parte, int):
            parte_int = parte
        elif isinstance(parte, str):
            s = parte.strip()
            if s.startswith("parte"):
                num_part = s[len("parte"):]
                digits = ""
                for ch in num_part:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    parte_int = int(digits)
            elif s.isdigit():
                parte_int = int(s)

    path = Path(path)
    if not path.exists():
        if Path("config.yaml").exists():
            path = Path("config.yaml")
        elif Path("pa2/config.yaml").exists():
            path = Path("pa2/config.yaml")
        else:
            raise FileNotFoundError(f"Config não encontrada: {path}")

    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    seed = raw.get("seed", 42)
    # Caminhos relativos do YAML valem a partir do diretório de execução (raiz do repo).
    out_dir_raw = raw.get("output_dir", "outputs")
    out_dir_path = Path(out_dir_raw)

    if parte_int is not None:
        parte_key = f"parte{parte_int}"
        parte_section = raw.get(parte_key, {})

        # Campos flat da seção da parte
        syn_fields = {"n_sequences", "n_frames", "frame_size", "n_objects_min", "n_objects_max",
                      "occlusion_duration_min", "occlusion_duration_max", "velocity_scale",
                      "noise_level", "contrast_scale", "detector_drop_rate",
                      "detector_noise", "detector_fp_rate"}
        data_fields = {"synthetic", "data_dir", "batch_size", "num_workers", "sequence_split"}
        model_fields = {"rnn_type", "input_size", "hidden_size", "num_layers", "dropout",
                        "use_delta_t", "predict_uncertainty"}
        assoc_fields = {"method", "iou_threshold", "max_age", "min_hits"}
        rnn_fields = {"window_T", "teacher_forcing_ratio"}
        train_fields = {"epochs", "lr", "checkpoint", "eval_only", "gradient_clipping", "grad_clip_value"}

        syn_raw = {k: v for k, v in parte_section.items() if k in syn_fields}
        data_raw = {k: v for k, v in parte_section.items() if k in data_fields}
        model_raw = {k: v for k, v in parte_section.items() if k in model_fields}
        assoc_raw = {k: v for k, v in parte_section.items() if k in assoc_fields}
        rnn_raw = {k: v for k, v in parte_section.items() if k in rnn_fields}
        train_raw = {k: v for k, v in parte_section.items() if k in train_fields}

        # Ablation é mais complexo — lê como dict e converte depois
        ablation_raw = parte_section.get("ablation", {})

        seed = parte_section.get("seed", seed)
        out_dir_raw = parte_section.get("output_dir", out_dir_raw)
        if out_dir_raw:
            out_dir_path = Path(out_dir_raw)
    else:
        syn_raw = {}
        data_raw = {}
        model_raw = {}
        assoc_raw = {}
        rnn_raw = {}
        train_raw = {}
        ablation_raw = {}

        if "output_dir" in raw:
            out_dir_raw = raw["output_dir"]
            out_dir_path = Path(out_dir_raw)

    # Detector source é campo de nível superior
    detector_source = raw.get("detector_source", "FRCNN")
    if parte_int is not None:
        detector_source = parte_section.get("detector_source", detector_source)
    use_torchvision_detector = raw.get("use_torchvision_detector", False)
    if parte_int is not None:
        use_torchvision_detector = parte_section.get("use_torchvision_detector", use_torchvision_detector)

    out_dir_str = str(out_dir_path)

    return Config(
        seed=seed,
        output_dir=out_dir_str,
        parte=parte_int,
        mode_tag=f"parte{parte_int}" if parte_int is not None else "custom",
        synthetic=SyntheticConfig(**{**syn_raw}),
        data=DataConfig(**{**data_raw}),
        model=ModelConfig(**{**model_raw}),
        association=AssociationConfig(**{**assoc_raw}),
        rnn=RNNConfig(**{**rnn_raw}),
        train=TrainConfig(**{**train_raw}),
        ablation=AblationConfig(
            axis=ablation_raw.get("axis", "eixo2"),
            seeds=ablation_raw.get("seeds", [42, 123, 456]),
            regimes=ablation_raw.get("regimes", []),
        ),
        detector_source=detector_source,
        use_torchvision_detector=use_torchvision_detector,
    )
