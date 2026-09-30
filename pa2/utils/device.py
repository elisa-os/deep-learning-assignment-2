"""Detecção do dispositivo de computação.

CPU vs GPU em Deep Learning:
-----------------------------
GPUs (Graphics Processing Units) foram originalmente feitas para renderizar
pixels em paralelo. Pixels independentes → operações paralelas massivas.
Essa mesma estrutura é perfeita para multiplicações de matrizes em redes neurais.

Uma GPU moderna tem milhares de núcleos simples (CUDA cores), enquanto uma CPU
tem dezenas de núcleos complexos. Para as operações de álgebra linear de DL,
a GPU ganha fácil: treinar uma rede em CPU pode levar horas; na GPU, minutos.

CUDA é o framework da NVIDIA que permite usar a GPU para computação geral.
PyTorch usa CUDA automaticamente quando disponível.
"""

import warnings
import torch


def _is_cuda_device_compatible(device_index: int = 0) -> bool:
    """Verifica se a GPU no índice dado é compatível com a build atual do PyTorch.

    Compara a Compute Capability (CC) do dispositivo com as CCs para as quais
    o binário do PyTorch foi compilado. Retorna False se a GPU foi detectada
    mas não é suportada, evitando erros em tempo de execução.
    """
    if not torch.cuda.is_available():
        return False

    device_cc = torch.cuda.get_device_capability(device_index)  # (major, minor)
    device_cc_int = device_cc[0] * 10 + device_cc[1]  # ex.: (6, 1) → 61

    # Cada entrada é a CC de compilação e o intervalo de hardware que ela suporta.
    # Baseado no mapeamento interno do PyTorch (torch/cuda/__init__.py).
    # Formato: (compiled_cc, min_supported_cc, max_supported_cc_exclusive, excluded)
    _SUPPORTED_RANGES = [
        (75,  75,  80, set()),        # sm_75 → Turing
        (80,  80,  90, {87}),         # sm_80 → Ampere (exclui 8.7)
        (86,  86,  90, {87}),         # sm_86 → Ampere upper (exclui 8.7)
        (90,  90, 100, set()),        # sm_90 → Hopper
        (100, 100, 110, {101}),       # sm_100 → Blackwell
        (120, 120, 130, set()),       # sm_120 → future arch
    ]

    # Obtém a lista de CCs compiladas desta instalação específica
    try:
        arch_list = torch.cuda.get_arch_list()  # ex.: ['sm_75', 'sm_80', ...]
    except AttributeError:
        arch_list = []

    if arch_list:
        # Verifica diretamente contra as CCs compiladas presentes no binário.
        # PyTorch compila com PTX fallback, então sm_XX suporta hardware ≥ CC XX
        # dentro de uma geração — mas sem JIT-PTX não há garantia para gerações antigas.
        compiled_ccs = sorted(
            int(s.replace("sm_", "")) for s in arch_list if s.startswith("sm_")
        )
        # GPU é suportada se há alguma CC compilada ≤ device_cc_int
        # (compatibilidade binária só funciona de CC baixa para cima, não o contrário)
        if compiled_ccs and min(compiled_ccs) > device_cc_int:
            return False  # GPU mais antiga que todas as CCs compiladas
        return True

    # Fallback: usa os intervalos hardcoded
    for _, min_cc, max_cc, excluded in _SUPPORTED_RANGES:
        if min_cc <= device_cc_int < max_cc and device_cc_int not in excluded:
            return True
    return False


def get_device() -> torch.device:
    """Retorna o melhor dispositivo disponível e compatível com esta build do PyTorch.

    Ordem de preferência: CUDA (GPU NVIDIA compatível) → CPU.

    Se uma GPU NVIDIA for detectada mas a sua Compute Capability não for suportada
    pela versão do PyTorch instalada, emite um aviso claro e usa a CPU — evitando
    erros silenciosos ou crashes em tempo de execução.
    """
    if torch.cuda.is_available():
        if _is_cuda_device_compatible(device_index=0):
            device = torch.device("cuda")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_cc = torch.cuda.get_device_capability(0)

            try:
                arch_list = torch.cuda.get_arch_list()
            except AttributeError:
                arch_list = ["desconhecidas"]

            warnings.warn(
                f"\n[get_device] GPU detectada mas incompatível com esta build do PyTorch.\n"
                f"  GPU:              {gpu_name}\n"
                f"  Compute Capability: {gpu_cc[0]}.{gpu_cc[1]} (sm_{gpu_cc[0]}{gpu_cc[1]})\n"
                f"  CCs suportadas:   {', '.join(arch_list)}\n"
                f"  PyTorch versão:   {torch.__version__}\n"
                f"\n"
                f"  → Usando CPU como fallback.\n"
                f"\n"
                f"  Para usar a GPU, instale uma versão compatível do PyTorch:\n"
                f"    GTX 1080 Ti / Pascal (CC 6.1):\n"
                f"      uv pip install torch==2.6.0 "
                f"--index-url https://download.pytorch.org/whl/cu118\n",
                stacklevel=2,
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    gpu_info = ""
    if device.type == "cuda":
        gpu_info = f" | GPU: {torch.cuda.get_device_name(0)}"

    print(f"Device: {device}{gpu_info} | PyTorch: {torch.__version__}")
    return device
