"""Fixação de seeds para reprodutibilidade.

Por que isso importa em Deep Learning?
---------------------------------------
Redes neurais dependem de aleatoriedade em vários pontos:
- Inicialização dos pesos (ex: Kaiming, Xavier)
- Embaralhamento dos dados no DataLoader
- Dropout durante o treino
- Operações não-determinísticas na GPU (ex: atomicAdd em operações de redução)

Fixar a seed garante que dois treinamentos com os mesmos hiperparâmetros
produzam exatamente os mesmos resultados — essencial para comparar experimentos.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixa todas as seeds relevantes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
