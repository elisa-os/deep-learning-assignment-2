"""Modelos de movimento intercambiáveis para o rastreador (Parte 2).

Todos têm a mesma interface, em lote sobre as tracks e com caixas ``[l, t, w, h]``::

    states, pred = model.step(states, fed, prev, observed)

- ``fed``: caixa que entra no passo (a detecção casada; ou, sem observação, a previsão
  anterior da própria track);  ``prev``: ``fed`` do passo anterior (``None`` = track nova);
  ``observed``: True se ``fed`` é uma detecção.
- ``pred``: onde a track deve estar no quadro seguinte (o que a associação compara por IoU).

``StaticMotion`` é a regra da Parte 1 (a caixa fica onde foi vista). ``ConstantVelocityMotion``
é um baseline de comparação (velocidade constante suavizada; **não** é o modelo temporal da
Parte 2). ``RNNMotion`` é o modelo da Trilha A.
"""

from __future__ import annotations

import numpy as np
import torch

from pa2.models.motion_rnn import MotionRNN
from pa2.mot17.trajectories import cxcywh_to_ltwh, ltwh_to_cxcywh


class StaticMotion:
    name = "static"

    def step(self, states, fed, prev, observed):
        return [None] * len(fed), np.asarray(fed, dtype=np.float64).copy()


class ConstantVelocityMotion:
    """Velocidade constante em [cx, cy, w, h], suavizada: v <- (1-beta) v + beta (fed - prev)
    quando há observação; sem observação v não muda e a caixa segue em linha reta."""

    name = "const_vel"

    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def step(self, states, fed, prev, observed):
        fed_c = ltwh_to_cxcywh(fed)
        new_states, preds = [], np.zeros_like(fed_c)
        for i in range(len(fed_c)):
            v = states[i] if states[i] is not None else np.zeros(4)
            if prev is not None and prev[i] is not None and observed[i]:
                v = (1 - self.beta) * v + self.beta * (fed_c[i] - ltwh_to_cxcywh(prev[i]))
            new_states.append(v)
            p = fed_c[i] + v
            p[2:] = np.maximum(p[2:], 1.0)
            preds[i] = p
        return new_states, cxcywh_to_ltwh(preds)


class RNNMotion:
    """Modelo recorrente da Trilha A: um vetor de estado por track, avançado em lote."""

    def __init__(self, model: MotionRNN, img_size: tuple[float, float], dt: float = 1.0,
                 blind_damping: float = 1.0):
        # blind_damping < 1: nos passos SEM observação, o deslocamento previsto pela rede é
        # multiplicado por este fator (a velocidade extrapolada decai geometricamente). 1.0 = sem efeito.
        self.blind_damping = blind_damping
        self.model = model.eval()
        self.size = torch.tensor(img_size, dtype=torch.float32)
        self.dt = dt
        self.name = f"rnn_{model.rnn_type.lower()}"

    @torch.no_grad()
    def step(self, states, fed, prev, observed):
        n = len(fed)
        fed_t = torch.as_tensor(ltwh_to_cxcywh(fed), dtype=torch.float32)
        # tracks novas (prev None) e antigas entram juntas: para as novas, prev := fed (delta 0)
        prev_c = np.stack([ltwh_to_cxcywh(p if p is not None else f) for p, f in
                           zip(prev if prev is not None else [None] * n, fed)])
        prev_t = torch.as_tensor(prev_c, dtype=torch.float32)
        init = self.model.init_state(1)[0]
        state = torch.stack([s if s is not None else init for s in states])
        pred, new, _ = self.model.step(fed_t, prev_t, torch.as_tensor(observed, dtype=torch.float32),
                                       torch.full((n,), self.dt), self.size.expand(n, 2), state)
        if self.blind_damping != 1.0:
            blind = torch.as_tensor(~np.asarray(observed, dtype=bool))[:, None]
            pred = torch.where(blind, fed_t + self.blind_damping * (pred - fed_t), pred)
        return list(new.unbind(0)), cxcywh_to_ltwh(pred.numpy().astype(np.float64))
