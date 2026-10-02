"""RNN como modelo de movimento (Parte 2, Trilha A).

Um estado recorrente por track. A cada passo a célula (RNN simples, GRU ou LSTM) recebe a
caixa "alimentada" (a observação ou, se não houve observação, a própria previsão anterior)
e prevê a caixa do passo seguinte. A associação usa o IoU entre essa caixa prevista e as
detecções do quadro.

Parametrização (invariante a escala e à resolução): para uma caixa ``b = [cx, cy, w, h]``
a rede emite ``o = [ox, oy, ow, oh]`` e a próxima caixa é

    cx' = cx + w * ox/S      cy' = cy + h * oy/S
    w'  = w * exp(ow/S)      h'  = h * exp(oh/S)           (S = 10: o ~ 1 é 10% da caixa)

Features de entrada (``N_FEATURES = 10``): deslocamento relativo da caixa alimentada em
relação à anterior (4, escalados por S); log(h/H) e log(w/h) (tamanho e proporção);
posição normalizada cx/W - 0.5, cy/H - 0.5; Δt (quadros até o próximo passo); e a flag
``observada`` (1 = detecção, 0 = a própria previsão, i.e. oclusão). A confiança do detector
não entra: os scores do SDP/FRCNN saturam em 1,0.

Perda: smooth-L1 sobre o erro da caixa prevista em relação ao GT, em unidades da caixa
(``[dcx/w, dcy/h, log w, log h]`` vezes S). O GT do MOT17 é contínuo, então há alvo mesmo
nos passos em que o modelo está "cego" (buraco de observação).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

S = 10.0
N_FEATURES = 10
CELLS = {"RNN": nn.RNNCell, "GRU": nn.GRUCell, "LSTM": nn.LSTMCell}


def decode(box: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Caixa seguinte [cx, cy, w, h] a partir da caixa atual e da saída ``o`` da rede."""
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx + w * o[..., 0] / S, cy + h * o[..., 1] / S,
                        w * torch.exp(o[..., 2] / S), h * torch.exp(o[..., 3] / S)], -1)


def box_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Erro em unidades da caixa verdadeira: [dcx/w, dcy/h, log(w_p/w), log(h_p/h)]."""
    return torch.stack([(pred[..., 0] - gt[..., 0]) / gt[..., 2], (pred[..., 1] - gt[..., 1]) / gt[..., 3],
                        torch.log(pred[..., 2] / gt[..., 2]), torch.log(pred[..., 3] / gt[..., 3])], -1)


def box_iou_cxcywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = a[..., 0] - a[..., 2] / 2, a[..., 1] - a[..., 3] / 2, a[..., 0] + a[..., 2] / 2, a[..., 1] + a[..., 3] / 2
    bx1, by1, bx2, by2 = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    iw = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp(min=0)
    ih = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp(min=0)
    inter = iw * ih
    return inter / (a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter).clamp(min=1e-9)


class MotionRNN(nn.Module):
    """Célula recorrente + cabeça que prevê o movimento até a próxima amostra."""

    def __init__(self, rnn_type: str = "GRU", hidden_size: int = 64, use_delta_t: bool = True,
                 num_layers: int = 1, dropout: float = 0.0, predict_uncertainty: bool = False):
        super().__init__()
        rnn_type = rnn_type.upper()
        if rnn_type not in CELLS:
            raise ValueError(f"rnn_type deve ser um de {list(CELLS)}, veio {rnn_type!r}")
        if num_layers != 1:
            raise NotImplementedError("só 1 camada recorrente por enquanto")
        if predict_uncertainty:
            raise NotImplementedError("incerteza/portão adaptativo ainda não implementado")
        self.rnn_type, self.hidden_size, self.use_delta_t = rnn_type, hidden_size, use_delta_t
        self.n_state = 2 if rnn_type == "LSTM" else 1
        self.cell = CELLS[rnn_type](N_FEATURES, hidden_size)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, 4))
        nn.init.zeros_(self.head[-1].weight)           # começa prevendo "fica parado"
        nn.init.zeros_(self.head[-1].bias)

    @property
    def config(self) -> dict:
        return {"rnn_type": self.rnn_type, "hidden_size": self.hidden_size,
                "use_delta_t": self.use_delta_t}

    # ── um passo ────────────────────────────────────────────────────────────
    def features(self, fed, prev, observed, dt, size):
        """fed/prev (B,4) cxcywh; prev pode ser None (primeiro passo); observed/dt (B,); size (B,2)."""
        if prev is None:
            d = torch.zeros_like(fed)
        else:
            d = torch.stack([(fed[:, 0] - prev[:, 0]) / prev[:, 2], (fed[:, 1] - prev[:, 1]) / prev[:, 3],
                             torch.log(fed[:, 2] / prev[:, 2]), torch.log(fed[:, 3] / prev[:, 3])], -1) * S
        W, H = size[:, 0], size[:, 1]
        extra = torch.stack([
            torch.log(fed[:, 3] / H) + 1.5,
            torch.log(fed[:, 2] / fed[:, 3]) + 0.9,
            fed[:, 0] / W - 0.5,
            fed[:, 1] / H - 0.5,
            dt if self.use_delta_t else torch.zeros_like(dt),
            observed,
        ], -1)
        return torch.cat([d, extra], -1)

    def init_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.n_state * self.hidden_size, device=device)

    def step(self, fed, prev, observed, dt, size, state):
        """Devolve (caixa prevista (B,4), novo estado (B, n_state*H), saída crua o (B,4))."""
        x = self.features(fed, prev, observed, dt, size)
        H = self.hidden_size
        if self.rnn_type == "LSTM":
            h, c = self.cell(x, (state[:, :H], state[:, H:]))
            new = torch.cat([h, c], -1)
        else:
            h = self.cell(x, state)
            new = h
        o = self.head(h)
        return decode(fed, o), new, o

    # ── janela inteira (treino / avaliação em trajetórias do GT) ─────────────
    def rollout(self, obs, can_observe, size, dt, tf_ratio: float = 1.0,
                generator: torch.Generator | None = None):
        """Percorre uma janela de L passos.

        obs  (B, L+1, 4): observações (GT com ruído) cxcywh;  can_observe (B, L+1) bool:
        False = buraco (sem observação nesse passo);  dt (B, L).
        Em cada passo i+1 a entrada é a observação com prob. ``tf_ratio`` (se existir) ou a
        própria previsão do passo i. Devolve as previsões (B, L, 4) para os passos 1..L e
        a flag de quanto cada entrada foi observada (B, L+1).
        """
        B, L1 = obs.shape[:2]
        fed, prev = obs[:, 0], None
        state = self.init_state(B, obs.device)
        observed = torch.ones(B, device=obs.device)
        flags, preds = [observed], []
        for i in range(L1 - 1):
            pred, state, _ = self.step(fed, prev, observed, dt[:, i], size, state)
            preds.append(pred)
            if tf_ratio >= 1.0:
                use = can_observe[:, i + 1]
            else:
                coin = torch.rand(B, device=obs.device, generator=generator) < tf_ratio
                use = can_observe[:, i + 1] & coin
            prev, fed = fed, torch.where(use[:, None], obs[:, i + 1], pred)
            observed = use.float()
            flags.append(observed)
        return torch.stack(preds, 1), torch.stack(flags, 1)


# ─────────────────────────────────────────────────────────────────────────────
# treino
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainSettings:
    """Hiperparâmetros de treino. Os padrões reproduzem a receita da Parte 2.

    ``tf_ratio`` é a probabilidade de alimentar a observação quando ela existe (1.0 = teacher
    forcing; 0.0 = free-running). Com ``tf_schedule = "linear"`` ela decai linearmente de
    ``tf_ratio`` até ``tf_end`` ao longo dos passos de treino (scheduled sampling).
    A validação usa um protocolo FIXO (``val_*``), igual para todos os regimes, para que a
    perda/IoU de validação e a escolha da época sejam comparáveis entre eles.
    """
    epochs: int = 30
    steps_per_epoch: int = 100
    batch_size: int = 64
    lr: float = 2e-3
    window_T: int = 32
    tf_ratio: float = 1.0
    tf_schedule: str = "constant"      # constant | linear
    tf_end: float = 0.0
    gap_prob: float = 0.5
    max_gap: int = 20
    obs_noise: tuple = (0.06, 0.025, 0.08, 0.045)
    clip: bool = True
    clip_value: float = 1.0
    seed: int = 42
    select: str = "best"               # best = melhor perda de validação | last = última época
    val_tf_ratio: float = 1.0
    val_gap_prob: float = 0.5
    val_max_gap: int = 20


def tf_ratio_at(cfg: TrainSettings, step: int, total_steps: int) -> float:
    """Probabilidade de alimentar a observação no passo global ``step`` (0-based)."""
    if cfg.tf_schedule == "constant":
        return cfg.tf_ratio
    if cfg.tf_schedule == "linear":
        frac = step / max(1, total_steps - 1)
        return cfg.tf_ratio + (cfg.tf_end - cfg.tf_ratio) * frac
    raise ValueError(f"tf_schedule deve ser 'constant' ou 'linear', veio {cfg.tf_schedule!r}")


def make_gaps(valid: torch.Tensor, gap_prob: float, max_gap: int, gen: np.random.Generator) -> torch.Tensor:
    """can_observe (B, L+1): buracos de 1..max_gap passos em janelas escolhidas com prob. gap_prob.

    Cada janela sorteada recebe de 1 a 2 buracos. O passo 0 sempre é observado."""
    B, L1 = valid.shape
    can = valid.clone()
    for b in range(B):
        if gen.random() < gap_prob:
            for _ in range(int(gen.integers(1, 3))):
                g = int(gen.integers(1, max_gap + 1))
                s = int(gen.integers(1, max(2, L1 - g)))
                can[b, s:s + g] = False
    return can


def add_obs_noise(boxes: torch.Tensor, sigma, gen: torch.Generator) -> torch.Tensor:
    """Simula o erro de um detector: [cx/w, cy/h, log w, log h] com desvio ``sigma``."""
    sg = torch.as_tensor(sigma, dtype=boxes.dtype)
    n = torch.randn(boxes.shape, generator=gen, dtype=boxes.dtype) * sg
    return torch.stack([boxes[..., 0] + n[..., 0] * boxes[..., 2], boxes[..., 1] + n[..., 1] * boxes[..., 3],
                        boxes[..., 2] * torch.exp(n[..., 2]), boxes[..., 3] * torch.exp(n[..., 3])], -1)


def window_loss(model: MotionRNN, batch, cfg: TrainSettings, gen_np, gen_t, noisy: bool = True,
                with_gaps: bool = True, tf_ratio: float | None = None):
    boxes, valid, size, dt = (torch.as_tensor(x) for x in batch)
    obs = add_obs_noise(boxes, cfg.obs_noise, gen_t) if noisy else boxes
    can = make_gaps(valid, cfg.gap_prob, cfg.max_gap, gen_np) if with_gaps else valid.clone()
    tf = cfg.tf_ratio if tf_ratio is None else tf_ratio
    preds, _ = model.rollout(obs.float(), can, size.float(), dt.float(), tf, gen_t)
    gt = boxes[:, 1:].float()
    err = box_error(preds, gt) * S
    loss_el = F.smooth_l1_loss(err, torch.zeros_like(err), reduction="none").mean(-1)
    m = valid[:, 1:].float()
    return (loss_el * m).sum() / m.sum().clamp(min=1), preds, gt, m


def _val_settings(cfg: TrainSettings) -> TrainSettings:
    """Protocolo de validação: o mesmo para qualquer regime de treino."""
    return replace(cfg, tf_ratio=cfg.val_tf_ratio, tf_schedule="constant",
                   gap_prob=cfg.val_gap_prob, max_gap=cfg.val_max_gap)


@torch.no_grad()
def eval_windows(model: MotionRNN, batch, cfg: TrainSettings, seed: int = 0) -> dict:
    """Perda e IoU médio de 1 passo e de rollout com buracos, em janelas fixas (validação).

    Usa o protocolo ``val_*`` de ``cfg`` (teacher forcing + buracos simulados por padrão),
    independente do regime de treino."""
    model.eval()
    vcfg = _val_settings(cfg)
    gen_np, gen_t = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    loss, preds, gt, m = window_loss(model, batch, vcfg, gen_np, gen_t)
    iou = box_iou_cxcywh(preds, gt)
    return {"loss": float(loss), "iou": float((iou * m).sum() / m.sum())}


@torch.no_grad()
def eval_shift(model: MotionRNN, batch, cfg: TrainSettings, seed: int = 0) -> dict:
    """Mede a diferença entre ser alimentado com observações e ser alimentado só com as
    próprias previsões (a "deriva" do enunciado), nas MESMAS janelas e com o mesmo ruído.

    - ``iou_obs``  : teacher forcing, sem buracos (como na inferência com detecção em todo quadro);
    - ``iou_gaps`` : teacher forcing com buracos de observação (protocolo de validação);
    - ``iou_free`` : free-running: só o 1º quadro é observado, o resto é a própria previsão;
    - ``iou_free_by_step``: IoU do free-running em função do passo à frente (1..L).
    Previsões não finitas contam como IoU 0 (e são contadas em ``nonfinite_frac``).
    """
    model.eval()
    vcfg = _val_settings(cfg)
    out: dict = {}
    for name, tf, gaps in (("obs", 1.0, False), ("gaps", 1.0, True), ("free", 0.0, False)):
        gen_np, gen_t = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
        _, preds, gt, m = window_loss(model, batch, vcfg, gen_np, gen_t, with_gaps=gaps, tf_ratio=tf)
        finite = torch.isfinite(preds).all(-1)
        iou = box_iou_cxcywh(torch.nan_to_num(preds, nan=1.0, posinf=1.0, neginf=1.0), gt)
        iou = torch.where(finite, iou, torch.zeros_like(iou))
        out[f"iou_{name}"] = float((iou * m).sum() / m.sum())
        out[f"nonfinite_frac_{name}"] = float(((~finite).float() * m).sum() / m.sum())
        if name == "free":
            out["iou_free_by_step"] = [float(x) for x in
                                       (iou * m).sum(0) / m.sum(0).clamp(min=1)]
    return out


def params_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def train_motion_rnn(model: MotionRNN, train_sampler, val_batch, cfg: TrainSettings,
                     log=print, save_to: str | Path | None = None, extra_meta: dict | None = None) -> dict:
    """Treina em trajetórias do GT.

    ``cfg.select = "best"`` guarda o modelo de menor perda de validação; ``"last"`` guarda o da
    última época (sem seleção por validação). Devolve o histórico por época:
    ``train_loss``, ``val_loss``, ``val_iou``, ``grad_norm`` (média), ``grad_norm_max``,
    ``clipped_frac`` (fração dos passos em que a norma passou de ``clip_value``, mesmo com o
    clipping desligado), ``nonfinite_loss`` / ``nonfinite_grad`` (nº de passos), ``tf_ratio``
    (valor no fim da época), e ``diverged`` / ``diverged_epoch`` (pesos não finitos: o treino
    para nessa época).
    """
    torch.manual_seed(cfg.seed)
    gen_np = np.random.default_rng(cfg.seed)
    gen_t = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.05)
    hist: dict = {"train_loss": [], "val_loss": [], "val_iou": [], "grad_norm": [], "grad_norm_max": [],
                  "clipped_frac": [], "nonfinite_loss": [], "nonfinite_grad": [], "tf_ratio": [],
                  "best_epoch": 0, "diverged": False, "diverged_epoch": None}
    best = float("inf")
    total_steps = cfg.epochs * cfg.steps_per_epoch
    step = 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tl, gn, nbad, ngrad, nclip = [], [], 0, 0, 0
        for _ in range(cfg.steps_per_epoch):
            tf = tf_ratio_at(cfg, step, total_steps)
            step += 1
            loss, *_ = window_loss(model, train_sampler.sample(cfg.batch_size), cfg, gen_np, gen_t,
                                   tf_ratio=tf)
            opt.zero_grad()
            if not torch.isfinite(loss):
                nbad += 1
                continue
            loss.backward()
            norm = float(nn.utils.clip_grad_norm_(model.parameters(),
                                                  cfg.clip_value if cfg.clip else float("inf")))
            if np.isfinite(norm):
                gn.append(norm)
                nclip += norm > cfg.clip_value
            else:
                ngrad += 1
            opt.step()
            tl.append(float(loss))
        sched.step()
        val = eval_windows(model, val_batch, cfg)
        hist["train_loss"].append(float(np.mean(tl)) if tl else float("nan"))
        hist["val_loss"].append(val["loss"])
        hist["val_iou"].append(val["iou"])
        hist["grad_norm"].append(float(np.mean(gn)) if gn else float("nan"))
        hist["grad_norm_max"].append(float(np.max(gn)) if gn else float("nan"))
        hist["clipped_frac"].append(float(nclip / max(1, len(gn))))
        hist["nonfinite_loss"].append(nbad)
        hist["nonfinite_grad"].append(ngrad)
        hist["tf_ratio"].append(float(tf))
        if cfg.select == "best" and val["loss"] < best:
            best, hist["best_epoch"] = val["loss"], ep
            if save_to is not None:
                save_checkpoint(model, save_to, {"settings": cfg.__dict__, "epoch": ep, **(extra_meta or {})})
        if ep == 1 or ep % 5 == 0 or ep == cfg.epochs:
            log(f"  época {ep:3d}/{cfg.epochs}  treino {hist['train_loss'][-1]:.4f}  "
                f"val {val['loss']:.4f}  IoU val {val['iou']:.3f}  |g| {hist['grad_norm'][-1]:.2f}"
                f"{'  (laços não finitos: %d)' % nbad if nbad else ''}  [{time.time() - t0:.0f}s]")
        if not params_finite(model):
            hist["diverged"], hist["diverged_epoch"] = True, ep
            log(f"  !! pesos não finitos na época {ep}: treino interrompido")
            break
    hist["epochs_run"] = len(hist["train_loss"])
    if cfg.select == "last":
        hist["best_epoch"] = hist["epochs_run"]
        if save_to is not None:
            save_checkpoint(model, save_to, {"settings": cfg.__dict__, "epoch": hist["epochs_run"],
                                             "diverged": hist["diverged"], **(extra_meta or {})})
    hist["seconds"] = time.time() - t0
    return hist


# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model: MotionRNN, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.config, "state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path: str | Path) -> tuple[MotionRNN, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MotionRNN(**ck["model"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("meta", {})
