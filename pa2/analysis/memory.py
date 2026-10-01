"""Horizonte de memória do modelo (Parte 4), das duas formas pedidas.

1. **Analítica**: ``||dL_t / dh_{t-k}||`` em função de k, no modelo e nos dados (autograd sobre a
   janela desenrolada). Duas situações: ``observed`` (todas as entradas são detecções) e
   ``blind`` (depois de ``context`` quadros observados, a rede só recebe as próprias previsões,
   como numa oclusão; o gradiente também passa pelo caminho previsão -> entrada).
2. **Empírica**: injeta oclusões de N quadros (remove as detecções de uma identidade) e mede
   se, quando ela reaparece, a detecção volta com o MESMO id da track. Compara com a
   distribuição de duração dos buracos que existem de verdade no dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from pa2.analysis.runner import match_gt_to_dets, run_trace
from pa2.models.motion_rnn import MotionRNN, S, add_obs_noise, box_error
from pa2.mot17.loader import Sequence


# ─────────────────────────────────────────────────────────────────────────────
# 1. horizonte analítico
# ─────────────────────────────────────────────────────────────────────────────
def gradient_horizon(model: MotionRNN, segments, sigma, L: int = 48, context: int = 8, n: int = 400,
                     mode: str = "observed", seed: int = 0) -> np.ndarray:
    """Normas por amostra de dL_t/dh_{t-k}: matriz (n, L), coluna k = 0..L-1.

    ``L`` passos desenrolados; a perda é a smooth-L1 (a do treino) do ÚLTIMO passo t = L-1
    contra o GT. ``h_j`` é o estado depois do passo j.
    """
    assert mode in ("observed", "blind")
    rng = np.random.default_rng(seed)
    eligible = [s for s in segments if len(s) >= L + 1]
    lens = np.array([len(s) - L for s in eligible], dtype=float)
    picks = [(eligible[i], int(rng.integers(0, int(lens[i])))) for i in
             rng.choice(len(eligible), size=n, p=lens / lens.sum())]
    gt = torch.as_tensor(np.stack([s.boxes[a:a + L + 1] for s, a in picks]), dtype=torch.float32)
    size = torch.as_tensor(np.array([s.size for s, _ in picks]), dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    obs = add_obs_noise(gt.double(), sigma, gen).float()

    model.eval()
    dt = torch.ones(n)
    fed, prev, state = obs[:, 0], None, model.init_state(n)
    observed = torch.ones(n)
    hs, pred = [], None
    for i in range(L):
        pred, state, _ = model.step(fed, prev, observed, dt, size, state)
        hs.append(state)
        if mode == "observed" or i + 1 < context:
            nxt, observed = obs[:, i + 1], torch.ones(n)
        else:
            nxt, observed = pred, torch.zeros(n)
        prev, fed = fed, nxt
    err = box_error(pred, gt[:, L]) * S
    loss = F.smooth_l1_loss(err, torch.zeros_like(err), reduction="none").mean(-1).sum()
    grads = torch.autograd.grad(loss, hs)                      # grads[j] = dL/dh_j, (n, dim)
    norms = torch.stack([g.norm(dim=-1) for g in grads], 1)    # (n, L), coluna j
    return norms.flip(1).detach().numpy()                       # coluna k = (L-1) - j


def horizon_summary(norms: np.ndarray) -> dict:
    """Mediana por k, normalizada pelo k = 0, e os k em que ela cai abaixo de 1/2, 1/10 e 1/100."""
    med = np.median(norms, axis=0)
    rel = med / med[0]
    out = {"median": med.tolist(), "relative": rel.tolist()}
    for frac in (0.5, 0.1, 0.01):
        below = np.nonzero(rel < frac)[0]
        out[f"k_below_{frac}"] = int(below[0]) if len(below) else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# duração dos buracos no dataset
# ─────────────────────────────────────────────────────────────────────────────
def detection_gap_runs(seq: Sequence, D: np.ndarray, max_run: int = 10**9) -> pd.DataFrame:
    """Buracos de DETECÇÃO: quadros seguidos, entre dois quadros detectados, em que uma identidade
    do GT não tem nenhuma detecção (IoU >= 0,5) — o tempo que o estado precisa sobreviver."""
    m = match_gt_to_dets(seq, D)
    gt = seq.gt_pedestrians()
    rows = []
    for gid in np.unique(gt[:, 1]).astype(int):
        life = gt[gt[:, 1] == gid]
        fs = life[np.argsort(life[:, 0]), 0].astype(int)
        vis = dict(zip(life[:, 0].astype(int), life[:, 6]))
        hit = [(int(f), (gid, int(f)) in m) for f in fs]
        run, start = 0, None
        seen = False
        for f, ok in hit:
            if ok:
                if run > 0 and seen:
                    rows.append({"video": seq.video, "gt_id": gid, "start": start, "length": run,
                                 "mean_visibility": float(np.mean([vis[x] for x in range(start, start + run)
                                                                   if x in vis]))})
                run, seen = 0, True
            elif seen:
                if run == 0:
                    start = f
                run += 1
    return pd.DataFrame(rows, columns=["video", "gt_id", "start", "length", "mean_visibility"])


def visibility_runs(seq: Sequence, thr: float = 0.2) -> pd.DataFrame:
    """Oclusão segundo o GT: sequências de quadros com visibilidade < thr dentro da vida da identidade."""
    gt = seq.gt_pedestrians()
    rows = []
    for gid in np.unique(gt[:, 1]).astype(int):
        life = gt[gt[:, 1] == gid]
        life = life[np.argsort(life[:, 0])]
        low = life[:, 6] < thr
        i = 0
        while i < len(low):
            if low[i]:
                j = i
                while j + 1 < len(low) and low[j + 1] and life[j + 1, 0] == life[j, 0] + 1:
                    j += 1
                rows.append({"video": seq.video, "gt_id": gid, "start": int(life[i, 0]), "length": j - i + 1})
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows, columns=["video", "gt_id", "start", "length"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. horizonte empírico: oclusões injetadas
# ─────────────────────────────────────────────────────────────────────────────
PRE = 5          # quadros observados antes (a track precisa estar confirmada)
POST = 1         # quadros depois do reaparecimento que também precisam existir


def sample_trials(match: dict, seq: Sequence, N: int, rng: np.random.Generator, per_run: int,
                  ids_pool: list[int]) -> list[tuple[int, int]]:
    """Até ``per_run`` oclusões (gid, s) de N quadros: a identidade tem detecção em s-PRE..s-1 e em
    s+N..s+N+POST, e nenhuma outra oclusão da mesma execução se sobrepõe no tempo."""
    T = seq.info.seq_length
    cand = []
    for gid in ids_pool:
        for s in range(PRE + 1, T - N - POST):
            if all((gid, f) in match for f in range(s - PRE, s)) and \
               all((gid, f) in match for f in range(s + N, s + N + POST + 1)):
                cand.append((gid, s))
    rng.shuffle(cand)
    chosen, taken = [], []
    for gid, s in cand:
        lo, hi = s - PRE - 2, s + N + POST + 2
        if all(hi < a or lo > b for a, b in taken) and gid not in {c[0] for c in chosen}:
            chosen.append((gid, s))
            taken.append((lo, hi))
            if len(chosen) >= per_run:
                break
    return chosen


def survival_experiment(seq: Sequence, D: np.ndarray, motions: dict, assoc: dict, Ns, reps: int = 4,
                        per_run: int = 12, seed: int = 0) -> pd.DataFrame:
    """Uma linha por (método, N, tentativa): desfecho ``kept`` | ``swapped`` | ``reborn``.

    ``kept``: a detecção que reaparece recebe o MESMO id que a identidade tinha antes da oclusão;
    ``swapped``: recebe o id de OUTRA track que já existia (troca de identidade);
    ``reborn``: recebe um id novo (a track original morreu ou foi perdida).
    """
    match = match_gt_to_dets(seq, D)
    ids_pool = sorted({g for g, _ in match})
    rng = np.random.default_rng(seed + int(seq.video))
    rows = []
    for N in Ns:
        for rep in range(reps):
            trials = sample_trials(match, seq, N, rng, per_run, ids_pool)
            if not trials:
                continue
            keep = np.ones(len(D), bool)
            for gid, s in trials:
                for f in range(s, s + N):
                    if (gid, f) in match:
                        keep[match[(gid, f)]] = False
            for name, make in motions.items():
                assign, _ = run_trace(D, seq.info.seq_length, make(seq), keep=keep, **assoc)
                row2id = [{r: tid for tid, r in a} for a in assign]
                for gid, s in trials:
                    before = row2id[s - 2].get(match[(gid, s - 1)])
                    after = row2id[s + N - 1].get(match[(gid, s + N)])
                    if before is None:
                        continue                                    # a track ainda não existia
                    existing = set(row2id[s - 2].values())
                    outcome = ("kept" if after == before else
                               "swapped" if after in existing else "reborn")
                    rows.append({"video": seq.video, "method": name, "N": N, "gt_id": gid, "start": s,
                                 "outcome": outcome})
    return pd.DataFrame(rows, columns=["video", "method", "N", "gt_id", "start", "outcome"])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def n50_curve(Ns, kept) -> float:
    """Menor N em que a fração que mantém o id cai abaixo de 0,5 (interpolação linear entre os N medidos)."""
    x, y = np.asarray(Ns, float), np.asarray(kept, float)
    for i in range(1, len(x)):
        if y[i] < 0.5 <= y[i - 1]:
            return float(x[i - 1] + (y[i - 1] - 0.5) / (y[i - 1] - y[i]) * (x[i] - x[i - 1]))
    return float(x[-1]) if y[-1] >= 0.5 else float(x[0])


def bootstrap_n50(trials: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """N50 por método com IC95% por bootstrap das TENTATIVAS (cada N reamostrado à parte), mais a
    diferença RNN − caixa parada e RNN − velocidade constante.

    O IC cobre só a variação de amostragem das oclusões injetadas (≈330 por N). Tentativas do mesmo
    vídeo são correlacionadas, então o intervalo é otimista; não cobre variação entre seeds de treino.
    Linhas: ``method``/``contraste``, ``n50``, ``ci_lo``, ``ci_hi``, ``p_gt_0`` (só nos contrastes).
    """
    rng = np.random.default_rng(seed)
    Ns = sorted(trials.N.unique())
    kept = {(m, n): (trials[(trials.method == m) & (trials.N == n)].outcome == "kept").to_numpy()
            for m in trials.method.unique() for n in Ns}
    boots = {m: np.array([n50_curve(Ns, [rng.choice(kept[(m, n)], len(kept[(m, n)])).mean() for n in Ns])
                          for _ in range(n_boot)]) for m in trials.method.unique()}
    rows = []
    for m, b in boots.items():
        rows.append({"metodo": m, "n50": n50_curve(Ns, [kept[(m, n)].mean() for n in Ns]),
                     "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
                     "p_gt_0": np.nan})
    for a, b_ in (("rnn", "static"), ("rnn", "const_vel")):
        if a in boots and b_ in boots:
            d = boots[a] - boots[b_]
            rows.append({"metodo": f"{a} - {b_}", "n50": float(d.mean()), "ci_lo": float(np.percentile(d, 2.5)),
                         "ci_hi": float(np.percentile(d, 97.5)), "p_gt_0": float((d > 0).mean())})
    return pd.DataFrame(rows)


def survival_table(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, N), g in trials.groupby(["method", "N"]):
        n, k = len(g), int((g.outcome == "kept").sum())
        lo, hi = wilson(k, n)
        rows.append({"method": m, "N": N, "n": n, "kept": k / n, "ci_lo": lo, "ci_hi": hi,
                     "swapped": float((g.outcome == "swapped").mean()),
                     "reborn": float((g.outcome == "reborn").mean())})
    return pd.DataFrame(rows).sort_values(["method", "N"]).reset_index(drop=True)


def blind_displacement_regression(make_motion, segments, sigma, context: int = 8, ks=(5, 10, 20),
                                  n: int = 1200, seed: int = 1) -> pd.DataFrame:
    """A extrapolação às cegas exagera ou encolhe o movimento?

    Para cada amostra (``context`` quadros observados com ruído de detector, depois k às cegas) compara o
    deslocamento horizontal PREVISTO do centro com o VERDADEIRO, em larguras da caixa. ``slope`` é a
    regressão pela origem de previsto sobre verdadeiro (>1 exagera, <1 encolhe), ``corr`` a correlação e
    ``wrong_sign`` a fração em que o sentido está errado (entre os casos em que a pessoa andou mais de
    0,1 largura)."""
    from pa2.mot17.trajectories import cxcywh_to_ltwh
    k_max = max(ks)
    rng = np.random.default_rng(seed)
    eligible = [s for s in segments if len(s) >= context + k_max]
    lens = np.array([len(s) - context - k_max + 1 for s in eligible], dtype=float)
    picks = [(eligible[i], int(rng.integers(0, int(lens[i])))) for i in
             rng.choice(len(eligible), size=n, p=lens / lens.sum())]
    sg = np.asarray(sigma)
    rows = {k: ([], []) for k in ks}
    for size in sorted({p[0].size for p in picks}):
        idx = [i for i, p in enumerate(picks) if p[0].size == size]
        gt = np.stack([picks[i][0].boxes[picks[i][1]: picks[i][1] + context + k_max] for i in idx])
        nz = rng.standard_normal((len(idx), context, 4)) * sg
        c = gt[:, :context]
        obs = np.stack([c[..., 0] + nz[..., 0] * c[..., 2], c[..., 1] + nz[..., 1] * c[..., 3],
                        c[..., 2] * np.exp(nz[..., 2]), c[..., 3] * np.exp(nz[..., 3])], -1)
        obs_l, gt_l = cxcywh_to_ltwh(obs), cxcywh_to_ltwh(gt)
        motion, m = make_motion(size), len(idx)
        states, prev = [None] * m, [None] * m
        for j in range(context):
            states, pred = motion.step(states, obs_l[:, j], prev, np.ones(m, bool))
            prev = list(obs_l[:, j])
        last_cx = obs_l[:, context - 1, 0] + obs_l[:, context - 1, 2] / 2
        w = obs_l[:, context - 1, 2]
        for k in range(1, k_max + 1):
            if k in rows:
                rows[k][0].append((pred[:, 0] + pred[:, 2] / 2 - last_cx) / w)
                rows[k][1].append((gt_l[:, context - 1 + k, 0] + gt_l[:, context - 1 + k, 2] / 2 - last_cx) / w)
            fed = pred
            states, pred = motion.step(states, fed, prev, np.zeros(m, bool))
            prev = list(fed)
    out = []
    for k in ks:
        dp, dt = np.concatenate(rows[k][0]), np.concatenate(rows[k][1])
        moving = np.abs(dt) > 0.1
        out.append({"k": k, "slope": float((dp * dt).sum() / (dt * dt).sum()),
                    "corr": float(np.corrcoef(dp, dt)[0, 1]),
                    "wrong_sign": float((np.sign(dp[moving]) != np.sign(dt[moving])).mean()),
                    "mean_abs_true": float(np.abs(dt).mean()), "mean_abs_pred": float(np.abs(dp).mean())})
    return pd.DataFrame(out)
