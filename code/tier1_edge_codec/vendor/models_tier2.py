"""
models_tier2.py  --  TiDE-Lite and the mandated baselines.

Blueprint 4.2: these are PURE functions of parameters. No file I/O, no
device detection, no logging, no optimizer, no global state. Constructing
one needs only a config dataclass. That is what lets a Flower server later
simulate 100 clients in a single process.

TASK: DIRECT MULTI-STEP, H = 24. Targets are y[t+1 .. t+24] from a single
forward pass -- no recursion, no error accumulation, one model. The linear
baselines are Linear(L -> 24) with no weight sharing across output
columns, so column h is mathematically identical to a dedicated scalar
model for step h.

HISTORY (m1): an earlier revision targeted the single step t+24 and this
header warned against comparing its MAE to published ECL tables. That no
longer applies -- H=24 direct multi-step is the standard LTSF protocol.
H remains a config field, so the scalar variant is one edit away.

WHY LayerNorm AND NOT BatchNorm (blueprint 4.8.2)
------------------------------------------------
BatchNorm running statistics are stateful buffers. FedAvg averages them
across non-IID clients, which is incorrect, and the edge trains with
batches of 8-32 on heavily autocorrelated windows where batch statistics
are meaningless. There is no BatchNorm anywhere in this file, and there
must never be.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from normstats_tier2 import norm_stats_torch


@dataclass
class TiDEConfig:
    lookback_len: int = 336        # L
    horizon_len: int = 24          # H  -- direct multi-step y[t+1..t+24]
    horizon_lag: int = 1           # offset of the FIRST target step
    pre_ln: bool = True            # spec F2
    n_dynamic: int = 15            # C  (M17: rolling stats moved out)
    n_window: int = 6              # W  window-level constants
    n_static: int = 8              # S
    covar_proj: int = 4            # r
    covar_block_hidden: int = 32   # h_r of the feature projection ONLY -- see note
    static_proj: int = 64          # S_p
    hidden: int = 128              # d
    block_hidden: int = 128        # h_r
    decoder_out: int = 4           # p
    n_encoder: int = 2
    n_decoder: int = 2
    dropout: float = 0.15
    revin_sigma_floor: float = 0.05
    revin_eps: float = 1e-5

    @staticmethod
    def edge_lean():
        return TiDEConfig(lookback_len=168, hidden=64, block_hidden=64,
                          covar_proj=2, covar_block_hidden=16, static_proj=32,
                          decoder_out=2, n_encoder=1, n_decoder=1, dropout=0.10)


def _layer_norm(dim):
    """Guarded constructor. LayerNorm over a size-1 dimension computes
    (x - mean(x))/std(x) where the mean IS x: the output is identically zero
    and so is the gradient to its input. It silently killed 326,937 of 327,278
    TiDE-Lite parameters in an earlier run. It is a silent failure by
    construction, so it can never be built here."""
    assert dim > 1, (
        f"LayerNorm(normalized_shape={dim}) zeroes its own output and the "
        f"gradient to everything upstream. Use a plain Linear terminal layer.")
    return nn.LayerNorm(dim)


class ResidualBlock(nn.Module):
    """Pre-LN by default (spec F2): LN -> Linear -> ReLU -> Dropout -> Linear,
    plus skip.

    Post-LN -- LayerNorm(h + skip(x)) -- matches the TiDE paper but is
    measurably harder to optimise, and normalising AFTER the add is what put a
    LayerNorm on a size-1 output in the first place. Pre-LN normalises the
    INPUT, so the block's output dimension never constrains the norm and the
    terminal-layer failure mode cannot recur.

    WIDE-INPUT SKIP: when d_in >> d_out the projection skip is a second full
    d_in x d_out matrix. On the first encoder block (1748 -> 128) that is
    223,744 extra parameters -- alone it takes Edge-Full from 331 K to 554 K.
    A residual path from a 1748-dim concatenated feature vector is also just
    another linear layer, so the skip is dropped when
    d_in > skip_ratio_limit * d_out.
    """

    def __init__(self, d_in, d_hidden, d_out, dropout, pre_ln=True,
                 skip_ratio_limit=4.0):
        super().__init__()
        self.pre_ln = pre_ln
        self.norm = _layer_norm(d_in if pre_ln else d_out)
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(dropout)
        if d_in == d_out:
            self.skip = nn.Identity()
        elif d_in <= skip_ratio_limit * d_out:
            self.skip = nn.Linear(d_in, d_out, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        h = self.norm(x) if self.pre_ln else x
        h = self.lin2(self.drop(torch.relu(self.lin1(h))))
        if self.skip is not None:
            h = h + self.skip(x)
        return h if self.pre_ln else self.norm(h)


class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al., ICLR 2022) over the
    look-back window, with a learnable per-meter affine.

    THE LOSS MUST NOT PUT gamma/beta ON THE TARGET SIDE.
    Measured failure, not a hypothetical: computing the loss as
        crit(y_hat_n, gamma * z_true + beta)
    is exactly zero-able by driving gamma -> 0, because the model output
    carries the same gamma and both sides collapse onto the constant beta.
    Training loss went to 0.00000 within a few epochs while validation MAE
    climbed from 179 to 730 kWh and gamma fell from 1.0 to 0.0032 -- the model
    input gamma*z + beta had become a constant, so the network was seeing no
    signal at all. Compare in the PARAMETER-FREE z space instead: use
    deaffine() on the prediction and normalize the target with mu/sigma only.

    THE SIGMA FLOOR IS NOT COSMETIC. An interpolated outage or an idle meter
    produces a look-back window with near-zero spread; dividing by it sends the
    normalized target to four or five digits and the model diverges. `+eps` is
    not enough -- it has to be a floor, in the units of the log1p series.
    """

    def __init__(self, sigma_floor=0.05, eps=1e-5):
        super().__init__()
        self.sigma_floor = sigma_floor
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    @staticmethod
    def statistics(y, sigma_floor):
        """M4. This used torch's `.std(dim=-1)`, whose default is the UNBIASED
        estimator, while WindowBuilder reimplemented the same statistic in
        numpy with ddof=0. They differed by sqrt(L/(L-1)) -- 0.149% at L=336,
        0.525% at L=96 -- so the covariate anchors and the target lived in
        slightly different spaces, in the one place the docstring claimed
        there was a single implementation. Both sides now call
        normstats_tier2, whose self-test asserts they agree to 1e-7."""
        return norm_stats_torch(y, sigma_floor)

    # gamma is clamped away from zero in use. With the loss computed in z
    # space there is no longer an incentive to collapse it, but a sign flip
    # would invert every forecast, and this costs nothing to prevent.
    #
    # S5/S6 -- DECISION RECORDED. gamma is very nearly unidentifiable: it
    # scales both the input normalisation and the output de-affine, so an
    # approximately homogeneous network cancels it and the data supplies
    # almost no gradient to oppose weight decay. Two consequences were raised:
    # (a) AdamW decays it toward zero unopposed; (b) it is PERSONAL while the
    # trunk it feeds is GLOBAL, so clients can drift to different gamma scales
    # while their GLOBAL weights are averaged -- quietly breaking FedAvg
    # semantics at step 12. It was measured at 1.00000 and buys nothing.
    # Decision: keep the parameter (so checkpoints and the codec contract do
    # not change) but exclude it and every LayerNorm affine from weight decay
    # via parameter groups in 20_train_tier2. Note the clamp has ZERO gradient
    # below the floor, so a gamma that reaches 0.1 would be permanently stuck;
    # the trainer asserts it never gets close.
    GAMMA_MIN = 0.1

    def normalize(self, y):
        mu, sigma = self.statistics(y, self.sigma_floor)
        yn = (y - mu) / (sigma + self.eps)
        return self.gamma.clamp_min(self.GAMMA_MIN) * yn + self.beta, (mu, sigma)

    def deaffine(self, yn):
        """Model output -> parameter-free z space. This is what the loss uses."""
        return (yn - self.beta) / self.gamma.clamp_min(self.GAMMA_MIN)

    def target_z(self, y_true_log1p, stats):
        """Target in z space. Depends on data only -- no learnable parameter
        appears here, which is the whole point.

        M5: this was a @staticmethod with a hard-coded `eps=1e-5` default
        while `denormalize` used `self.eps`. They agreed only because the
        config happened to set 1e-5; changing it would have silently split the
        training target space from the inversion space. Now an instance method
        using self.eps, so they cannot diverge."""
        mu, sigma = stats
        return (y_true_log1p - mu) / (sigma + self.eps)

    def denormalize(self, yn, stats):
        mu, sigma = stats
        return self.deaffine(yn) * (sigma + self.eps) + mu       # m10


class TiDELite(nn.Module):
    """MLP-only encoder-decoder with covariates. Channel-independent: one
    shared model applied to each meter separately, with meter identity
    entering only through the computed static profile.

    forward(y_past, x_past, x_fut, s) -> (y_hat_norm, revin_stats)
      y_past : [B, L]      log1p load over the look-back
      x_past : [B, L, C]   dynamic covariates over the look-back
      x_fut  : [B, H, C]   dynamic covariates at the target step(s)
      s      : [B, S]      computed static profile
    """

    def __init__(self, cfg: TiDEConfig):
        super().__init__()
        self.cfg = cfg
        L, H, C, r, p, d = (cfg.lookback_len, cfg.horizon_len, cfg.n_dynamic,
                            cfg.covar_proj, cfg.decoder_out, cfg.hidden)

        self.revin = RevIN(cfg.revin_sigma_floor, cfg.revin_eps)
        # NOTE ON covar_block_hidden. The feature projection runs on every one
        # of the L past steps and every future step: at B=256, L=336 that is
        # 86,016 rows per batch, which makes it the single most expensive layer
        # in the model even though it is the smallest. Giving a 21 -> 4
        # compression a 128-wide inner layer costs 4x the training time and
        # 4x the on-device inference time for a map that has nowhere near that
        # much to express. Measured: 263 ms -> 78 ms per step at B=256 on 4
        # CPU cores. Sweep it if you like, but start here, not at block_hidden.
        self.feature_proj = ResidualBlock(C, cfg.covar_block_hidden, r,
                                          cfg.dropout, cfg.pre_ln)
        # A dedicated module rather than "the first S columns of encoder block 1".
        # Functionally the same composition, but it gives the codec a clean
        # named parameter to mark PERSONAL and trainable, instead of needing to
        # slice a weight matrix -- which no ParameterCodec can do safely.
        self.static_proj = nn.Linear(cfg.n_static, cfg.static_proj)

        # M17: the window-level constants join here, once, instead of being
        # broadcast across L+H timesteps through the feature projection.
        e_in = L + (L + H) * r + cfg.static_proj + cfg.n_window
        self.encoder = nn.ModuleList(
            [ResidualBlock(e_in if i == 0 else d, cfg.block_hidden, d,
                           cfg.dropout, cfg.pre_ln)
             for i in range(cfg.n_encoder)])

        dec = []
        for i in range(cfg.n_decoder):
            last = (i == cfg.n_decoder - 1)
            dec.append(ResidualBlock(d, cfg.block_hidden, H * p if last else d,
                                     cfg.dropout, cfg.pre_ln))
        self.decoder = nn.ModuleList(dec)

        # Spec F1: fix properly, not minimally. The terminal layer is a plain
        # Linear, not a ResidualBlock -- a residual block whose output is a
        # scalar has no meaningful skip path either. Shared across the H
        # horizon steps.
        self.temporal_decoder = nn.Sequential(
            _layer_norm(p + r), nn.Linear(p + r, cfg.block_hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.block_hidden, 1))
        self.global_skip = nn.Linear(L, H)
        self.encoder_input_dim = e_in

    def forward(self, y_past, x_past, x_fut, s, w=None):
        B = y_past.shape[0]
        cfg = self.cfg
        if w is None:
            w = y_past.new_zeros(B, cfg.n_window)

        y_n, stats = self.revin.normalize(y_past)                  # [B, L]

        xp_past = self.feature_proj(x_past)                        # [B, L, r]
        xp_fut = self.feature_proj(x_fut)                          # [B, H, r]

        e = torch.cat([y_n,
                       xp_past.reshape(B, -1),
                       xp_fut.reshape(B, -1),
                       self.static_proj(s), w], dim=1)
        for blk in self.encoder:
            e = blk(e)

        g = e
        for blk in self.decoder:
            g = blk(g)
        g = g.reshape(B, cfg.horizon_len, cfg.decoder_out)          # [B, H, p]

        t = self.temporal_decoder(torch.cat([g, xp_fut], dim=-1))   # [B, H, 1]
        y_hat_n = t.squeeze(-1) + self.global_skip(y_n)             # [B, H]
        return y_hat_n, stats

    def to_kwh(self, y_hat_n, stats):
        """Inverse in the mandated order: RevIN inverse -> expm1 -> clip 0."""
        return torch.expm1(self.revin.denormalize(y_hat_n, stats)).clamp_min(0.0)


# =====================================================================
# Baselines. Same forward signature so the trainer and evaluator are shared.
# Blueprint 1.4: TiDE-Lite does not ship unless it beats the best of these by
# >= 3% MAE. The literature says that margin is not a given on this dataset.
# =====================================================================
class _BaselineBase(nn.Module):
    def __init__(self, cfg: TiDEConfig):
        super().__init__()
        self.cfg = cfg
        self.revin = RevIN(cfg.revin_sigma_floor, cfg.revin_eps)

    def to_kwh(self, y_hat_n, stats):
        return torch.expm1(self.revin.denormalize(y_hat_n, stats)).clamp_min(0.0)


class DLinear(_BaselineBase):
    """Trend/seasonal decomposition, one Linear per branch. 2*L*H params."""

    def __init__(self, cfg, kernel=25):
        super().__init__(cfg)
        self.kernel = kernel
        self.trend = nn.Linear(cfg.lookback_len, cfg.horizon_len)
        self.seasonal = nn.Linear(cfg.lookback_len, cfg.horizon_len)

    def forward(self, y_past, x_past, x_fut, s, w=None):
        y_n, stats = self.revin.normalize(y_past)
        pad = self.kernel // 2
        x = torch.nn.functional.pad(y_n.unsqueeze(1), (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(x, self.kernel, stride=1).squeeze(1)
        return self.trend(trend) + self.seasonal(y_n - trend), stats


class NLinear(_BaselineBase):
    """Subtract the last look-back value, one Linear, add it back."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lin = nn.Linear(cfg.lookback_len, cfg.horizon_len)

    def forward(self, y_past, x_past, x_fut, s, w=None):
        y_n, stats = self.revin.normalize(y_past)
        last = y_n[:, -1:]
        return self.lin(y_n - last) + last, stats


class SparseTSF(_BaselineBase):
    """Cross-period sparse forecasting, period w = 24 (Lin et al., ICML 2024).

    The canonical form: fold the look-back into (L/w, w), then apply ONE
    Linear(L/w -> H/w) shared across all w phases. With L=336, H=24, w=24 that
    is Linear(14 -> 1): 15 parameters for the whole model. It is a serious
    competitor at that size, not a formality -- do not skip it because it
    looks trivial.
    """

    def __init__(self, cfg, period=24):
        super().__init__(cfg)
        assert cfg.lookback_len % period == 0 and cfg.horizon_len % period == 0, \
            "SparseTSF requires L and H to be whole multiples of the period"
        self.period = period
        self.n_in = cfg.lookback_len // period
        self.n_out = cfg.horizon_len // period
        self.smooth = nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False)
        self.lin = nn.Linear(self.n_in, self.n_out)

    def forward(self, y_past, x_past, x_fut, s, w=None):
        y_n, stats = self.revin.normalize(y_past)
        y_s = self.smooth(y_n.unsqueeze(1)).squeeze(1) + y_n        # sparse smoothing
        # (B, L) -> (B, n_in, w) -> (B, w, n_in) : one row per phase
        ph = y_s.reshape(-1, self.n_in, self.period).transpose(1, 2)
        out = self.lin(ph).transpose(1, 2).reshape(-1, self.cfg.horizon_len)
        return out, stats


class LinearAug(_BaselineBase):
    """MATCHED COMPARATOR FOR RIDGE (review Blocking 2).

    The superset argument -- "Ridge now contains DLinear's features, so if it
    still loses, something is broken" -- is invalid as an inference about kWh
    MASE. Ridge and DLinear differ in three ways, not one: features, OBJECTIVE
    (squared error closed-form vs Huber via SGD) and REGULARISATION (ridge
    lambda vs weight decay plus implicit SGD/early-stop). A feature superset
    only implies Ridge wins on SQUARED ERROR IN NORMALISED SPACE. It implies
    nothing about kWh MAE, because expm1 amplifies large normalised residuals
    and Huber deliberately down-weights exactly the tails that pure L2 fits
    hard. On a dataset with RMSE/MAE around 10, that is precisely where the
    reported metric lives -- so Ridge can lose on MASE with everything working.

    This model removes the confound. Identical feature set to Ridge (lags,
    per-horizon covariates, window statistics, static profile, intercept),
    trained through the SAME trainer with the SAME Huber loss, optimiser,
    schedule and budget as DLinear. The DLinear -> LinearAug gap is then the
    covariate effect with no objective or sample-size confound, which is what
    the S4 decomposition claim actually needs. Ridge stays as the closed-form
    reference row.

    Per-horizon weights throughout, matching Ridge's per-step solve: at
    L=336, H=24, C=15, S=8, W=6 that is 24*(336+15+14+1) = 8,784 parameters,
    the same as the ridge design.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        H = cfg.horizon_len
        self.lag = nn.Linear(cfg.lookback_len, H)
        self.cov = nn.Parameter(torch.zeros(H, cfg.n_dynamic))
        self.ctx = nn.Linear(cfg.n_static + cfg.n_window, H)

    def forward(self, y_past, x_past, x_fut, s, w=None):
        y_n, stats = self.revin.normalize(y_past)
        if w is None:
            w = y_past.new_zeros(y_past.shape[0], self.cfg.n_window)
        out = self.lag(y_n)
        out = out + (x_fut * self.cov.unsqueeze(0)).sum(-1)   # per-horizon block
        return out + self.ctx(torch.cat([s, w], dim=-1)), stats


MODELS = {"tide_lite": TiDELite, "dlinear": DLinear, "nlinear": NLinear,
          "sparsetsf": SparseTSF, "linear_aug": LinearAug}


def build(name: str, cfg: TiDEConfig) -> nn.Module:
    return MODELS[name](cfg)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def falsification_test(cfg=None, verbose=True):
    """Spec 4.4 -- PERMANENT REGRESSION TEST. Run before believing any
    TiDE-Lite result.

    TiDE-Lite contains a linear model as a strict subclass: its global skip is
    Linear(L -> H) applied to the RevIN-normalized look-back. Zero the temporal
    decoder and only that path remains, so it must be able to reproduce NLinear
    exactly. NLinear computes W(y_n - y_L) + b + y_L, which is linear in y_n,
    so the equivalent skip matrix is available in closed form:

        W'[h, j] = W[h, j]                        for j < L-1
        W'[h, L-1] = W[h, L-1] + 1 - sum_j W[h, j]

    If this does not match to numerical noise, there is a bug -- not a
    training-budget problem. This is the test that would have caught the
    LayerNorm(1) defect on the first run instead of after a full evaluation.
    """
    cfg = cfg or TiDEConfig()
    torch.manual_seed(0)
    tide = TiDELite(cfg).eval()
    nlin = NLinear(cfg).eval()

    with torch.no_grad():
        # 1. silence the temporal decoder: only the global skip survives
        tide.temporal_decoder[-1].weight.zero_()
        tide.temporal_decoder[-1].bias.zero_()
        # 2. transplant NLinear's function into the skip, in closed form
        W, b = nlin.lin.weight, nlin.lin.bias
        Wp = W.clone()
        Wp[:, -1] += 1.0 - W.sum(dim=1)
        tide.global_skip.weight.copy_(Wp)
        tide.global_skip.bias.copy_(b)
        # 3. same RevIN affine on both sides
        tide.revin.gamma.copy_(nlin.revin.gamma)
        tide.revin.beta.copy_(nlin.revin.beta)

        B, L, C, S = 32, cfg.lookback_len, cfg.n_dynamic, cfg.n_static
        y = torch.randn(B, L) * 3 + 7
        args = (y, torch.randn(B, L, C), torch.randn(B, cfg.horizon_len, C),
                torch.randn(B, S), torch.randn(B, cfg.n_window))
        a, _ = tide(*args)
        e, _ = nlin(*args)
        err = float((a - e).abs().max())

    # 4. gradient coverage: nothing may be structurally unreachable
    t2 = TiDELite(cfg).train()
    o, _ = t2(torch.randn(4, cfg.lookback_len),
              torch.randn(4, cfg.lookback_len, cfg.n_dynamic),
              torch.randn(4, cfg.horizon_len, cfg.n_dynamic),
              torch.randn(4, cfg.n_static), torch.randn(4, cfg.n_window))
    o.sum().backward()
    dead = [n for n, p in t2.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]

    if verbose:
        print(f"falsification test (spec 4.4)")
        print(f"  skip path reproduces NLinear : max |diff| = {err:.2e}  "
              f"{'PASS' if err < 1e-4 else 'FAIL'}")
        print(f"  gradient coverage            : {len(dead)} dead tensors  "
              f"{'PASS' if not dead else 'FAIL ' + str(dead)}")
    assert err < 1e-4, f"skip path does not reproduce NLinear (max diff {err})"
    assert not dead, f"parameters receive no gradient: {dead}"
    return err


if __name__ == "__main__":
    for name in MODELS:
        for cfg, tag in ((TiDEConfig(), "Edge-Full"),
                         (TiDEConfig.edge_lean(), "Edge-Lean")):
            m = build(name, cfg)
            B, L, H, C, S = 4, cfg.lookback_len, cfg.horizon_len, cfg.n_dynamic, cfg.n_static
            out, st = m(torch.randn(B, L), torch.randn(B, L, C),
                        torch.randn(B, H, C), torch.randn(B, S),
                        torch.randn(B, cfg.n_window))
            assert out.shape == (B, H), (name, tag, out.shape)
            n, _ = count_params(m)
            extra = f"  enc_in={m.encoder_input_dim}" if hasattr(m, "encoder_input_dim") else ""
            print(f"{name:10s} {tag:9s} params {n:8,d}  out {tuple(out.shape)}{extra}")
    print()
    falsification_test()
