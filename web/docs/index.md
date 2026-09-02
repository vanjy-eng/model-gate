# BDP Model Gate

Automated **pre-deployment governance** for machine-learning models. Fairness,
performance, compliance, security and validation-methodology checks run as a
single gate that returns
one status your pipeline can branch on.

<span class="verdict-pass">PASS</span> deploy ·
<span class="verdict-review">NEEDS_REVIEW</span> stop for sign-off ·
<span class="verdict-blocked">BLOCKED</span> hard fail

```bash
pip install "bdp-model-gate[structured]"
```

It runs **after a model is trained and before it is promoted** — not on every
pull request. The output is a report a reviewer can read months later: it
records which metric ran, which checks were skipped, and why. Take it as JSON
for the system that files it, or as a
[self-contained HTML page](reference/reports.md) for the person who has to
sign it off.

## Where to go

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)** — install, gate your first model,
  read the report
- **[Concepts](concepts.md)** — contexts, checks, verdicts, and the
  degradation contract
- **[Tasks](tasks/binary.md)** — binary, multiclass and ordinal, regression,
  insurance pricing
- **[Any model](models.md)** — scikit-learn, PyTorch, XGBoost, remote endpoints
- **[Generative side-cars](security.md)** — prompt injection, canaries, and the
  two surfaces
- **[Reference](reference/checks.md)** — the checks, configuration, CLI, API
- **[Reports and plots](reference/reports.md)** — the page a reviewer signs,
  and the fourteen charts in it
- **[Examples](examples/index.md)** — eight runnable notebooks

</div>

## What it checks

| Category | Blocking | Checks |
|---|---|---|
| Validation | Yes | target leakage, split overlap, validation strategy, feature contract, feature drift (review) |
| Fairness | No → review | proxy correlation, demographic parity, SHAP subgroup gaps, counterfactual flip, equalised odds, subgroup calibration |
| Fairness (regression) | No → review | loss-ratio parity, group mean gap, error parity, calibration parity |
| Performance | Yes | model score, calibration, p95 latency, cost per inference, actual-vs-expected, risk discrimination |
| Compliance | Yes | model-card completeness, DPIA trigger, explainability requirement, monotonicity, prediction dislocation (review) |
| Security | Yes | adversarial robustness, PII leakage, prompt injection, report injection (review) |

Validation comes first on purpose. A performance finding says the model is not
good enough; a validation finding says you do not yet know whether it is — and
nothing stops you passing the training set as the validation set, which used
to report a superb score and a clean `PASS`. See
[Concepts](concepts.md#is-the-evidence-sound).

Fairness is deliberately non-blocking. Those findings frequently need human
judgement, so they route to a reviewer rather than failing a build — which is
the distinction most CI gates collapse.

The last four in the performance and compliance rows are **pricing** measures,
and they are why the suite is worth pointing at a rate filing rather than a
generic regression report. An RMSE cannot say whether a book is under-priced,
whether the shortfall sits in one decile, whether the rating structure orders
risk at all, whether premium still rises with prior claims, or who gets a 30%
increase. See [Insurance pricing](tasks/insurance.md).

The security row changed shape in 0.5.4. `prompt_injection` no longer asks
*"did the model refuse?"* — a question no string can answer — and it tests
both the direct and the **indirect** surface, which is where the realistic
attack against a regulated pipeline arrives. `report_injection` is the odd
one out: its victim is whatever reads the report next, which is increasingly
an LLM. See [Generative side-cars](security.md).

The fairness checks cover all three families — independence, separation and
sufficiency — which cannot all hold at once whenever base rates differ between
groups. Reporting one without the others makes that choice silently; see
[Fairness: three families](tasks/fairness.md).

## Regulatory defaults

Out of the box the compliance and PII checks target **NDPA/NDPR** (Nigeria's
data-protection regime): PII patterns match NIN, BVN and Nigerian phone
formats, and pricing, underwriting, credit scoring and claims decisioning are
treated as DPIA triggers.

These are **defaults, not assumptions**. Every pattern, required field and
high-risk use case is configurable — see
[Configuration](reference/configuration.md) — so the same suite works against
GDPR, CCPA or an internal standard.

!!! warning "Not regulatory advice"
    Default thresholds are reasonable starting points chosen to be useful, not
    positions on what any regulator requires. Set them with your compliance
    function.
