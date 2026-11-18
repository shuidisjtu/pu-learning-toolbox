# Choosing a PU Method

> Prerequisite: complete the [quickstart](../quickstart.md).
> Background: [the PU problem and the roles of π](pu_problem.md), and
> [SCAR/SAR mechanisms](scar_sar.md).

## 1. Choose explicit candidates

Inspect registered capabilities with `pu-toolbox list-methods` and the
[classifier reference](../reference/api/classifiers.md). Select candidates according to the
sampling design, labeling assumptions, prior requirements, available supervision, and budget.
Use [PUTuner or PUModelComparator](../howto/model_tuning.md) to evaluate explicit candidates
with appropriate validation. Profiles inform judgment; they do not determine a ranking.

## 2. The role of π in method selection

- **Methods requiring an external π:** risk-based methods such as uPU, nnPU, PNU,
  LLSVM, and Dist-PU, and deep methods such as Self-PU, WConPU, and DGPU use a prior
  in their risks or weighted objectives. An incorrect prior can substantially change
  predictions, including recall. Interpret each prior in the distribution consumed
  by the method's risk view.
- **Methods not requiring an external π:** Elkan-Noto estimates the labeling
  probability c; LBE can use an optional prior to initialize its posterior;
  InfoMax PU can estimate a prior internally. The current `pusb` baseline does not
  consume the supplied prior. `requires_class_prior=False` means the caller may omit
  an external prior, rather than establishing that the method has no prior dependence.
- **When π is unknown:** estimate it with recpe / pen_l1 / km1 / km2, or choose a
  method that accepts an omitted external prior. `PUPipeline` can connect prior
  estimation to training when the selected method needs it.

See [prior semantics](class_prior_semantics.md) for prior targets,
the [pipeline guide](../howto/pipeline.md) for resolution priority and automatic
estimation, and [sensitivity analysis](../howto/sensitivity_analysis.md) for checking
the effect of prior assumptions.

## 3. Choose according to data conditions

Start with the properties of your data:

| Data condition | Options to investigate | Reason and limits |
|---|---|---|
| π is known | uPU / nnPU / PNU / LLSVM / Dist-PU | Use the prior in a risk or objective under the method's assumptions; verify the required prior target |
| π is unknown | Estimate π first, or consider Elkan-Noto / InfoMax PU | Prior estimation is an upstream step that `PUPipeline` can connect automatically |
| Some labeled negatives are available | PNU | Combines PN/PU/NU risks; use the PNU label contract explicitly |
| Feature-dependent labeling is suspected (SAR) | Review LBE / DGPU and the distinct `pusb` / `pusb_kernel` paths | Check each implementation's assumptions and additional requirements; the `pusb` baseline alone does not identify the latent class posterior |
| Small data (< 1000 samples) | Elkan-Noto and other shallow methods | Compare simpler models before paying the fixed training cost of deep or fixed-epoch methods |
| Large data and a GPU | nnPU / Dist-PU / Self-PU / WConPU / InfoMax PU | Investigate representation learning when data and budget allow; Self-PU also requires clean validation support |
| Very few labeled positives | Simple baselines such as Elkan-Noto / `pusb` | Deep methods may not improve on simple baselines with limited labeling |
| Audit evidence is needed | Supply `y_true` for audited profiling and oracle metrics | Audit truth supports identifying evidence and supervised evaluation; keep it out of PU training and selection |

This table supports manual review; it is not an empirical performance guarantee.

## 4. Algorithm-family ideas

Understanding the objective of a family is more useful than memorizing a method list:

- **Class Prior Estimation** (penL1 / KM1/KM2 / ReCPE) estimates π upstream and does
  not itself classify examples. penL1 uses convex optimization, Kernel Mean uses
  kernel mean matching, and ReCPE uses a discriminative regrouping procedure.
- **Risk Estimation** (uPU / nnPU / PNU / Dist-PU / LDCE / KLDCE / LLSVM) uses or
  derives a prior in a risk/objective. nnPU corrects the negative empirical risk
  problem of uPU; PNU adds labeled-negative information; Dist-PU models label
  distributions. The precise assumptions and risk forms differ by method.
- **Classic & Calibration** (Elkan-Noto) estimates the labeling probability and
  corrects outputs, providing a simple baseline under SCAR.
- **Bias-Aware / SAR** includes different implementation routes. The current
  `pusb` is an observed-P-vs-U logistic baseline, rather than the paper's partially
  identified scoring algorithm or a calibrated latent-class posterior.
  `pusb_kernel` is a separate kernel-method entry. LBE explicitly estimates
  labeling bias; DGPU combines discriminative and generative components. DGPU's
  registry family is `Fam.DEEP_PU`, despite its relevance to selection bias.
- **Deep PU** (Self-PU / InfoMax PU / WConPU / DGPU) combines representation learning
  with method-specific PU objectives. InfoMax PU uses information-theoretic
  representations and internal prior estimation; WConPU uses weighted contrastive
  learning and hard-negative mining; Self-PU combines self-training, meta
  reweighting, and distillation and requires clean validation support.

[Method cards](../../research/method_cards/) (optional Chinese research background)
provide formulas, reproduction status, and implementation boundaries.

## 5. Risks to review

1. If SCAR fails, SCAR-based methods such as Elkan-Noto, uPU, and nnPU can be biased.
2. Incorrect π estimates can substantially affect prior-dependent objectives; use
   sensitivity analysis to bound the effect.
3. With few labeled positives, deep methods may not outperform simple baselines.
4. When selection bias is apparent, investigate how labels were obtained and compare
   methods under their own identification assumptions. A SAR metadata declaration
   alone does not establish that selection bias is corrected.
5. Read candidate reasons and warnings, including experimental maturity, prior
   requirements, and SCAR risk, as well as the score.

## 6. Other methods in the field

TIcE and AlphaMax (prior estimation), and PU Bagging, Biased SVM, and Weighted LR
(classifiers) are established methods in the broader PU literature. The toolbox
does not currently implement them; they are listed here for context.

## Next steps

- Run the selected method in a complete [pipeline](../howto/pipeline.md).
- Check precise parameter contracts in the [API reference](../reference/api/classifiers.md#classifiers-and-estimators).
