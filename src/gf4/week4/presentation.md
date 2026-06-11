# GF4 Week 4 — 5-minute presentation deck

**Key message:** *We built a working incremental SfM system; the gap **to** COLMAP is bundle adjustment + feature density, and the gap **past** COLMAP is learned correspondence.*

**Canonical numbers (use these everywhere — all from `out/week4-doge-noretri/`):**
- Final reconstruction: **78 / 169 cameras**, **24,587 points**, reprojection **median ≈0.58 px** (gate-bound; read exact off a run summary if needed).
- Retriangulation ablation: off **78 / 24,587** vs on **75 / 23,420** → **no improvement** (needs BA).
- COLMAP (same 169): **169/169**, **114k points**. Partial-doge (both 20/20 cams): ours **6,412** pts / **2,001** kp/img vs COLMAP **22,882** pts / **10,673** kp/img.
- Failure: **91 rejects**, ~376 footholds each but PnP ratio **median 0.00 / max 0.19** (registered ≥0.31); multi-model recovers **+29**.

**Figures:** all in `out/presentation_figs/` (embedded below).
**Speaker split (3):** A = slides 1–2 + close · B = 3–4 · C = 5–6. (2 people: split at slide 4.) Rehearse transitions — they eat the 5 minutes.

---

## Slide 1 — Title + the problem  *(Speaker A, ~20s)*


**On slide:**
- Title: *Incremental Structure-from-Motion & COLMAP Comparison*
- "169 phone photos of a statue → 3D points + camera poses"

**Script:** "Structure-from-Motion recovers both a 3D scene *and* where every camera was, from nothing but 2D photos. We extended our two-view pipeline into a full incremental system and benchmarked it against COLMAP on 169 images of this statue."

---

## Slide 2 — End-to-end method + our design choices  *(Speaker A, ~50s)*

**Figure:** slide-drawn pipeline strip (recreate in editor):
```
[169 images] → [SIFT] → [match (Lowe)] → [verify: F + RANSAC]
   → [seed: parallax gate] → [2-view init: E → pose → triangulate]
   → ┌── incremental loop ──────────────────────────────┐
     │ pick next (max 2D-3D) → PnP (ratio gate) →        │── repeat
     │ triangulate new (angle-filtered) → retriangulate  │
     └───────────────────────────────────────────────────┘
   → [sparse cloud + camera poses]
```
Reuses Weeks 1–3 up to "2-view init"; the loop is the Week-4 contribution.

**On slide — 3 defended choices:**
- **Seed by triangulation *angle* (parallax)**, not inlier count — high-match pairs are often degenerate low-baseline.
- **Next image = most 2D-3D correspondences** (greedy).
- **Accept by PnP inlier *ratio***, not raw count — the gate that keeps the cloud clean (evidence on slide 5).
- *(retriangulation = explored extension → ablated on slide 3)*

**Script:** "We reuse our Weeks 1–3 blocks — SIFT, matching, geometric verification, two-view geometry — and add the incremental layer. Two choices matter most: we pick the seed by *parallax*, because the highest-match pairs triangulate terribly; and we accept a new camera by the *fraction* of its matches the pose explains, not the raw count — that one gate is what keeps the reconstruction clean."

---

## Slide 3 — Final reconstruction + the retriangulation ablation  *(Speaker B, ~60s)*

**On slide:**
- Headline: **78 / 169 cameras, 24,587 points, ≈0.58 px reprojection**
- Frusta trace the capture path; gaps = where it stalled.
- Ablation: retriangulation **78 → 75** cameras → **no improvement**.

**Script:** "Here's our final system — 78 cameras around the statue, sub-pixel reprojection. We also implemented retriangulation and ablated it: it made *no* difference — 78 versus 75 cameras, within run-to-run noise. That's actually informative: re-triangulating against poses we never refined can't improve the model. So the missing ingredient isn't retriangulation — it's bundle adjustment, which we'll come back to."

---

## Slide 4 — COLMAP comparison  *(Speaker B→C, ~60s)*

**On slide — the required quantitative table:**

| | Ours | COLMAP |
|---|---|---|
| Full set, cameras | 78/169 | **169/169** |
| Full set, points | 24.6k | **114k** |
| Same 20 cams, points | 6.4k | **22.9k** |
| Reprojection | ≈0.58 px | ~1.0 px |

**Script:** "COLMAP registers all 169 and builds ~5× our points. But the most telling row is the third: on the *same 20 cameras*, COLMAP still has 3× more points — purely because its detector finds ~5× more keypoints per image, 10,700 versus our 2,000. Our triangulation efficiency is comparable. So the *density* gap is feature extraction, not the reconstruction engine — and *coverage* is a separate gap."

---

## Slide 5 — Key failure: why 91 images can't register  *(Speaker C, ~50s)*

**On slide:**
- Rejects have ~376 correspondences each — **not** starved — but PnP ratio **median 0.00 (max 0.19)** → matches are *geometrically invalid*.
- Registered images sit at **ratio ≥0.31**; the 0.30 gate falls in the **empty gap** → robust, not arbitrary.
- No threshold registers them. **Multi-model** (a 2nd model on the rejects) recovers **+29**.

**Script:** "Our most interesting finding: the 91 unregistered images aren't short of correspondences — they have ~400 each. But PnP explains essentially *zero* of them — their matches to our cloud are false. This split is bimodal: rejects below 0.19, accepted above 0.31, with our gate cleanly in the gap. No threshold fixes it, so we ran a *second* model on just the rejects and recovered 29 — that's the multi-model insight, backed by data."

---

## Slide 6 — What's missing, and beyond COLMAP  *(Speaker A/C, ~60s)*

**On slide — two columns (text):**
- **To reach COLMAP:** **bundle adjustment** (kills drift, makes retriangulation actually pay off) · **denser features** (lower SIFT contrast threshold) · **multi-model + Sim(3) merge** for disjoint clusters.
- **To fix COLMAP's *own* limits** (Week-1 link): it still fails on **textureless / repetitive / pure-rotation / low-overlap** scenes → needs **learned features + matching** (SuperPoint / SuperGlue / LoFTR) and priors / loop-closure — exactly the hand-crafted SIFT+RANSAC failure modes we analysed in Week 1.

**Script:** "To match COLMAP we'd need bundle adjustment and a denser detector — our own ablation showed retriangulation is wasted without BA. But COLMAP isn't the ceiling: it shares SIFT and RANSAC's blind spots — repetitive texture, no parallax, little overlap, the failure cases from our Week 1 analysis. Going beyond it needs *learned* correspondence. So the one-line takeaway: the gap *to* COLMAP is optimisation and density; the gap *past* it is learning."

---

## README minimums — coverage check
- [x] final method → slide 2
- [x] show final reconstruction → slides 1, 3
- [x] COLMAP comparison (quant + qual) → slide 4
- [x] one key quantitative table/metric → slide 4 table + slide 5 histogram
- [x] one important success/failure → slide 5 (failure) / slide 3 (ablation)
- [x] what's missing vs COLMAP + what would fix COLMAP's own limits → slide 6
- [x] "~10+ images, high quality" → partial-doge 20/20 clean is the exemplar; full 169 is the stress-test (mention in slide 1 or 4)
