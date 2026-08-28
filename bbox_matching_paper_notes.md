# 2D-to-3D Object Localization from a Detection Bounding Box

Working notes on the bbox→point-cloud matching method (`bench_bbox_to_tree.py`,
`visualize_bbox_match.py`, `interactive_bbox_match.py`), organized for reuse in a
paper's Method / Evaluation / Limitations sections. Numbers below are measured on
real captured data (`data/calib_indoor_level`, `data/calib_data_yard_2`), not
synthetic.

## 1. Problem statement

Given a 2D object detection (e.g. a YOLO bounding box) in one camera view of a
calibrated LiDAR-camera rig, recover the 3D **center, bottom, and top** points of
the corresponding physical object in the LiDAR point cloud. The extrinsic
`T_cam_lidar` (or any per-view solved extrinsic) is assumed known from a prior
calibration step; this method consumes it, it does not solve for it.

The core difficulty is not projection — that part is exact, given a correct
extrinsic and pinhole intrinsics — it is **object isolation**: a detection box is
an axis-aligned rectangle in image space, but its corresponding camera frustum in
3D passes through everything along that viewing direction: the target object, any
background surface behind it, and any occluder or neighboring object whose
silhouette overlaps the box edges. A tight box mitigates this; a real detector's
box is rarely perfectly tight.

## 2. Method

### 2.1 Projection model

Standard pinhole projection, camera frame to image:

```
[X, Y, Z]^T = R · p_lidar + t          (extrinsic: LiDAR frame -> camera frame)
u = fx * X / Z + cx
v = fy * Y / Z + cy
depth = ||[X, Y, Z]||_2
```

`(fx, fy, cx, cy)` are known in closed form for the synthesized rectilinear views
this rig uses (see the calibration README's "virtual-view detour"), so no
distortion model is needed.

### 2.2 Pipeline

**Algorithm 1 — bbox → object keypoints**

```
Input:  point cloud P (N x 3), extrinsic (R, t), intrinsics (fx,fy,cx,cy),
        detection box (x1,y1,x2,y2), up-axis a
Output: center, bottom, top (3D points), or NONE

1.  (u, v, Z, depth) <- project(P, R, t, fx, fy, cx, cy)         # O(N), whole cloud
2.  C <- { p in P : x1<=u(p)<x2, y1<=v(p)<y2, Z(p) > near_clip } # frustum select, O(N)
3.  d* <- mode(depth(C))  via 40-bin histogram peak
    G <- { p in C : |depth(p) - median(depth(C) near d*)| < 3 * 1.4826 * MAD }
                                                                   # depth-mode gate
4.  {K_1..K_m} <- connected_components(G, voxel=0.08m, 6-connectivity)
                                                                   # union-find, O(m alpha(m))
5.  K_best <- argmax_i |K_i|                                      # default: largest
    if reproj_dist(centroid(K_best), bbox_center) > 0.5 * half_diag(bbox):
        K_alt <- argmin over {K_i : reproj_dist(K_i) < 0.3*half_diag} of reproj_dist
        if K_alt exists: K_best <- K_alt
    for each other K_i with |K_i| <= 3*|K_best| and
                    ||centroid(K_i) - centroid(K_best)||_2 < 0.3m:
        K_best <- K_best U K_i                                    # merge near-fragments
6.  center <- median(K_best)
    bottom  <- K_best point at 1st percentile of axis a
    top     <- K_best point at 99th percentile of axis a
```

Step 4 is a from-scratch 6-connected voxel-grid union-find (no `scipy.ndimage.label`
or `sklearn.cluster.DBSCAN`) — both were unavailable in the target deployment
environment (a numpy2/scipy ABI mismatch on the dev host, and an explicit intent to
avoid the dependency weight on the embedded target), and the candidate-set sizes
involved (10^2–10^4 points, never the full cloud) make an O(m) grid-based approach
fast enough without them.

### 2.3 Design rationale — why steps 3–5 are not simplifiable

Two properties motivate the specific rules in steps 3–5, each discovered as a
concrete failure while validating on real data (Section 4), not chosen a priori:

- **Depth alone under-separates.** Two physically distinct surfaces at similar
  range (a chair ~0.15 m in front of a wall panel) survive the same MAD window
  together. Step 4's connected-component split is necessary; a single depth gate
  is not sufficient to isolate the object.

- **Neither "largest cluster" nor "most central cluster" is independently
  correct.** A background surface that leaks into a loose box is usually the
  larger cluster (solid surfaces return denser LiDAR hits than thin object
  frames), so picking by size alone selects the background. But a small stray
  cluster can be more central in image space than a large, correctly-shaped
  object whose box isn't perfectly centered on it, so picking by centrality alone
  can lose the real object to noise. Step 5 defaults to size (the correct choice
  in the common case of a reasonably tight box) and only overrides that default
  when the size-leading cluster is clearly off-center *and* a clearly-central
  alternative of non-trivial size exists — encoding "a background surface leaked
  in" as a specific, checkable condition rather than a general preference.

### 2.4 Complexity

Let N = full cloud size, n = |candidates in frustum| (n << N), m = |depth-gated
points|. Step 1 is O(N) (unavoidable per new camera pose, but not per detection —
see Section 5). Steps 2–6 are O(n) to O(m log m) (voxel hashing + union-find with
path compression), operating only on the small frustum subset. In the measurements
below, n was 2–3 orders of magnitude smaller than N.

## 3. Implementation notes

- Percentile (not min/max) for top/bottom: robust to the occasional stray return
  (e.g. a bird, a wire, a floor straggler) that would otherwise set the reported
  extent from a single point.
- Median (not mean) for center: same robustness argument against residual
  outliers that survive gating.
- All thresholds (voxel size 0.08 m, MAD multiplier 3.0, merge radius 0.3 m, merge
  size ratio 3x, centrality overrides at 0.5/0.3 × half-diagonal) were tuned
  against the two real scenes in Section 4, not learned or exhaustively swept —
  flagged explicitly as a limitation (Section 6).

## 4. Empirical validation: two failure modes and their fixes

Both cases below use a solved extrinsic (`configs/results/extrinsic_NN.txt`, the
`livox_camera_calib` output for that view), not a seed guess — feeding a seed
guess into this pipeline produces plausible-looking but geometrically wrong
matches, since frustum select in step 2 depends on projection accuracy the seed
guess doesn't have (only ~5–10° accurate by design).

### Case A — background-surface rejection (indoor office scene)

A hand-drawn box around an office chair, in front of a fabric partition panel at
a similar range (depth-gated median range 4.41 m, gated points spanning
4.25–4.57 m — a 0.32 m window containing both surfaces).

| | candidates (step 2) | depth-gated (step 3) | connected components (step 4) |
|---|---|---|---|
| count | 21,477 | 5,154 | 7 clusters, sizes 2308 / 1300 / 1127 / 356 / 31 / 22 / 10 |

The naive baseline (no clustering, just "shrink toward the depth-gated centroid")
converges on the 2,308-point partition cluster — the wrong object. Largest-cluster
selection alone makes the same mistake. The centrality-override rule in step 5
correctly identifies the 356-point chair cluster (reprojected centroid 27.1 px
from the box center, vs. 75.8 px for the partition) and merges in a 31-point
fragment (a separated part of the chair's return), yielding a final 387-point
object cluster with center `(3.20, 2.94, 0.11)`.

### Case B — large-object retention against a distractor (outdoor tree trunk)

A box around a visible tree trunk, well-aligned and reasonably tight.

| | candidates (step 2) | depth-gated (step 3) | connected components (step 4) |
|---|---|---|---|
| count | 23,784 | 11,181 | 3 clusters, sizes 11,120 / 59 / 2 |

Here, a pure "nearest-cluster-to-box-center" rule (the naive fix for Case A) fails
in the opposite direction: the 59-point noise cluster reprojects closer to the box
center (28.5 px) than the true 11,120-point trunk cluster (80.3 px), so a
centrality-only rule discards the correct object almost entirely. The combined
rule in step 5 (default to largest; override only under the specific off-center +
central-alternative condition) keeps the trunk, since 80.3 px does not clear the
0.5×half-diagonal (83.5 px) override threshold for this box.

**Together, these two cases are why step 5 cannot be replaced by either single
heuristic** — each one independently fails one of the two cases while "fixing"
the other. This pair is a natural minimal ablation/regression pair for testing
any future change to the cluster-selection rule.

## 5. Runtime performance

Measured on a laptop CPU (Intel i7-10750H), single-threaded (`OMP_NUM_THREADS=1`
made no measurable difference — the projection step's 3×N matmul is too thin to
benefit from BLAS multithreading), no GPU/accelerator involved. Point cloud sizes:
~1.17M points (Case A dataset), ~1.18M points (Case B dataset).

| stage | Case A, base clock (~0.9 GHz) | Case B, base clock (~0.9 GHz) | Case A, turbo (~2.6–4 GHz) |
|---|---|---|---|
| project (whole cloud) | 63.4 ms | 64.7 ms | 23.6 ms |
| frustum mask | 21.7 ms | 22.3 ms | 6.1 ms |
| depth-mode gate | 4.0 ms | 5.4 ms | 1.0 ms |
| clustering | 31.4 ms | 50.7 ms | 8.5 ms |
| keypoints | 0.7 ms | 1.9 ms | 0.25 ms |
| **total** | **121.1 ms** | **145.1 ms** | **40.7 ms** |
| rate | 8.3 matches/s | 6.9 matches/s | 24.5 matches/s |

The two clock regimes on the *same host* (governor-throttled to ~0.9 GHz vs.
observed turbo of ~2.6–4 GHz during sustained earlier runs) show a ~3× runtime
change for a ~3–4.4× clock-frequency change — i.e., this workload is
close to clock-bound for a fixed microarchitecture, not dominated by fixed
overhead. That is a useful (if informal) calibration point for extrapolating to
a different CPU by clock ratio alone, though it does not account for differing
per-cycle throughput (IPC) or SIMD width across architectures (see below).

**Split that matters for a real-time loop:** projection (step 1) is the largest
single cost and scales with the *whole* cloud, but it does not need to be
recomputed per detection — only when the camera pose or the cloud changes. Steps
2–6 (mask/gate/cluster/keypoints) operate on the frustum subset only and are the
per-detection cost: 57.8 ms (Case A) / 80.4 ms (Case B) at base clock, 17.1 ms
(Case A) at turbo. Caching the projection is what makes a 10 Hz single-view match
loop plausible at all.

### Target platform: Raspberry Pi 5 + Hailo-8

Not yet measured on this hardware — the benchmark script (`bench_bbox_to_tree.py`)
is plain numpy with no accelerator dependency and is directly runnable on the
target device for a real number. Ballpark reasoning in the meantime: a Cortex-A76
core at 2.4 GHz sits below this host's turbo range and has a narrower SIMD width
(NEON 128-bit vs. AVX2 256-bit), so a 5–8× slowdown relative to the *turbo*
column above is a reasonable planning estimate for numpy-heavy code — i.e., the
57.8–80.4 ms per-detection cost (with projection cached and excluded) might land
around 30–65 ms on-device, inside the 100 ms/10 Hz budget but without much margin,
particularly for scenes resembling Case B (more depth-gated points, more
clustering work). The Hailo-8 running the detector leaves the Pi 5's CPU otherwise
idle for this work, which is the main favorable factor. **This estimate should be
replaced with an on-device measurement before being treated as a result.**

## 6. Limitations and future work

- **Single view, single frame.** No multi-view triangulation or temporal fusion
  across frames; a persistent-object tracker averaging matches over a few frames
  (the target is typically static, e.g. a tree) would improve robustness for
  free, and was discussed but not implemented.
- **Hand-tuned thresholds.** Voxel size, MAD multiplier, merge radius/ratio, and
  the centrality-override thresholds were set from two real scenes, not a
  systematic sweep or a learned parameter. Generalization to substantially
  different object scales (e.g. much smaller or much larger targets than a chair
  or a tree trunk) is untested.
- **No class-specific priors.** The detector's class label (e.g. "chair" vs.
  "tree") is available but unused — expected physical scale/shape per class
  (height range, planarity) could reject an implausible cluster outright rather
  than relying only on the generic size/centrality rule.
- **No ground-plane removal.** A one-time RANSAC ground fit (mentioned in the
  original design discussion) would help specifically with bottom-point accuracy
  when a box's lower edge includes ground/floor returns; not yet integrated into
  this pipeline.
- **Depth-gate window can under-cover a tall/deep object.** In Case A, the
  extracted chair cluster spans only ~0.16 m vertically — plausibly a seat/back
  slice rather than the chair's full ~0.9 m height, if parts of the chair fall
  outside the single MAD window used in step 3. This did not affect the
  qualitative validation (correct object, correct rejection of the background),
  but would affect absolute top/bottom accuracy for objects whose own depth
  extent approaches or exceeds the gate's window.
- **Requires a solved, not seed, extrinsic.** Accuracy of every downstream number
  is gated by calibration accuracy; this method adds no independent correction.

## 7. Figures

Generated by `visualize_bbox_match.py` / `interactive_bbox_match.py`, suitable as
qualitative-result figures:

- `bbox_match_chairs_view05.png` — Case A. Gray = frustum candidates, orange =
  depth-gated but excluded (visibly includes the partition panel and rejected
  chair-frame fragments), green = final selected cluster, colored rings =
  center/bottom/top.
- `bbox_match_trunk_view00.png` — Case B, same color scheme; illustrates the
  clustering correctly retaining the full trunk cluster against a small
  distractor.
