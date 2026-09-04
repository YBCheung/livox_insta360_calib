# 2D-to-3D Object Localization from a Detection Bounding Box

Working notes on the bbox→point-cloud matching method, organized for reuse in a
paper's Method / Evaluation / Limitations sections. Numbers below are measured on
real captured data (`data/calib_indoor_level`, `data/calib_data_yard_2`), not
synthetic.

Two variants of the isolation step are documented and compared:

| | separation | implementation | offline tools | real-time tool |
|---|---|---|---|---|
| **cluster** | depth-mode gate, then 3D voxel connected components | `bbox_match.select_cluster` | `bench_bbox_to_tree.py`, `visualize_bbox_match.py`, `interactive_bbox_match.py` | `live_cluster_match.py` |
| **fast** | nearest range run only (1D) | `bbox_match.select_nearest_run` | — | `live_view_overlay.py` |

Both share one implementation of projection, unprojection and keypoint extraction
(`bbox_match.py`), so a measured difference between them is a difference in point
selection and nothing else. `live_view_overlay.py --match both` runs the two on the
same detection, in the same process, against the same frame and the same cloud —
necessary because the camera and the Hailo accelerator are each exclusive to one
process, so two programs cannot observe the same instant.

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

**Algorithm 1 — bbox → object keypoints ("cluster" variant)**

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
see Section 6). Steps 2–6 are O(n) to O(m log m) (voxel hashing + union-find with
path compression), operating only on the small frustum subset. In the measurements
below, n was 2–3 orders of magnitude smaller than N.

### 2.5 The "fast" variant: nearest range run

For a real-time loop the whole-cloud projection of step 1 is not available to
amortize — every frame carries a new cloud — and the isolation step runs inside a
per-frame budget shared with image decoding, view synthesis and the detector. The
fast variant replaces steps 3–5 with a single 1D operation on the in-box ranges.

**Algorithm 2 — bbox → object keypoints ("fast" variant)**

```
Input:  in-view projection (u, v, r) already computed for display,
        detection box (x1,y1,x2,y2), shrink factor s, gap threshold g
Output: center, bottom, top (3D points), or NONE

1.  B <- central s-fraction of the box, about its centre         # default s = 0.6
2.  C <- { i : u_i, v_i inside B }                               # O(n_view)
3.  sort r_C ascending
    split r_C wherever consecutive ranges differ by more than g  # default g = 0.5 m
4.  K_best <- the NEAREST run with |K| >= max(min_points, 0.15|C|)
5.  unproject only K_best back to the LiDAR frame                # O(|K_best|)
6.  keypoints as in Algorithm 1, step 6
```

Three properties matter for the real-time setting:

- **No whole-cloud projection.** `(u, v, r)` is already computed each frame to draw
  the overlay; the matcher consumes it. A point's `(u, v, r)` is a complete
  description of its position — the pixel fixes the bearing, the range fixes the
  distance — so 3D coordinates are recovered by unprojection for the surviving
  points only, and nothing extra is carried through the per-frame path.
- **Nearest rather than dominant.** Algorithm 1 step 3 keeps the *most populous*
  depth peak; Algorithm 2 keeps the *nearest* run with support. For a detection
  standing in front of a background, nearest is the object by construction, and
  no lateral test is needed to reject the background — provided the two are
  separated by more than `g` in range.
- **Box shrink instead of clustering.** With no lateral separation available, the
  frustum itself is narrowed (default: central 60% of the box) so that fewer
  background returns enter the candidate set at all.

### 2.6 Keypoint extraction (shared)

Both variants end at the same step, and it is deliberately common code so the
comparison isolates selection. Given the selected point set:

```
center <- median(K)                                    # per-axis, robust
top    <- mean of the points in the 10th percentile band of the anchor axis
bottom <- mean of the points in the 90th percentile band
```

The anchor axis is either world-up `z` in the LiDAR frame (the offline tools'
`up_axis`, and the right choice once views are pitched) or the image row (valid
while the panorama is gravity-locked and views are cut level, and the definition
that ties top/bottom to the box that produced them). A decile *band mean* rather
than a single extreme point (`argmin`/`argmax`) prevents one stray return from
setting the reported extent; the offline tools' 1st/99th-percentile form is the
same intent expressed as a scalar height.

## 3. Implementation notes

- Percentile (not min/max) for top/bottom: robust to the occasional stray return
  (e.g. a bird, a wire, a floor straggler) that would otherwise set the reported
  extent from a single point.
- Median (not mean) for center: same robustness argument against residual
  outliers that survive gating.
- All thresholds (voxel size 0.08 m, MAD multiplier 3.0, merge radius 0.3 m, merge
  size ratio 3x, centrality overrides at 0.5/0.3 × half-diagonal) were tuned
  against the two real scenes in Section 4, not learned or exhaustively swept —
  flagged explicitly as a limitation (Section 7).

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

## 5. Method comparison: fast vs cluster

All numbers in this section are measured on the **target platform** (Raspberry Pi 5,
Cortex-A76 @ 2.4 GHz, `time.perf_counter`, ≥20 iterations after warm-up), on
`data/calib_indoor_level` with boxes produced by the actual detector (YOLOv8s on a
Hailo-8L, see Section 6.2). Both variants consume the *same* projection arrays in
every row, so the differences are attributable to the isolation step alone.

Two cloud densities are reported because they differ by two orders of magnitude and
the methods separate differently at each:

- **live window** — 0.4 s of accumulated MID-360 scans (~29k points, ~6.2k in view,
  827 in the box), i.e. what a real-time loop actually holds;
- **dense capture** — the 1.16M-point integrated cloud the offline tools use
  (~62k in view, ~8.3k in the box).

### 5.1 Runtime

Per detection, single box, live window:

| stage | fast | cluster |
|---|---|---|
| box mask over in-view points | 0.02 ms | 0.02 ms |
| unproject to LiDAR frame | inliers only, included below | 0.14 ms (whole candidate set) |
| depth-mode gate | — | 0.20 ms |
| 3D voxel connected components | — | 0.82 ms |
| range sort + run split, selection, keypoints | remainder | remainder |
| **total** | **0.79 ms** | **2.34 ms** |

Dense capture, same box: fast **6.87 ms**, cluster **7.27 ms** — the gap closes
because at that density both are dominated by the shared per-candidate work rather
than by the separation logic.

The cluster variant as re-implemented in `bbox_match.py` produces **selections
identical to the original** — 29 vs 29 points at the live window, 4,434 vs 4,434 on
the dense capture, same centres — while running faster in proportion to the
candidate count. Timing the gate, components and selection on an *identical*
candidate set with no box shrink on either side:

| | candidates in box | original | ported | speedup |
|---|---|---|---|---|
| live window | 827 | 1.93 ms | 1.79 ms | 1.1× |
| dense capture | 33,493 | 13.43 ms | 4.13 ms | 3.3× |

The gain scales with the candidate count because what changed is the part that
scales — interpreted per-voxel work becomes array work — while the fixed overheads
are unchanged. Three changes, none altering the algorithm:

1. voxel neighbours found by `searchsorted` over a monotone integer encoding of the
   occupied voxels, instead of a Python dict lookup per voxel per offset — only the
   union-find over discovered pairs remains interpreted;
2. the depth-mode histogram computed with `bincount` over scaled indices instead of
   `np.histogram` (verified to produce **identical gate masks on 180/180 real
   boxes**);
3. cluster centrality measured on the pixel coordinates already in hand rather than
   reprojecting each candidate centroid through `R p + t`.

### 5.2 Agreement

Same frame, same cloud, same extrinsic, three live detections (live window):

| detection | fast centre (m) | cluster centre (m) | points kept | Δ centre |
|---|---|---|---|---|
| chair, conf 0.91 | (2.54, 2.25, −0.09) | (2.36, 2.20, −0.11) | 122 → 59 | **0.18 m** |
| chair, conf 0.40 | (2.92, 1.44, 0.00) | (2.49, 1.35, 0.13) | 317 → 92 | **0.46 m** |
| monitor, conf 0.74 | (2.98, 2.34, 0.70) | (2.99, 2.29, 0.68) | 20 → 15 | **0.06 m** |

The pattern is consistent: where the object stands clear of its background in depth
the two agree to a few centimetres, and the disagreement grows with how loose the
box is. The cluster variant keeps roughly a third as many points — the difference
being background it rejects and the fast variant absorbs.

On the dense capture the same effect is starker: fast retains 16,582 points and
places the chair's centre at (2.99, 2.15, −0.08), against 3,485 points at
(2.56, 2.13, −0.06) for cluster — the fast variant has swallowed the fabric
partition roughly 0.5 m behind the chair, which lies within the 0.5 m gap threshold
and therefore cannot be separated along the range axis at all. This is Case A of
Section 4 seen from the other side: it is precisely the failure the 3D connected
components exist to prevent.

*Caveat on absolute coordinates:* these were produced with a **seed** extrinsic, so
the absolute XYZ values are not calibration-accurate in the sense of Section 4 (which
used solved extrinsics). The Δ column is unaffected — both variants consume the same
projection, so an extrinsic error is common-mode and cancels in the difference.

### 5.3 Ablation: the gate and the components are not separable

Feeding in-box candidates directly to the connected-component stage, skipping the
depth-mode gate, fails in both directions:

| | dense capture | live window |
|---|---|---|
| candidates in box | 33,493 | 827 |
| selected without the gate | 31,547 (94%, one merged blob) | 36 points at 3.85 m (wrong surface) |
| selected with the gate | 3,485 | 59 |

At 8 cm voxel connectivity the floor chains object to background into a single
component, so without the depth gate the "largest cluster" is the whole box; when
that chaining happens not to occur, the size and centrality rules then arbitrate
among fragments of the wrong surface. The gate is a precondition for the components
to mean anything, not a pre-filter that merely saves work.

### 5.4 When each is preferable

- **fast** — real-time loops where the object is separated from its background in
  range by more than the gap threshold, and where the per-detection budget is shared
  with a detector and a video path. Roughly 3× cheaper, and its failure mode is
  benign and predictable: it reports a centre biased *behind* the object, never a
  centre on a different object.
- **cluster** — when the background sits at a similar range to the target (indoor
  scenes with walls and partitions, an object against a facade), when boxes are
  loose, or offline where the extra ~1.5 ms per detection is irrelevant. Its failure
  mode is less benign: mis-selection returns a confidently wrong object, which is
  why the two ablation cases of Section 4 are retained as a regression pair.

Both cost far less than the detector that produces their input (≈20 ms per view,
Section 6.2), so on this platform the choice is governed by scene geometry rather
than by compute.

## 6. Runtime performance

### 6.1 Offline, static cloud (development host)

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

### 6.2 Target platform, measured: Raspberry Pi 5 + Hailo-8L

Hardware: Raspberry Pi 5 (4× Cortex-A76 @ 2.4 GHz), Insta360 ONE X5 delivering a
2880×1440 equirectangular MJPEG stream at 30 fps over UVC, Livox MID-360 at 10 Hz
(~20k points per message, vertical field of view −7°…+52°), Hailo-8L M.2
accelerator. CPU figures are `utime+stime` sampled from `/proc/<pid>/stat` over a
10 s steady-state window and expressed as a fraction of one core; stage figures are
`time.perf_counter` means after warm-up.

**Per-frame front-end** (the costs the matcher shares its budget with, two 480 px
views at 10 fps):

| stage | cost | note |
|---|---|---|
| MJPEG decode, 2880×1440 | 23 ms CPU/frame | single-threaded, irreducible; bounded by capping the processed frame rate, since the camera offers 30 fps and no other 2:1 mode |
| view synthesis (gnomonic remap) | 4.4 ms CPU/view | bilinear with fixed-point (`CV_16SC2`) maps; 8.3 ms bicubic |
| — all views in one fused remap | 11.3 ms vs 14.6 ms | one call for the grid vs one per view (2 views) |
| LiDAR ingest, `CustomMsg` via rclpy | 134 ms deserialize + 27 ms callback | ~1.6 core-seconds per second at 10 Hz |
| LiDAR ingest, raw CDR parsed with numpy | 1.6 ms/message | ~100× less; layout verified field-by-field against rclpy's own deserialization |
| projection of the accumulated window | 5.92 ms | culling in the LiDAR frame before rotating; 9.40 ms projecting first, identical output to 1.2·10⁻⁴ px |
| detector, YOLOv8s @640 on Hailo-8L | 20.3 ms/view (49.3 fps) | 56.9 fps hardware-only; 1.5 ms of that is resize + colour conversion |
| **match, per detection** | **0.79 ms (fast) / 2.34 ms (cluster)** | Section 5.1 |

**Whole-loop CPU**, two views, 10 fps, live camera:

| configuration | CPU |
|---|---|
| overlay only, no detector | 31.4% |
| detector on one view | 39.4% |
| detector on both views | 48.3% |
| `live_cluster_match.py`, detector + cluster match on both views | 50.9% |

**Reconciliation with the earlier estimate.** A previous revision of these notes
projected 30–65 ms per detection on this hardware, extrapolating from the host
column of Section 6.1 by clock and SIMD width. The measured cost is 0.8–2.3 ms —
one to two orders of magnitude lower — and the extrapolation was not what was wrong.
The dominant term is candidate-set size, not the CPU: Section 6.1 matches against a
15 s integrated 1.17M-point capture (≈21k points in the frustum), whereas a
real-time loop matches against a 0.4 s accumulation window (≈29k points total, ≈830
in the box). Run on-device against the dense capture, the per-detection cost is
7.3 ms — below the estimate but the same order, and the original implementation's
gate-plus-components stage is 3.3× slower than the ported one at that candidate
count (Section 5.1), which accounts for part of the remaining gap. The lesson for
the paper's evaluation section: for this pipeline, report the cost against the cloud
a deployment actually holds, since the isolation stages scale with the frustum
subset and not with the capture.

**Projection is not amortizable in the real-time case.** Section 6.1 notes that step
1 need not be recomputed per detection when the cloud is static — the offline tools
exploit exactly this, projecting once at startup so each interactive box costs only
steps 2–6. Live, every frame carries a new cloud, so that cache does not exist. The
real-time path instead (a) culls to each view's cone before rotating, and (b) reuses
the projection already computed to render the overlay, so the matcher adds no
projection cost of its own.

## 7. Limitations and future work

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
- **The fast variant separates in one dimension only.** It cannot distinguish an
  object from a background surface within `g` (default 0.5 m) of it in range, and
  it is symmetrically vulnerable to *foreground* clutter: a railing, branch or
  doorframe crossing the box is the nearest run and will be reported as the object.
  Shrinking the box narrows both exposures but does not remove either.
- **Temporal skew between the box and the points.** The box comes from a single
  camera frame; the points come from a 0.4 s accumulation window with no motion
  compensation, because the MID-360's non-repetitive scan is too sparse in one
  message to isolate an object by eye or by clustering. For a walking person that
  window is a smear roughly 1 m long, and both variants will mix positions along it.
  A shorter window trades density for currency; per-scan deskewing (or consuming an
  already-deskewed cloud such as FAST-LIO's `/cloud_registered_body`, with the known
  LiDAR→IMU offset removed) is the principled fix and is not implemented.
- **Vertical coverage limits the bottom keypoint.** The MID-360 spans −7°…+52°
  vertically, so for a view cut at pitch 0 the returns lie mostly above the horizon.
  The bottom anchor is therefore the least trustworthy of the three, and a target's
  lower extent (a person's legs, a trunk's base) may have no returns at all —
  independent of which isolation variant is used.
- **Parallax between the sensors.** With the camera origin ~0.22 m from the LiDAR,
  some in-box returns are surfaces the camera cannot see, and no image-space test
  can identify them.
- **Comparing variants requires one process.** The camera device and the Hailo
  accelerator are each exclusive to a single process, so two programs cannot observe
  the same instant; any A/B measurement must run both matchers inside one loop on
  one frame (`live_view_overlay.py --match both`). This is a constraint on the
  evaluation design, not on deployment.
- **Requires a solved, not seed, extrinsic.** Accuracy of every downstream number
  is gated by calibration accuracy; this method adds no independent correction.

## 8. Figures

Generated by `visualize_bbox_match.py` / `interactive_bbox_match.py`, suitable as
qualitative-result figures:

- `bbox_match_chairs_view05.png` — Case A. Gray = frustum candidates, orange =
  depth-gated but excluded (visibly includes the partition panel and rejected
  chair-frame fragments), green = final selected cluster, colored rings =
  center/bottom/top.
- `bbox_match_trunk_view00.png` — Case B, same color scheme; illustrates the
  clustering correctly retaining the full trunk cluster against a small
  distractor.

Generated live by `live_cluster_match.py` (same stage colours, driven by detector
boxes instead of drawn ones) and by `live_view_overlay.py --match both`:

- a live staged figure showing, for a real detection, the frustum candidates, what
  the depth gate removed, the selected component and the reprojected C/B/T markers —
  the real-time counterpart of the two offline figures above;
- an A/B figure with both variants' anchors on one detection (white = fast,
  cyan = cluster), which is the figure that makes Section 5.2's disagreement
  column visual rather than tabular.
