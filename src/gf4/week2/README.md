# GF4 Structure from Motion - Week 2 Guide
## Pairwise Feature Matching and Epipolar Geometry

**Goal:** Build a reusable pairwise image-matching and epipolar-geometry pipeline that will become part of your own Structure from Motion system.

**Main deliverable:** A Python pipeline that detects features, matches image pairs, estimates fundamental matrices, visualises epipolar geometry, builds a match graph over a small image set, and suggests candidate image pairs for reconstruction.

In Week 1, you used COLMAP as a professional reference system. In Week 2, you will begin opening the black box. You will build a working pipeline using OpenCV for SIFT feature detection, descriptor matching, and robust fundamental matrix estimation.

---

## Week 2 learning objectives

By the end of Week 2, you should be able to:

1. Detect local image features and compute descriptors using OpenCV.
2. Match descriptors between image pairs and filter matches using Lowe's ratio test.
3. Estimate the fundamental matrix using RANSAC.
4. Interpret RANSAC inliers, inlier ratios, and epipolar errors.
5. Visualise keypoints, feature matches, and epipolar lines.
6. Compare good and bad image pairs using quantitative evidence.
7. Build a pairwise match graph over a small image set.
8. Suggest candidate image pairs for Week 3 triangulation and camera-pose recovery.

This week should produce reusable code. Week 3 will use your Week 2 matches and candidate pair choices to initialise a sparse 3D reconstruction.

---

## Required reading

Read the following:

1. Scale-Invariant Feature Transform

   https://en.wikipedia.org/wiki/Scale-invariant_feature_transform

2. OpenCV: Introduction to SIFT

   https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

3. OpenCV: Feature Matching

   https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

4. Random Sample Consensus (RANSAC)

   https://en.wikipedia.org/wiki/Random_sample_consensus

5. OpenCV: Epipolar Geometry

   https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

For deeper background on the fundamental matrix, use:

Richard Hartley and Andrew Zisserman,

*Multiple View Geometry in Computer Vision*, Chapter 9:

**"Epipolar Geometry and the Fundamental Matrix"**

https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf


---

## Software

You should continue using your Week 1 Python environment.

Required Python packages:

- opencv-python>=4.4
- numpy
- matplotlib

You must use SIFT. Do not use ORB, SuperPoint, or any other feature detector for the required Week 2 results. Check that SIFT is available:

```bash
python -c "import cv2; assert hasattr(cv2, 'SIFT_create'); print('SIFT available')"
```

---

## Starter code

Starter code is provided:

```text
week2/week2_pipeline.py
week2/sfm_utils.py
```

The starter code handles:

- command-line argument parsing,
- pair mode and image-set mode,
- image loading and resizing,
- CSV saving for metrics,
- basic keypoint and match drawing helpers,
- match-graph drawing.

You have to complete the TODO functions in `sfm_utils.py`.

---

## Suggested command-line interface

Your Week 2 pipeline should be runnable from the command line.

For pair-only analysis, use:

```bash
python week2/week2_pipeline.py \
  --image1 path/to/image_01.jpg \
  --image2 path/to/image_08.jpg \
  --output-dir week2/output/pair_debug \
  --max-features 4000 \
  --ratio 0.75
```

You do not need to use this exact interface, but your code should be reproducible and not depend on manual editing for each experiment.

---

## Useful OpenCV functions

You are expected to write the pipeline logic yourself, but the following OpenCV functions and object fields are the main API calls you will need.

For SIFT feature detection:

- `cv2.cvtColor`
- `cv2.SIFT_create`
- `sift.detectAndCompute`

For descriptor matching:

- `cv2.BFMatcher`
- `matcher.match`
- `matcher.knnMatch`
- `cv2.NORM_L2`

For drawing and visual inspection:

- `cv2.drawKeypoints`
- `cv2.drawMatches`

For fundamental matrix estimation:

- `cv2.findFundamentalMat`
- `cv2.FM_RANSAC`

Important OpenCV objects:

- `cv2.KeyPoint.pt`: keypoint location as `(x, y)`
- `cv2.DMatch.queryIdx`: index into image 1 keypoints/descriptors
- `cv2.DMatch.trainIdx`: index into image 2 keypoints/descriptors
- `cv2.DMatch.distance`: descriptor distance

Use the OpenCV documentation to check exact arguments and return values. Your implementation should show that you understand how keypoints, descriptors, matches, and point coordinates flow into fundamental matrix estimation.

---

## AI use policy

You may use AI tools to help you learn the background material, understand the concepts, and interpret documentation. For example, you may use AI to ask:

- what SIFT keypoints and descriptors represent,
- how Lowe's ratio test works,
- what OpenCV functions such as `cv2.SIFT_create`, `BFMatcher.knnMatch`, or `cv2.findFundamentalMat` return.

You may **not** use AI tools to write code for this assignment, edit code, fix code, debug code, generate tests, or diagnose implementation errors. The implementation you submit must be written and debugged by your group.

Any use of AI tools must be clearly mentioned in your experiment log.

---

## Phase A - Feature detection and matching

Start with two images from your best Week 1 capture. Choose a pair with good overlap and visible camera motion.

### Detect local features

Implement or complete the SIFT feature detector function:

```python
def detect_sift_features(image, max_features=4000):
    ...
```

Settings:

```text
Maximum features: 4000 per image
```

Create a keypoint visualisation so you can inspect where SIFT responds strongly or weakly.

### Match descriptors

Implement descriptor matching between two images.

Required filtering:

1. Lowe's ratio test.

Start with Lowe ratio threshold:

```text
0.75
```

Create a representative match visualisation so you can inspect the retained matches.

Also save a raw nearest-neighbour match visualisation before Lowe filtering. This is useful for seeing what the ratio test removes.

---

## Phase B - Fundamental matrix estimation

Use the filtered matches from Phase A to estimate the fundamental matrix.

You may use OpenCV:

```python
F, inlier_mask = cv2.findFundamentalMat(
    pts1,
    pts2,
    method=cv2.FM_RANSAC,
    ransacReprojThreshold=1.0,
    confidence=0.99,
)
```

Here, `ransacReprojThreshold=1.0` means a match is treated as geometrically consistent if its epipolar error is within about 1 pixel. The `confidence=0.99` value is the target probability that RANSAC has sampled at least one all-inlier minimal set; increasing it can require more iterations.

The **inlier ratio** is:

```text
number of RANSAC inliers / number of Lowe-filtered matches
```

### Epipolar error

Implement your own epipolar error calculation. For a point \(x_1\) in image 1, the corresponding epipolar line in image 2 is:

```text
l_2 = F x_1
```

The point-to-line distance for the corresponding point \(x_2 = [u, v, 1]^T\) is:

```text
abs(a*u + b*v + c) / sqrt(a^2 + b^2)
```

where \(l_2 = [a, b, c]^T\).

Compute epipolar errors for all Lowe-filtered matches, then compare them with the errors for the RANSAC inlier subset.

### Epipolar line visualisation

Select 10-20 RANSAC inlier matches.

Create a figure showing:

1. Points in image 1.
2. Corresponding epipolar lines in image 2.
3. Corresponding points in image 2.

The epipolar lines should pass close to the corresponding points.

---

## Phase C - Good pair versus bad pair

Repeat Phases A and B for one challenging pair from your Week 1 captures.

Choose a pair that is expected to be difficult. Use the same pipeline on both pairs so that the comparison is fair.

---

## Phase D - Sensitivity analysis

In this phase, perform **two** one-variable sensitivity analyses in pair mode only. Use pair-mode commands rather than directory mode, and do not run a large grid search.

First, compare two Lowe ratio thresholds. Second, choose one additional parameter:

```text
maximum number of SIFT features
RANSAC reprojection threshold
```

For each sensitivity analysis, compare the default setting against one alternative on one good pair and one bad pair.

Examples:

```text
required: ratio 0.75 versus 0.90 on a challenging pair
or required: ratio 0.60 versus 0.90 on a difficult repeated-texture pair
second analysis: max features 4000 versus 2000
or second analysis: RANSAC threshold 1.0 px versus 2.0 px
```

Avoid comparisons where the two settings are so close that the result is almost unchanged. For example, if `0.60` and `0.75` give almost the same Lowe-ratio result, use a more challenging pair or choose a different parameter.

Use only the quantities needed to support your conclusion.

---

## Phase E - Pairwise matching over an image set

Now move from single image pairs to a small image set.

Use ~20 images from your best capture. If your images are very large, resize them for this week so that experiments run quickly.

For every image pair, compute pairwise matching and epipolar-geometry metrics.

For image-set analysis, run the pipeline in directory mode:

```bash
python week2/week2_pipeline.py \
  --image-dir week2/data/good_subset/images \
  --output-dir week2/output/good_subset \
  --max-images 20 \
  --max-features 4000 \
  --ratio 0.75
```

### Match graph

Create a graph where:

- each node is an image,
- each edge is an image pair,
- edge weight is the number of RANSAC inliers or the inlier ratio.

The match-graph visualisation function is already provided in the starter code. Your task is to produce the pairwise metrics that feed the graph and to interpret what the graph shows. The graph should help you identify which images overlap and which image pairs are reliable.

### Suggesting candidate pairs for Week 3

Use your pairwise metrics to suggest 3 candidate initial pairs for Week 3. These are hypotheses that you will test next week by recovering pose and triangulating points.

A good initial pair usually has:

- many RANSAC inliers,
- a high inlier ratio,
- visible viewpoint change,
- enough parallax for triangulation.

Do not simply choose the pair with the highest number of inliers. Adjacent images with tiny baseline may match extremely well but triangulate poorly in Week 3.

---

## Intermediate report

Submit an individual report, max **4 pages**.

Practical work is performed in groups of 3, but each student must submit their own report with their own interpretation and analysis.

Pages 1-3 should contain your technical results and discussion. Page 4 must be an **experiment log** describing how your implementation developed.

1. **Brief theory answers**
   - What is the epipolar constraint?
   - What does the fundamental matrix represent?
   - Why is RANSAC needed for estimating the fundamental matrix?
   - Why might a pair of images with many matches still be bad for 3D reconstruction?

Sections 2 and 3 should use your good pair of images.

2. **Feature and matching results**
   - keypoint visualisation for your good pair,
   - short explanation of where features are dense or sparse and why,
   - number of raw matches and filtered matches,
   - visualisation of raw nearest-neighbour matches and filtered Lowe-ratio matches,
   - a qualitative comment on likely false matches for the ratio 0.75 result.

3. **Fundamental matrix and epipolar geometry**
   - RANSAC inlier visualisation,
   - epipolar-line visualisation,
   - number of RANSAC inliers,
   - inlier ratio,
   - median epipolar error for RANSAC inliers,
   - median epipolar error for all filtered matches before applying the RANSAC inlier mask.

4. **Good pair versus bad pair analysis**
   - complete the good/bad comparison table below,

   | Metric | Good Pair | Bad Pair |
   |---|---:|---:|
   | Number of keypoints in image 1 | | |
   | Number of keypoints in image 2 | | |
   | Raw matches | | |
   | Filtered matches at ratio 0.75 | | |
   | RANSAC inliers | | |
   | Inlier ratio | | |
   | Median epipolar error for all filtered matches (px) | | |
   | Median epipolar error for RANSAC inliers (px) | | |
   | Main observed failure | | |

   Answer:
   - Why does the bad pair not achieve good results?

5. **Targeted sensitivity analysis**
   - complete two one-variable sensitivity analyses using pair mode only,
   - the first must compare two Lowe ratio thresholds,
   - the second must compare either maximum number of SIFT features or RANSAC reprojection threshold,
   - do not use directory mode for alternative settings,
   - state which setting you will use in Week 3 and why.

   You can use figures or numbers to support your answer.

6. **Pairwise image-set analysis**
   - include the pairwise match graph,
   - use `pairwise_metrics.csv` as supporting evidence,
   - report the top 3 candidate initial pairs for Week 3,
   - explain why you suggested them.

   | Candidate Pair | RANSAC Inliers | Inlier Ratio | Median Epipolar Error |
   |---|---:|---:|---:|
   | | | | |
   | | | | |
   | | | | |

7. **Group dynamics**
   - briefly describe the contribution of each member in your group (max 2-3 sentences per person).

### Experiment log

Page 4 of your report must be an experiment log. It should be a concise record of what you tried, what changed, and what you learned while developing the pipeline.

Include:

1. **Implementation steps**
   - Which parts of the pipeline did you implement first?
   - Which functions or scripts did each group member contribute to?

2. **Debugging record**
   - What went wrong during implementation?
   - How did you diagnose the problem?
   - What evidence showed that your fix worked?

3. **Experiment choices**
   - Which image pairs or subsets did you try?
   - Why did you choose the final good and bad pairs?
   - Did any result surprise you?

4. **AI and external assistance**
   - You must state clearly what AI assistance you used for learning concepts or understanding documentation.
   - Do not use AI tools to write, edit, test, or debug your code.
   - You are responsible for understanding every part of the code you submit.

---

## Code walkthrough requirement

Each group must be prepared for a short code walkthrough session.

All group members should understand the submitted code and results. The walkthrough is part of the assessment of whether the pipeline has been understood, validated, and used thoughtfully.

---

## Deliverables checklist

By the end of Week 2, submit:

- group Interim Code submission: `week2_pipeline.py` and `sfm_utils.py`
- individual Interim Report 2 PDF

Your code should be clean enough that it can be reused directly in Week 3.

---

## Looking ahead to Week 3

Week 3 will use your suggested image pairs to recover relative camera pose, triangulate sparse 3D points, and register additional camera views.

Your most important Week 2 outputs for Week 3 are:

- reliable feature matches,
- RANSAC inlier correspondences,
- suggested initial image pairs,
- evidence about which matching settings work best for your scene.
