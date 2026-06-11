from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PairCandidate:
    """A dataclass to manage two-view geometry candidate scores for initial pair selection."""

    image_i: str
    image_j: str
    ransac_inliers: int      # from the Week 2 CSV (fundamental-matrix RANSAC)
    pose_inliers: int            # cheirality-valid inliers after pose recovery
    triangulated_points: int     # finite points used for the angle estimate
    median_tri_angle: float      # degrees -- the baseline / parallax measure

    def as_dict(self) -> dict:
        return {
            "image_i": self.image_i,
            "image_j": self.image_j,
            "ransac_inliers": self.ransac_inliers,
            "pose_inliers": self.pose_inliers,
            "triangulated_points": self.triangulated_points,
            "median_tri_angle_deg": round(self.median_tri_angle, 4),
        }


def median_triangulation_angle(
    points3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> float:
    """Median parallax angle at the triangulated points. 
    We look at the angle between the projected rays from the two cameras to each point using the recovered relative pose (R, t), 
    allowing us to assess the quality of the triangulation.
    """
    points3d = np.asarray(points3d, dtype=np.float64)
    if points3d.ndim != 2 or points3d.shape[1] != 3 or len(points3d) == 0:
        return 0.0

    finite = np.isfinite(points3d).all(axis=1)
    X = points3d[finite] # Only consider points with finite coordinates for angle calculation
    if len(X) == 0:
        return 0.0

    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    c2 = (-R.T @ t).ravel()

    ray1 = -X            # origin - X
    ray2 = c2 - X
    n1 = np.linalg.norm(ray1, axis=1)
    n2 = np.linalg.norm(ray2, axis=1)
    valid = (n1 > 0) & (n2 > 0) # # Avoid division by zero
    if not np.any(valid):
        return 0.0

    cos_angle = np.sum(ray1[valid] * ray2[valid], axis=1) / (n1[valid] * n2[valid])
    angles = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))  # Convert to degrees and clip to valid range for arccos
    return float(np.median(angles))


def choose_initial_pair(
    candidates: list[PairCandidate],
    min_tri_angle: float = 5.0,
) -> PairCandidate:
    """Pick the best initial pair from evaluated candidates.

    Prefer pairs whose median triangulation angle clears `angle_check` (enough
    baseline for stable triangulation); among those, take the one with the most
    pose inliers. If none clear the gate, fall back to the best by pose inliers
    so the caller is never left without an initial pair.
    """
    if not candidates:
        raise ValueError("No candidates to choose from")

    passing = [c for c in candidates if c.median_tri_angle >= min_tri_angle]
    pool = passing if passing else candidates # If no candidates clear the angle gate, consider them all (fallback)
    return max(pool, key=lambda c: c.pose_inliers)


def _triangulation_angles(X, Rr, tr, Ru, tu):
    """Parallax angle (deg) at each point between cameras r and u, allowing us to assess the quality of the
    triangulation. Low angles are unstable and often wrong, so we filter them out in `triangulate_new_points`
    This is similar in spirit to `median_triangulation_angle`, but we want the angle at each point, not the median,
    and we have two arbitrary registered cameras (Rr,tr and Ru,tu) instead of one at the origin."""
    Cr = (-Rr.T @ tr.reshape(3, 1)).ravel()
    Cu = (-Ru.T @ tu.reshape(3, 1)).ravel()
    v1, v2 = X - Cr, X - Cu
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.einsum("ij,ij->i", v1, v2) / np.maximum(n1 * n2, 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def triangulate_general(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Generalisation of Week 3's triangulate_points for arbitrary registered cameras, by using two projection matrices instead of assuming the first camera is at the origin. 
    The projection matrices P1 and P2 are constructed using the camera intrinsics K and the extrinsic parameters (R, t) for each camera. 
    The OpenCV function cv2.triangulatePoints is then used to triangulate the 3D points from the corresponding 2D points in each image.
    """
    K = np.asarray(K, dtype=np.float64)
    P1 = K @ np.hstack((np.asarray(R1, dtype=np.float64), np.asarray(t1, dtype=np.float64).reshape(3, 1)))
    P2 = K @ np.hstack((np.asarray(R2, dtype=np.float64), np.asarray(t2, dtype=np.float64).reshape(3, 1)))

    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts1) == 0:
        return np.empty((0, 3), dtype=np.float64)

    points4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        points3d = (points4d[:3] / points4d[3]).T
    return points3d.astype(np.float64)


def _depths_in_camera(points3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Z-coordinate of each point in the camera frame X_cam = R X + t."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    cam = R @ points3d.T + t
    return cam[2]


@dataclass
class Track:
    """ A dataclass that stores the 3D point's ID, its 3D coordinates (xyz), its color, and a dictionary of observations that maps image IDs to keypoint indices. 
    This structure allows us to keep track of each 3D point and its corresponding 2D observations across multiple images in the incremental SfM pipeline, which  """

    point_id: int
    xyz: np.ndarray                       # (3,) 3D coordinates of the point in the reconstruction frame
    colour: np.ndarray                    # (3,) uint8
    observations: dict[int, int] = field(default_factory=dict)  # creates a dictionary mapping of image_ids to the respective keypoint_indices which was used to triangulate the point


@dataclass
class RegisteredCamera:
    """A camera that has been placed in the reconstruction frame and used to triangulate points.
    Stores the camera's ID, its rotation and translation relative to the world frame, and a mapping 
    from keypoint indices to 3D point IDs for the keypoints that have been linked to the map."""

    image_id: int
    R: np.ndarray                        
    t: np.ndarray                       
    kpidx_to_point: dict[int, int] = field(default_factory=dict)  # similar to Track, but for the registered camera, it maps keypoint indices to the corresponding 3D point IDs in the reconstruction. 


@dataclass
class StepMetrics:
    """One incremental registration step (report evidence)."""

    step: int
    image: str
    n_2d3d: int
    pnp_inliers: int
    new_points: int
    n_registered: int
    n_points: int
    retriangulated: int = 0          # points updated by the periodic retriangulation pass

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "image": self.image,
            "n_2d3d": self.n_2d3d,
            "pnp_inliers": self.pnp_inliers,
            "new_points": self.new_points,
            "n_registered": self.n_registered,
            "n_points": self.n_points,
            "retriangulated": self.retriangulated,
        }


class IncrementalReconstruction:
    """incremental SfM over a pool of images."""

    def __init__(self, K, features: dict, pair_matches: dict, week3):
        self.K = np.asarray(K, dtype=np.float64)
        self.features = features
        self.pair_matches = pair_matches # An extension of cv2.DMatch, except but stored as a dictionary mapping pairs of image IDs (i, j) to lists of matched keypoint index pairs (keypoint_idx_i, keypoint_idx_j). 
        self.week3 = week3 # injected Week 3 geometry module

        self.registered: dict[int, RegisteredCamera] = {} # Stores all registered cameras, mapping image IDs to their corresponding RegisteredCamera instances
        self.tracks: dict[int, Track] = {} # Stores all reconstructed 3D points (Tracks), mapping point IDs to their corresponding Track instances, used for managing the 3D points and their observations across the registered cameras.
        self.next_point_id = 0
        self._last_pnp_inliers = 0
        # Reprojection errors either side of the final retriangulation pass
        # Allowsthe driver to report the improvement (set at the end of run()).
        self.errors_before_retri = np.empty(0)
        self.errors_after_retri = np.empty(0)
        self.final_retri_updated = 0

    def matches_between(self, a: int, b: int) -> list[tuple[int, int]]:
        """Similar to cv2.DMatch,  Matches oriented as (keypoint_idx_a, keypoint_idx_b)."""
        if a == b:
            return []
        if a < b: # Using the convention that matches are sorted, we can prevent duplicate storage and ensure consistent access by always looking up the pair in a specific order (a, b) where a < b.
            return self.pair_matches.get((a, b), [])
        return [(q, p) for (p, q) in self.pair_matches.get((b, a), [])]

    def _pts_from_pairs(self, image1_id: int, image2_id: int, pairs: list[tuple[int, int]]):
        """ Utilitly function that finds the 2D coordinates of the matched keypoints in both images given their IDs and the list of matched keypoint index pairs. 
        It retrieves the keypoints for each image from the precomputed features, extracts the coordinates of the matched keypoints based on the provided pairs"""
        keypoint_a = self.features[image1_id].keypoints
        keypoint_b = self.features[image2_id].keypoints
        pa = np.array([keypoint_a[i].pt for (i, _) in pairs], dtype=np.float64)
        pb = np.array([keypoint_b[j].pt for (_, j) in pairs], dtype=np.float64)
        return pa, pb

    def _add_track(self, xyz: np.ndarray, colour: np.ndarray, observations: dict[int, int]) -> int:
        """Add a new 3D point to the reconstruction."""
        pid = self.next_point_id
        self.next_point_id += 1
        self.tracks[pid] = Track(
            point_id=pid,
            xyz=np.asarray(xyz, dtype=np.float64),
            colour=np.asarray(colour),
            observations=dict(observations),
        )
        for img_id, kp in observations.items():
            self.registered[img_id].kpidx_to_point[kp] = pid
        return pid

    def two_view_geometry(
        self,
        i: int,
        j: int,
        ransac_threshold: float = 1.0,
        confidence: float = 0.999,
    ) -> tuple[int, int, float] | None:
        """Score a candidate pair: (pose_inliers, finite_points, tri_angle), inspired from week2's function but using the
        recovered pose to count inliers and triangulate points for the angle estimate.
        Does not modify reconstruction state; used for initial-pair selection.
        """
        pairs = self.matches_between(i, j)
        if len(pairs) < 8:
            return None
        pts_i, pts_j = self._pts_from_pairs(i, j, pairs)
        try:
            E, emask = self.week3.estimate_essential_matrix(
                pts_i, pts_j, self.K, threshold=ransac_threshold, confidence=confidence
            )
            R, t, pose_mask = self.week3.recover_relative_pose(
                E, pts_i, pts_j, self.K, inlier_mask=emask
            )
        except Exception:
            return None
        X = self.week3.triangulate_points(pts_i[pose_mask], pts_j[pose_mask], self.K, R, t)
        angle = median_triangulation_angle(X, R, t)
        return int(np.sum(pose_mask)), int(np.isfinite(X).all(axis=1).sum()), angle

    def select_initial_pair(
        self,
        rows: list[dict],
        *,
        top_k: int,
        min_inliers: int,
        min_tri_angle: float,
        ransac_threshold: float = 1.0,
        confidence: float = 0.999,
    ) -> tuple[int, int, list["PairCandidate"]]:
        """Pick the initial seed pair from parsed Week-2 metric rows. Converts the row data from the csv into a list of PairCandidate objects, 
        which are then scored using the two_view_geometry function to evaluate their suitability as the initial pair for reconstruction. 
        The candidates are filtered based on the number of RANSAC inliers and the median triangulation angle, and the best candidate is selected using the choose_initial_pair function.
        The function returns the image IDs of the selected seed pair and the list of evaluated candidates for reporting purposes.
        """
        name_to_id = {f.path.name: idx for idx, f in self.features.items()}
        shortlist = sorted(
            (r for r in rows if int(r["ransac_inliers"]) >= min_inliers), # Reduces search space by filtering out pairs that don't meet the minimum inlier threshold
            key=lambda r: int(r["ransac_inliers"]),
            reverse=True,
        )[:top_k]

        candidates: list[PairCandidate] = []
        for row in shortlist:
            i = name_to_id.get(row["image_i"])
            j = name_to_id.get(row["image_j"])
            if i is None or j is None:
                continue
            geom = self.two_view_geometry(
                i, j, ransac_threshold=ransac_threshold, confidence=confidence
            )
            if geom is None:
                continue
            pose_inliers, n_pts, angle = geom
            candidates.append(
                PairCandidate(
                    image_i=row["image_i"],
                    image_j=row["image_j"],
                    ransac_inliers=int(row["ransac_inliers"]),
                    pose_inliers=pose_inliers,
                    triangulated_points=n_pts,
                    median_tri_angle=angle,
                )
            )

        if not candidates:
            raise ValueError("No usable initial-pair candidates from the CSV / pool")

        chosen = choose_initial_pair(candidates, min_tri_angle=min_tri_angle)
        return name_to_id[chosen.image_i], name_to_id[chosen.image_j], candidates

    def initialise_two_view(
        self,
        i: int,
        j: int,
        *,
        ransac_threshold: float = 1.0,
        confidence: float = 0.999,
        max_reproj: float = 4.0,
    ) -> int:
        """Initialise the reconstruction from a pair of images. Returns number of points in the initial map."""
        pairs = self.matches_between(i, j)
        if len(pairs) < 8:
            raise ValueError(f"Seed pair has too few matches, needs at least 8 for RANSAC: {len(pairs)}")

        pts_i, pts_j = self._pts_from_pairs(i, j, pairs)
        E, emask = self.week3.estimate_essential_matrix(
            pts_i, pts_j, self.K, threshold=ransac_threshold, confidence=confidence
        )
        R, t, pose_mask = self.week3.recover_relative_pose(
            E, pts_i, pts_j, self.K, inlier_mask=emask
        )

        eye = np.eye(3)
        zero = np.zeros((3, 1))
        self.registered[i] = RegisteredCamera(i, eye, zero) # Registers first image as original camera at the world origin with identity rotation and zero translation.
        self.registered[j] = RegisteredCamera(j, np.asarray(R, dtype=np.float64),
                                             np.asarray(t, dtype=np.float64).reshape(3, 1))

        pts_i_posed = pts_i[pose_mask]
        pts_j_posed = pts_j[pose_mask]
        pairs_p = [p for p, keep in zip(pairs, pose_mask) if keep] 

        X = self.week3.triangulate_points(pts_i_posed, pts_j_posed, self.K, R, t)
        err1 = self.week3.compute_reprojection_errors(X, pts_i_posed, self.K, eye, zero)
        err2 = self.week3.compute_reprojection_errors(X, pts_j_posed, self.K, R, t)
        keep = self.week3.filter_reconstructed_points(
            X, err1, err2, R, t, max_reprojection_error=max_reproj
        )

        colours = self.week3.sample_point_colours(self.features[i].image, pts_i_posed[keep])
        kept_pairs = [p for p, k in zip(pairs_p, keep) if k]
        for (i_kp, j_kp), xyz, colour in zip(kept_pairs, X[keep], colours):
            self._add_track(xyz, colour, {i: i_kp, j: j_kp}) # enables back-tracking from the 3D point to its 2D observations in both images, and links those keypoints to the point in the registered cameras.
        return int(np.sum(keep))

    def gather_2d3d(self, new_image_id: int) -> dict:
        """Collect 2D-3D correspondences for an unregistered image index 'new_image_id'.

        For each registered image, follow matches into its keypoint->point map.
        Each keypoint in the unregistered image `new_image_id` is used once (first match wins).
        """
        seen: dict[int, int] = {}   # stores a collection of seen keypoint indices in the unregistered image `new_image_id` and their corresponding 3D point IDs. 
        for registered_image_id in self.registered:
            registered_image = self.registered[registered_image_id]
            for (r_kp, u_kp) in self.matches_between(registered_image_id, new_image_id):
                if u_kp in seen:
                    continue
                point_id = registered_image.kpidx_to_point.get(r_kp)
                if point_id is not None:
                    seen[u_kp] = point_id

        if not seen:
            return {"count": 0, "points3d": np.empty((0, 3)), "pts2d": np.empty((0, 2)),
                    "u_kp": [], "point_ids": []}

        u_kps = list(seen.keys())
        point_ids = [seen[k] for k in u_kps]
        kp = self.features[new_image_id].keypoints
        pts2d = np.array([kp[k].pt for k in u_kps], dtype=np.float64)
        points3d = np.array([self.tracks[pid].xyz for pid in point_ids], dtype=np.float64)
        return {"count": len(u_kps), "points3d": points3d, "pts2d": pts2d,
                "u_kp": u_kps, "point_ids": point_ids}

    def _find_next(self, rejected: set, min_corr: int) -> tuple[int | None, dict | None]:
        """Most-connected eligible image, or (None, None) if none qualifies.

        Returns the unregistered, non-rejected image with the most 2D-3D
        correspondences, provided it reaches `min_corr`. Falling below `min_corr`
        does not reject an image its count can still rise as tracks are
        extended, so it stays a candidate for later rounds. The loop therefore
        stops only when no image can currently be placed, not at the first miss.
        """
        best_u = None
        best_corr = None
        best_count = min_corr - 1          # only counts >= min_corr qualify
        for u in self.features: #iteratively goes through each image and matches it against the registered images to gather 2D-3D correspondences. 
            if u in self.registered or u in rejected:
                continue
            corr = self.gather_2d3d(u)
            if corr["count"] > best_count: # keeps track of the image with the most correspondences that meets the minimum threshold, and returns that image and its correspondences for registration in the next step.
                best_count = corr["count"]
                best_u = u
                best_corr = corr
        return best_u, best_corr

    def _register(
        self,
        u: int,
        corr: dict,
        *,
        min_pnp_inliers: int,
        min_pnp_ratio: float,
        pnp_threshold: float,
        confidence: float,
    ) -> bool:
        """ Tries to register a new image_id 'u' from its 2D-3D correspondences in `corr`.
        Returns True on success, False on failure (not enough inliers or too low inlier ratio)."""
        min_pnp_ratio = np.clip(min_pnp_ratio, 0.01, 0.5) # Ensure that the minimum PnP inlier ratio is at least 1% to prevent overly lenient acceptance of pose estimations with very few correspondences.
        if corr["count"] < 6:
            return False
        try:
            R, t, mask = self.week3.estimate_camera_pose_pnp(
                corr["points3d"], corr["pts2d"], self.K,
                threshold=pnp_threshold, confidence=confidence,
            )
        except Exception:
            return False

        inliers = int(np.sum(mask))
        if inliers < min_pnp_inliers: # Checks if it satisfies the minimum number of inliers required for a valid pose estimation.
            return False
        if corr["count"] > 0 and inliers < min_pnp_ratio * corr["count"]: # Checks if the the number of inliers relative to the total correspondences is reasonable (low pnp ratio means the pose estimation is not reliable from )
            return False

        self.registered[u] = RegisteredCamera(
            u, np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3, 1)
        )
        for idx in np.nonzero(mask)[0]:
            pid = corr["point_ids"][idx]
            u_kp = corr["u_kp"][idx]
            self.registered[u].kpidx_to_point[u_kp] = pid
            self.tracks[pid].observations[u] = u_kp
        self._last_pnp_inliers = inliers
        return True

    def _extend_tracks(self, image_id: int, max_reproj: float = 4.0) -> int:
        """Link remaining keypoints in `image_id` to 3D points owned by its registered neighbours.
        Find keypoints in the new image that match keypoints already associated with 3D points in nearby registered images.
        Add only the first match per keypoint and only if the reprojection error under `image_id`'s pose is within `max_reproj`.
        This extends tracks, avoids duplicate triangulation, and improves connectivity so other images can reach the minimum correspondence threshold.
        Returns the number of observations added.
        """
        reg_u = self.registered[image_id]
        candidates: dict[int, int] = {}   # image_id_keypoint_idx -> point_id (first match wins)
        for r in self.registered:
            if r == image_id:
                continue
            reg_r = self.registered[r]
            for (r_kp, u_kp) in self.matches_between(r, image_id):
                if u_kp in reg_u.kpidx_to_point or u_kp in candidates:
                    continue
                pid = reg_r.kpidx_to_point.get(r_kp) # 
                if pid is not None:
                    candidates[u_kp] = pid
        if not candidates:
            return 0

        u_kps = list(candidates)
        pids = [candidates[k] for k in u_kps]
        kp = self.features[image_id].keypoints
        pts2d = np.array([kp[k].pt for k in u_kps], dtype=np.float64)
        X = np.array([self.tracks[p].xyz for p in pids], dtype=np.float64)
        err = self.week3.compute_reprojection_errors(X, pts2d, self.K, reg_u.R, reg_u.t)

        added = 0
        for u_kp, pid, e in zip(u_kps, pids, err):
            if np.isfinite(e) and e <= max_reproj:
                reg_u.kpidx_to_point[u_kp] = pid
                self.tracks[pid].observations[image_id] = u_kp
                added += 1
        return added

    def triangulate_new_points(self, image_id: int, max_reproj: float = 4.0, min_tri_angle: float = 2.0) -> int:
        """Triangulate points `image_id` shares with registered neighbours that are not yet in the map. Returns the number of new points added."""
        added = 0
        reg_u = self.registered[image_id]
        Ru, tu = reg_u.R, reg_u.t

        for r in list(self.registered):
            if r == image_id:
                continue
            reg_r = self.registered[r]
            fresh = [
                (r_kp, u_kp)
                for (r_kp, u_kp) in self.matches_between(r, image_id)
                if r_kp not in reg_r.kpidx_to_point and u_kp not in reg_u.kpidx_to_point
            ]
            if not fresh:
                continue

            pts_r, pts_u = self._pts_from_pairs(r, image_id, fresh)
            X = triangulate_general(self.K, reg_r.R, reg_r.t, Ru, tu, pts_r, pts_u)
            err_r = self.week3.compute_reprojection_errors(X, pts_r, self.K, reg_r.R, reg_r.t)
            err_u = self.week3.compute_reprojection_errors(X, pts_u, self.K, Ru, tu)
            depth_r = _depths_in_camera(X, reg_r.R, reg_r.t)
            depth_u = _depths_in_camera(X, Ru, tu)
            ang = _triangulation_angles(X, reg_r.R, reg_r.t, Ru, tu)
            keep = (
                np.isfinite(X).all(axis=1)
                & (depth_r > 0) & (depth_u > 0)
                & np.isfinite(err_r) & np.isfinite(err_u)
                & (err_r <= max_reproj) & (err_u <= max_reproj)
                & (ang >= min_tri_angle)  # low-angle points are unstable and often wrong
            )
            if not np.any(keep):
                continue

            colours = self.week3.sample_point_colours(self.features[r].image, pts_r[keep])
            kept_fresh = [p for p, k in zip(fresh, keep) if k]
            for (r_kp, u_kp), xyz, colour in zip(kept_fresh, X[keep], colours):
                # A u keypoint may match several neighbours; skip if already linked
                # by an earlier neighbour in this loop.
                if u_kp in reg_u.kpidx_to_point:
                    continue
                self._add_track(xyz, colour, {r: r_kp, image_id: u_kp})
                added += 1
        return added

    def _triangulate_multiview(self, observations: dict[int, int]) -> np.ndarray | None:
        """Linear triangulation of one point from ALL its registered views,
        holding poses fixed. Generalises triangulate_general to N>=2 cameras.
        Returns xyz, or None if < 2 views or the point falls at infinity.
        """
        rows = []
        for img_id, kp in observations.items():
            reg = self.registered.get(img_id)
            if reg is None:                      # only triangulate from placed cameras
                continue
            P = self.K @ np.hstack((reg.R, reg.t.reshape(3, 1)))
            x, y = self.features[img_id].keypoints[kp].pt
            rows.append(x * P[2] - P[0]) # x * (P·X)=0 -> two rows per view,
            rows.append(y * P[2] - P[1]) # where P0,P1,P2 are the rows of P
        if len(rows) < 4:                        # need >= 2 views (2 rows each)
            return None
        _, _, Vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        Xh = Vt[-1]
        if abs(Xh[3]) < 1e-12:                   # reject points at infinity (homogeneous w close to 0)
            return None
        return Xh[:3] / Xh[3]                    # dehomogenise (divide by w)  

    def retriangulate(self, max_reproj: float = 4.0) -> int:
        """Re-estimate every track's xyz from all its registered observations,
        holding camera poses fixed.

        An update is committed only if the new position is in front of every
        observing camera (cheirality) and reprojects within `max_reproj` in all
        of them; otherwise the old position is kept. Returns the number of
        points whose position was updated.

        NB: not bundle adjustment
        """
        updated = 0
        for track in self.tracks.values():
            obs = {i: k for i, k in track.observations.items() if i in self.registered}
            if len(obs) < 2: # need >= 2 views to retriangulate
                continue
            X = self._triangulate_multiview(obs)
            if X is None:
                continue
            ok = True # all-or-nothing: any bad view rejects it
            for img_id, kp in obs.items():
                reg = self.registered[img_id]
                if _depths_in_camera(X.reshape(1, 3), reg.R, reg.t)[0] <= 0:   # cheirality
                    ok = False
                    break
                pt = np.array([self.features[img_id].keypoints[kp].pt], dtype=np.float64)
                e = self.week3.compute_reprojection_errors(
                    X.reshape(1, 3), pt, self.K, reg.R, reg.t
                )[0]
                if not np.isfinite(e) or e > max_reproj:
                    ok = False
                    break
            if ok:
                track.xyz = X
                updated += 1
        return updated

    def run(
        self,
        *,
        min_corr: int = 12,
        min_pnp_inliers: int = 10,
        min_pnp_ratio: float = 0.0,
        pnp_threshold: float = 6.0,
        confidence: float = 0.999,
        max_reproj: float = 4.0,
        retriangulate_interval: int = 5,   # 0 disables in-loop retriangulation
    ) -> list[StepMetrics]:
        """Greedily register the remaining pool images. Returns per-step metrics."""
        if len(self.registered) < 2:
            raise RuntimeError("Call initialise_two_view before run()")

        metrics: list[StepMetrics] = []
        rejected: set = set()
        step = 0

        while True:
            u, corr = self._find_next(rejected, min_corr) 
            if u is None:             # nothing left that reaches min_corr
                break

            ok = self._register(
                u, corr,
                min_pnp_inliers=min_pnp_inliers,
                min_pnp_ratio=min_pnp_ratio,
                pnp_threshold=pnp_threshold,
                confidence=confidence,
            )
            if not ok:
                rejected.add(u)        # too weak for now; retry after structure grows
                continue

            self._extend_tracks(u, max_reproj=max_reproj)
            new_points = self.triangulate_new_points(u, max_reproj=max_reproj)
            rejected.clear()           # the cloud changed; give failed images another chance
            step += 1
            retri = 0
            if retriangulate_interval and step % retriangulate_interval == 0:
                retri = self.retriangulate(max_reproj=max_reproj)
                rejected.clear()       # cloud improved; let stalled images retry
            metrics.append(StepMetrics(
                step=step,
                image=self.features[u].path.name,
                n_2d3d=corr["count"],
                pnp_inliers=self._last_pnp_inliers,
                new_points=new_points,
                n_registered=len(self.registered),
                n_points=len(self.tracks),
                retriangulated=retri,
            ))
        self.errors_before_retri = self.reprojection_errors()
        self.final_retri_updated = self.retriangulate(max_reproj=max_reproj)
        self.errors_after_retri = self.reprojection_errors()
        return metrics


    def point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.tracks:
            return np.empty((0, 3)), np.empty((0, 3))
        pts = np.array([t.xyz for t in self.tracks.values()], dtype=np.float64)
        cols = np.array([t.colour for t in self.tracks.values()])
        return pts, cols

    def camera_list(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        return [
            (self.features[i].path.name, self.registered[i].R, self.registered[i].t)
            for i in sorted(self.registered)
        ]

    def reprojection_errors(self) -> np.ndarray:
        errors = []
        for track in self.tracks.values():
            for img_id, kp in track.observations.items():
                reg = self.registered[img_id]
                proj = self.week3.project_points(track.xyz.reshape(1, 3), self.K, reg.R, reg.t)
                obs = np.array(self.features[img_id].keypoints[kp].pt, dtype=np.float64)
                errors.append(float(np.linalg.norm(proj[0] - obs)))
        return np.array(errors, dtype=np.float64)
