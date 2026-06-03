import numpy as np

from gf4.week3.two_view_utils import estimate_essential_matrix, recover_relative_pose


def _skew(v):
    x, y, z = v.ravel()
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _project(points3d, K, R, t):
    camera_points = (R @ points3d.T + t.reshape(3, 1)).T
    image_points = (K @ camera_points.T).T
    return image_points[:, :2] / image_points[:, 2:]


def _synthetic_pair():
    K = np.array(
        [
            [900.0, 0.0, 320.0],
            [0.0, 900.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    angle = np.deg2rad(6.0)
    R = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    t = np.array([[0.7], [0.05], [0.1]])
    t /= np.linalg.norm(t)

    xs = np.linspace(-1.0, 1.0, 6)
    ys = np.linspace(-0.7, 0.7, 5)
    points = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = 5.0 + 0.2 * i + 0.15 * j
            points.append([x, y, z])
    points3d = np.asarray(points)

    pts1 = _project(points3d, K, np.eye(3), np.zeros((3, 1)))
    pts2 = _project(points3d, K, R, t)
    return K, R, t, pts1, pts2


def test_estimate_essential_matrix_returns_boolean_inlier_mask():
    K, _, _, pts1, pts2 = _synthetic_pair()

    E, mask = estimate_essential_matrix(pts1, pts2, K, threshold=0.5)

    assert E.shape[1] == 3
    assert mask.dtype == bool
    assert mask.shape == (len(pts1),)
    assert 0 < np.sum(mask) <= len(pts1)


def test_recover_relative_pose_uses_mask_and_returns_boolean_mask():
    K, expected_R, expected_t, pts1, pts2 = _synthetic_pair()
    E = _skew(expected_t) @ expected_R

    R, t, mask = recover_relative_pose(E, pts1, pts2, K, inlier_mask=np.ones(len(pts1)))

    t /= np.linalg.norm(t)
    assert mask.dtype == bool
    assert mask.shape == (len(pts1),)
    assert np.sum(mask) >= 0.8 * len(pts1)
    assert np.allclose(R, expected_R, atol=1e-5)
    assert float(t.ravel() @ expected_t.ravel()) > 0.99
