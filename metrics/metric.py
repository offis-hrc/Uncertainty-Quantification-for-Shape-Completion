import torch
import open3d as o3d

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance
from extensions.earth_movers_distance.emd_confidence import EarthMoverDistanceConfidence
from extensions.chamfer_distance.chamfer_distance_confidence import ChamferDistanceConfidence


CD = ChamferDistance()
EMD = EarthMoverDistance()
EMD_CONFIDENCE = EarthMoverDistanceConfidence()
CD_Confidence = ChamferDistanceConfidence()

def l2_cd(pcs1, pcs2):
    pcs1 = pcs1[:,:,:3]
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return torch.sum(dist1 + dist2)


def l1_cd(pcs1, pcs2):
    pcs1 = pcs1[:,:,:3]
    dist1, dist2 = CD(pcs1, pcs2)
    dist1_point = dist1
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2

def l1_cd_point_old(pcs1, pcs2):
    confidence_sig = torch.nn.functional.sigmoid(pcs1[:,:,3])
    confidence = confidence_sig[:, :, None]
    pcs1 = torch.cat([pcs1[:,:,:3], confidence], dim=2)

    dist1, dist2 = CD_Confidence(pcs1, pcs2)
    dist1_point = torch.sqrt(dist1)
    dist2_point = torch.sqrt(dist2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2, dist1_point, dist2_point

def l1_cd_point(pcs1, pcs2):
    #confidence_sig = torch.nn.functional.sigmoid(pcs1[:,:,3])
    #confidence = confidence_sig[:, :, None]
    #pcs1 = torch.cat([pcs1[:,:,:3], confidence], dim=2)

    dist1, dist2 = CD_Confidence(pcs1, pcs2)
    dist1_point = torch.sqrt(dist1)
    dist2_point = torch.sqrt(dist2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2, dist1_point, dist2_point


def emd(pcs1, pcs2):
    dists, match = EMD(pcs1, pcs2)
    return torch.sum(dists)

def emd_point(pcs1, pcs2):
    dists, match = EMD_CONFIDENCE(pcs1, pcs2)
    return dists


def f_score(pred, gt, th=0.01):
    """
    References: https://github.com/lmb-freiburg/what3d/blob/master/util.py

    Args:
        pred (np.ndarray): (N1, 3)
        gt   (np.ndarray): (N2, 3)
        th   (float): a distance threshhold
    """
    pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

    dist1 = pred.compute_point_cloud_distance(gt)
    dist2 = gt.compute_point_cloud_distance(pred)

    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0
