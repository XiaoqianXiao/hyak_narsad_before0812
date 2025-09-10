import numpy as np
from nilearn.image import index_img, load_img
from nilearn.maskers import NiftiSpheresMasker, NiftiLabelsMasker
from nilearn.input_data import NiftiMasker
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import nibabel as nib
from joblib import Parallel, delayed
import os
import re
import logging

logger = logging.getLogger(__name__)

def searchlight_similarity(img1, img2, radius=6, affine=None, mask_img=None, similarity='pearson', n_jobs=4):
    """
    Compute voxel-wise similarity between two 3D or 4D images using a searchlight approach.

    Parameters:
        img1: nib.Nifti1Image - first image (3D or 4D)
        img2: nib.Nifti1Image - second image (must match img1 shape)
        radius: int - radius of the searchlight sphere in mm
        affine: optional affine to transform coordinates (default: from mask_img)
        mask_img: binary Nifti image for where to apply searchlight
        similarity: 'pearson' or 'cosine'
        n_jobs: int - number of parallel jobs for voxel processing

    Returns:
        similarity_map: nib.Nifti1Image with similarity at each voxel
    """
    logger.info(f"Starting searchlight similarity with radius={radius}, similarity={similarity}, n_jobs={n_jobs}")
    try:
        masker = NiftiMasker(mask_img=mask_img)
        masker.fit()
        logger.info(f"Masker fitted, mask shape: {masker.mask_img_.shape}")
        img1_data = masker.transform(img1)  # Cache masked data
        img2_data = masker.transform(img2)
        logger.info(f"Transformed img1 shape: {img1_data.shape}, img2 shape: {img2_data.shape}")
    except Exception as e:
        logger.error(f"Error in masker setup or transform: {e}")
        raise

    coordinates = np.argwhere(masker.mask_img_.get_fdata() > 0)
    logger.info(f"Number of voxels to process: {len(coordinates)}")
    if affine is None:
        affine = masker.mask_img_.affine

    world_coords = nib.affines.apply_affine(affine, coordinates)

    def compute_voxel_similarity(coord, img1, img2, radius, voxel_num, total_voxels):
        try:
            sphere_masker = NiftiSpheresMasker([coord], radius=radius, detrend=False, standardize=False)
            sphere_ts1 = sphere_masker.fit_transform(img1)
            sphere_ts2 = sphere_masker.transform(img2)
            if sphere_ts1.shape[1] < 2:
                logger.warning(f"Skipping voxel {coord} ({voxel_num}/{total_voxels}): insufficient data points")
                return np.nan
            if similarity == 'pearson':
                sim = pearsonr(sphere_ts1.ravel(), sphere_ts2.ravel())[0]
            elif similarity == 'cosine':
                sim = cosine_similarity(sphere_ts1, sphere_ts2)[0, 0]
            else:
                raise ValueError("similarity must be 'pearson' or 'cosine'")
            logger.debug(f"Voxel {voxel_num}/{total_voxels} at {coord} similarity: {sim:.4f}")
            return sim
        except Exception as e:
            logger.error(f"Error computing similarity for voxel {coord} ({voxel_num}/{total_voxels}): {e}")
            return np.nan

    total_voxels = len(world_coords)
    similarity_values = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_voxel_similarity)(coord, img1, img2, radius, idx + 1, total_voxels)
        for idx, coord in enumerate(world_coords)
    )
    logger.info(f"Computed {len(similarity_values)} similarity values")

    similarity_map = np.full(masker.mask_img_.shape, np.nan)
    for i, coord in enumerate(coordinates):
        similarity_map[tuple(coord)] = similarity_values[i]

    return nib.Nifti1Image(similarity_map, masker.mask_img_.affine)

def roi_similarity(img1, img2, atlas_img, roi_labels, similarity='pearson', n_jobs=4):
    """
    Compute pairwise ROI similarities between two images.

    Parameters:
        img1: nib.Nifti1Image - first image
        img2: nib.Nifti1Image - second image
        atlas_img: nib.Nifti1Image - labeled Nifti image (ROIs > 0)
        roi_labels: list - list of valid ROI labels
        similarity: 'pearson' or 'cosine'
        n_jobs: int - number of parallel jobs for ROI pairs

    Returns:
        np.ndarray: Matrix of shape (n_rois, n_rois) with pairwise similarities
    """
    logger.info(f"Starting ROI similarity with {len(roi_labels)} ROIs, similarity={similarity}, n_jobs={n_jobs}")
    try:
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, detrend=False)
        roi_ts1 = masker.fit_transform(img1)  # Cache ROI time-series
        roi_ts2 = masker.transform(img2)
        logger.info(f"ROI time-series shape: ts1={roi_ts1.shape}, ts2={roi_ts2.shape}")
    except Exception as e:
        logger.error(f"Error in ROI masker setup or transform: {e}")
        raise

    n_rois = len(roi_labels)
    sim_matrix = np.zeros((n_rois, n_rois))

    def compute_roi_pair(i, j, ts1, ts2, pair_num, total_pairs):
        try:
            if similarity == 'pearson':
                sim = pearsonr(ts1[:, i], ts2[:, j])[0]
            elif similarity == 'cosine':
                sim = cosine_similarity(ts1[:, i].reshape(1, -1), ts2[:, j].reshape(1, -1))[0, 0]
            else:
                raise ValueError("similarity must be 'pearson' or 'cosine'")
            logger.debug(f"ROI pair {pair_num}/{total_pairs} ({i} vs {j}) similarity: {sim:.4f}")
            return sim
        except Exception as e:
            logger.error(f"Error computing ROI pair {i} vs {j} ({pair_num}/{total_pairs}): {e}")
            return np.nan

    pairs = [(i, j) for i in range(n_rois) for j in range(n_rois)]
    total_pairs = len(pairs)
    logger.info(f"Computing {total_pairs} ROI pairs")
    sim_values = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_roi_pair)(i, j, roi_ts1, roi_ts2, idx + 1, total_pairs)
        for idx, (i, j) in enumerate(pairs)
    )

    for idx, (i, j) in enumerate(pairs):
        sim_matrix[i, j] = sim_values[idx]

    return sim_matrix

def load_roi_names(names_file_path, roi_labels):
    """
    File format: alternating lines
      - Odd lines: ROI name (e.g., 'HIP-rh', '7Networks_LH_Vis_1')
      - Even lines: 'label R G B A'
    Returns: dict with **int** keys and formatted names
    """
    logger.info(f"Loading ROI names from {names_file_path}")
    def format_name(name: str) -> str:
        s = name.strip()
        m = re.match(r"^(.+)-(rh|lh)$", s, flags=re.IGNORECASE)
        if m:
            region, hemi = m.group(1), m.group(2).lower()
            return f"{hemi}_{region}"
        m = re.match(r"^7Networks_(LH|RH)_(.+)$", s)
        if m:
            hemi = m.group(1).lower()
            rest = m.group(2)
            m_idx = re.match(r"^(.*)_(\d+)$", rest)
            if m_idx:
                base, idx = m_idx.group(1), m_idx.group(2)
                return f"{hemi}_{base}-{idx}"
            else:
                return f"{hemi}_{rest}"
        return s

    if not os.path.exists(names_file_path):
        logger.warning(f"ROI names file not found: {names_file_path}. Using numerical labels.")
        return {int(l): f"combined_ROI_{int(l)}" for l in roi_labels}

    intlabel_to_rawname = {}
    try:
        with open(names_file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i in range(0, len(lines), 2):
            name_line = lines[i]
            if i + 1 >= len(lines):
                continue
            nums = lines[i + 1].split()
            try:
                label_int = int(nums[0])
            except (IndexError, ValueError):
                continue
            intlabel_to_rawname[label_int] = name_line
    except Exception as e:
        logger.error(f"Error reading ROI names file: {e}. Using numerical labels.")
        return {int(l): f"combined_ROI_{int(l)}" for l in roi_labels}

    roi_names = {}
    for lab in roi_labels:
        lab_int = int(lab)
        raw = intlabel_to_rawname.get(lab_int)
        roi_names[lab_int] = format_name(raw) if raw is not None else f"combined_ROI_{lab_int}"
    logger.info(f"Loaded {len(roi_names)} ROI names. Example: {list(roi_names.items())[:10]}")
    return roi_names

def get_roi_labels(atlas_img, atlas_name):
    logger.info(f"Extracting ROI labels from {atlas_name}")
    atlas_data = atlas_img.get_fdata()
    roi_labels = np.unique(atlas_data)[np.unique(atlas_data) > 0]
    if len(roi_labels) == 0:
        logger.error(f"No valid ROIs found in {atlas_name} atlas (all values <= 0)")
        raise ValueError(f"No valid ROIs found in {atlas_name} atlas")
    logger.info(f"Found {len(roi_labels)} ROI labels")
    return roi_labels