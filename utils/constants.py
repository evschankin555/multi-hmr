# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os
from pathlib import Path

# Получаем абсолютный путь к директории multi-hmr
# __file__ = tools/multi-hmr/utils/constants.py
# parents[1] = tools/multi-hmr
_MULTIHMR_ROOT = Path(__file__).resolve().parents[1]

# Устанавливаем абсолютные пути
SMPLX_DIR = str(_MULTIHMR_ROOT / 'models')
MEAN_PARAMS = str(_MULTIHMR_ROOT / 'models' / 'smpl_mean_params.npz')
CACHE_DIR_MULTIHMR = str(_MULTIHMR_ROOT / 'models' / 'multiHMR')

ANNOT_DIR = str(_MULTIHMR_ROOT / 'data')
BEDLAM_DIR = str(_MULTIHMR_ROOT / 'data' / 'BEDLAM')
EHF_DIR = str(_MULTIHMR_ROOT / 'data' / 'EHF')
THREEDPW_DIR = str(_MULTIHMR_ROOT / 'data' / '3DPW')

SMPLX2SMPL_REGRESSOR = str(_MULTIHMR_ROOT / 'models' / 'smplx' / 'smplx2smpl.pkl')