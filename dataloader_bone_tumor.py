import os
import random
import numpy as np
import torch
from typing import List, Dict, Optional
from scipy import ndimage

from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
    RandAffined,
    NormalizeIntensityd,
    RandScaleIntensityd,
    ScaleIntensityRangePercentilesd,
)

from monai.data import DataLoader, CacheDataset, PersistentDataset, list_data_collate, MetaTensor
from torch.utils.data import Dataset as TorchDataset


def _detect_zero_edges_by_contagion(slice_data: np.ndarray) -> np.ndarray:
    edge_mask = np.zeros_like(slice_data, dtype=bool)
    h, w = slice_data.shape
    ew, eh = max(1, int(w * 0.15)), max(1, int(h * 0.15))
    edge_mask[:eh, :] |= (slice_data[:eh, :] == 0)
    edge_mask[-eh:, :] |= (slice_data[-eh:, :] == 0)
    edge_mask[:, :ew] |= (slice_data[:, :ew] == 0)
    edge_mask[:, -ew:] |= (slice_data[:, -ew:] == 0)
    for _ in range(5):
        edge_mask = ndimage.binary_dilation(edge_mask, structure=np.ones((3, 3)))
        edge_mask &= (slice_data == 0)
    return edge_mask


def _detect_zero_edges_by_connectivity(slice_data: np.ndarray) -> np.ndarray:
    all_zeros = slice_data == 0
    if not np.any(all_zeros):
        return np.zeros_like(slice_data, dtype=bool)
    labeled, n = ndimage.label(all_zeros)
    out = np.zeros_like(slice_data, dtype=bool)
    for i in range(1, n + 1):
        m = labeled == i
        if np.any(m[0, :]) or np.any(m[-1, :]) or np.any(m[:, 0]) or np.any(m[:, -1]):
            out |= m
    return out


def _replace_zero_edge_background_3d(vol: np.ndarray, black_value: float = -2048) -> np.ndarray:
    if vol.ndim != 3:
        return vol
    out = vol.copy()
    for i in range(out.shape[-1]):
        mask = _detect_zero_edges_by_contagion(out[..., i]) | _detect_zero_edges_by_connectivity(out[..., i])
        out[..., i][mask] = black_value
    return out


class ReplaceZeroEdgeBackgroundd:
    """CT 边缘连通零值背景 → -2048，输出保留 MetaTensor 元数据。"""

    def __init__(self, keys, black_value: float = -2048):
        self.keys = list(keys)
        self.black_value = black_value

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            v = d[key]
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            if arr.ndim == 4 and arr.shape[0] == 1:
                out = _replace_zero_edge_background_3d(arr[0], self.black_value)[np.newaxis, ...].astype(arr.dtype, copy=False)
            elif arr.ndim == 3:
                out = _replace_zero_edge_background_3d(arr, self.black_value).astype(arr.dtype, copy=False)
            else:
                continue
            if isinstance(v, MetaTensor) and getattr(v, "meta", None) is not None:
                meta = dict(v.meta) if isinstance(v.meta, dict) else {}
                d[key] = MetaTensor(out, meta=meta)
            else:
                d[key] = out
        return d


def get_paired_ct_mr_data_dict(root_dir: str, phase: str = 'train',
                                train_ratio: float = 0.8, random_seed: int = 42) -> tuple[List[Dict], List[Dict]]:
    """
    获取配对的CT和MR数据字典，确保CT和MR来自同一个患者
    
    Args:
        root_dir: 数据集根目录
        phase: 'train' 或 'val'
        train_ratio: 训练集比例
        random_seed: 随机种子
    
    Returns:
        (ct_data_dicts, mr_data_dicts): 配对的CT和MR数据字典列表
    """
    batch_folders = ['第1批', '第2批', '第3批', '第4批', '第5批', '上海市一']
    ct_data = []
    mr_data = []
    
    # 用于存储每个患者的CT和MR数据
    patient_data = {}
    
    for batch_folder in batch_folders:
        batch_path = os.path.join(root_dir, batch_folder)
        if not os.path.exists(batch_path):
            print(f"[Warning] Batch folder not found: {batch_path}")
            continue
        
        patient_folders = [f for f in os.listdir(batch_path)
                          if os.path.isdir(os.path.join(batch_path, f))]
        
        for patient_id in patient_folders:
            patient_path = os.path.join(batch_path, patient_id)
            
            # 查找label文件 (.nii.gz 优先，fallback .nii)
            label_file = None
            for candidate in [
                os.path.join(patient_path, f"{patient_id}.nii.gz"),
                os.path.join(patient_path, f"{patient_id}.nii"),
            ]:
                if os.path.exists(candidate):
                    label_file = candidate
                    break

            if label_file is None:
                continue

            # 查找CT和MR图像文件
            ct_candidates = [
                os.path.join(patient_path, f"{patient_id}_ct_reg.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DF_ct_reg.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DFKN_ct_reg.nii.gz"),
            ]
            mr_candidates = [
                os.path.join(patient_path, f"{patient_id}_mr.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DF_mr.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DFKN_mr.nii.gz"),
            ]
            
            ct_file = None
            for candidate in ct_candidates:
                if os.path.exists(candidate):
                    ct_file = candidate
                    break
            
            mr_file = None
            for candidate in mr_candidates:
                if os.path.exists(candidate):
                    mr_file = candidate
                    break
            
            # 只有当CT和MR都存在时，才添加到配对数据中
            if ct_file is not None and mr_file is not None:
                patient_data[patient_id] = {
                    'ct': {
                        'image': ct_file,
                        'label': label_file,
                        'modality': 'CT',
                        'name': patient_id
                    },
                    'mr': {
                        'image': mr_file,
                        'label': label_file,
                        'modality': 'MR',
                        'name': patient_id
                    }
                }
    
    # 转换为列表并配对
    patient_ids = list(patient_data.keys())
    random.seed(random_seed)
    random.shuffle(patient_ids)
    
    split_idx = int(len(patient_ids) * train_ratio)
    
    if phase == 'train':
        patient_ids = patient_ids[:split_idx]
    elif phase == 'val':
        patient_ids = patient_ids[split_idx:]
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    # 创建配对的CT和MR数据列表
    for patient_id in patient_ids:
        ct_data.append(patient_data[patient_id]['ct'])
        mr_data.append(patient_data[patient_id]['mr'])
    
    print(f"[Paired Data] {phase.upper()} - Paired samples: {len(ct_data)} (CT and MR from same patients)")
    
    return ct_data, mr_data
#为了分别处理 CT 和 MR

def get_paired_data_dicts(root_dir: str, phase: str = 'train',
                          train_ratio: float = 0.8, random_seed: int = 42) -> List[Dict]:
    """
    获取配对的CT+MR数据字典（合并格式），用于联合增强。

    每个元素包含 'ct', 'mr', 'label', 'name'，CT和MR对应同一病例。
    只收录CT和MR都存在的病例。

    Returns:
        List[Dict]，每个元素: {'ct': str, 'mr': str, 'label': str, 'name': str}
    """
    batch_folders = ['第1批', '第2批', '第3批', '第4批', '第5批', '上海市一']
    patient_data = {}

    for batch_folder in batch_folders:
        batch_path = os.path.join(root_dir, batch_folder)
        if not os.path.exists(batch_path):
            print(f"[Warning] Batch folder not found: {batch_path}")
            continue

        patient_folders = [f for f in os.listdir(batch_path)
                           if os.path.isdir(os.path.join(batch_path, f))]

        for patient_id in patient_folders:
            patient_path = os.path.join(batch_path, patient_id)

            # 查找 label 文件，优先查找 .nii.gz，如果不存在则查找 .nii
            label_file = None
            for candidate in [
                os.path.join(patient_path, f"{patient_id}.nii.gz"),
                os.path.join(patient_path, f"{patient_id}.nii"),
            ]:
                if os.path.exists(candidate):
                    label_file = candidate
                    break  # 找到一个就停止

            if label_file is None:
                print(f"[Warning] Label file not found for patient: {patient_id}")
                continue  # 跳过此病例

            ct_file = None
            for candidate in [
                os.path.join(patient_path, f"{patient_id}_ct_reg.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DF_ct_reg.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DFKN_ct_reg.nii.gz"),
            ]:
                if os.path.exists(candidate):
                    ct_file = candidate
                    break

            mr_file = None
            for candidate in [
                os.path.join(patient_path, f"{patient_id}_mr.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DF_mr.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DFKN_mr.nii.gz"),
            ]:
                if os.path.exists(candidate):
                    mr_file = candidate
                    break

            if ct_file is not None and mr_file is not None:
                patient_data[patient_id] = {
                    'ct': ct_file,
                    'mr': mr_file,
                    'label': label_file,
                    'name': patient_id,
                }

    patient_ids = sorted(patient_data.keys())
    random.seed(random_seed)
    random.shuffle(patient_ids)

    split_idx = int(len(patient_ids) * train_ratio)
    if phase == 'train':
        selected = patient_ids[:split_idx]
    elif phase == 'val':
        selected = patient_ids[split_idx:]
    else:
        raise ValueError(f"Unknown phase: {phase}")

    data_list = [patient_data[pid] for pid in selected]
    print(f"[Paired Data] {phase.upper()} - {len(data_list)} paired CT+MR cases")
    return data_list
#为了联合处理 CT 和 MR 数据

def get_bone_tumor_data_dict(root_dir: str, modality: str = 'CT', phase: str = 'train',
                              train_ratio: float = 0.8, random_seed: int = 42) -> List[Dict]:
    """
    获取bone tumor数据集的数据字典

    数据集结构:
    root_dir/
        第1批/
            patient_id/
                patient_id.nii.gz (label)
                patient_id_ct_reg.nii.gz 或 patient_id-DF_ct_reg.nii.gz (CT image)
                patient_id_mr.nii.gz 或 patient_id-DF_mr.nii.gz (MR image)
        第2批/
        第3批/
        ...

    Args:
        root_dir: 数据集根目录
        modality: 'CT', 'MR', 或 'MIX'
        phase: 'train' 或 'val'
        train_ratio: 训练集比例
        random_seed: 随机种子

    Returns:
        数据字典列表，每个元素包含 'image', 'label', 'modality', 'name'
    """

    # 先按患者收集完整信息，再做患者级切分，避免 MIX 模式下同一患者跨 train/val 泄漏
    batch_folders = ['第1批', '第2批', '第3批', '第4批', '第5批', '上海市一']
    patient_data = {}

    for batch_folder in batch_folders:
        batch_path = os.path.join(root_dir, batch_folder)
        if not os.path.exists(batch_path):
            print(f"[Warning] Batch folder not found: {batch_path}")
            continue

        # 遍历每个病例文件夹
        patient_folders = [f for f in os.listdir(batch_path)
                          if os.path.isdir(os.path.join(batch_path, f))]

        for patient_id in patient_folders:
            patient_path = os.path.join(batch_path, patient_id)

            # 查找label文件 (.nii.gz 优先，fallback .nii)
            label_file = None
            for candidate in [
                os.path.join(patient_path, f"{patient_id}.nii.gz"),
                os.path.join(patient_path, f"{patient_id}.nii"),
            ]:
                if os.path.exists(candidate):
                    label_file = candidate
                    break

            if label_file is None:
                print(f"[Warning] Label file not found for patient: {patient_id}")
                continue

            # 查找CT和MR图像文件
            # 注意：只使用带有ct_reg的CT图像（已配准），不使用只有ct的文件（未配准）
            ct_candidates = [
                os.path.join(patient_path, f"{patient_id}_ct_reg.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DF_ct_reg.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DFKN_ct_reg.nii.gz"),
            ]
            mr_candidates = [
                os.path.join(patient_path, f"{patient_id}_mr.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DF_mr.nii.gz"),
                os.path.join(patient_path, f"{patient_id}-DFKN_mr.nii.gz"),
            ]

            ct_file = None
            for candidate in ct_candidates:
                if os.path.exists(candidate):
                    ct_file = candidate
                    break

            mr_file = None
            for candidate in mr_candidates:
                if os.path.exists(candidate):
                    mr_file = candidate
                    break

            patient_data[patient_id] = {
                'label': label_file,
                'ct': ct_file,
                'mr': mr_file,
                'name': patient_id,
            }

    if len(patient_data) == 0:
        raise ValueError(f"No data found in {root_dir} for modality {modality}")

    # 患者级切分
    patient_ids = sorted(patient_data.keys())
    random.seed(random_seed)
    random.shuffle(patient_ids)

    split_idx = int(len(patient_ids) * train_ratio)

    if phase == 'train':
        selected_patient_ids = patient_ids[:split_idx]
    elif phase == 'val':
        selected_patient_ids = patient_ids[split_idx:]
    else:
        raise ValueError(f"Unknown phase: {phase}")

    data_list = []
    for patient_id in selected_patient_ids:
        patient = patient_data[patient_id]
        label_file = patient['label']
        ct_file = patient['ct']
        mr_file = patient['mr']

        if modality == 'CT':
            if ct_file is not None:
                data_list.append({
                    'image': ct_file,
                    'label': label_file,
                    'modality': 'CT',
                    'name': patient_id,
                    'patient_id': patient_id,
                })
        elif modality == 'MR':
            if mr_file is not None:
                data_list.append({
                    'image': mr_file,
                    'label': label_file,
                    'modality': 'MR',
                    'name': patient_id,
                    'patient_id': patient_id,
                })
        elif modality == 'MIX':
            if ct_file is not None:
                data_list.append({
                    'image': ct_file,
                    'label': label_file,
                    'modality': 'CT',
                    'name': f"{patient_id}_CT",
                    'patient_id': patient_id,
                })
            if mr_file is not None:
                data_list.append({
                    'image': mr_file,
                    'label': label_file,
                    'modality': 'MR',
                    'name': f"{patient_id}_MR",
                    'patient_id': patient_id,
                })
        else:
            raise ValueError(f"Unknown modality: {modality}")

    if len(data_list) == 0:
        raise ValueError(f"No data found in {root_dir} for modality {modality} after patient-level split")

    print(f"[Bone Tumor Data] {phase.upper()} - {len(selected_patient_ids)} patients, {len(data_list)} samples ({modality})")
    return data_list
#构建单模态数据集

def get_loader_bone_tumor(root_dir: str,
                          modality: str = 'CT',
                          phase: str = 'train',
                          batch_size: int = 1,
                          roi_size: tuple = (96, 96, 96),
                          train_num_samples: int = 2,
                          train_ratio: float = 0.8,
                          random_seed: int = 42,
                          persistent: bool = True,
                          cache_dir: str = "",
                          num_workers: int = 8):
    """
    创建bone tumor数据集的DataLoader

    Args:
        root_dir: 数据集根目录
        modality: 'CT', 'MR', 或 'MIX'
        phase: 'train' 或 'val'
        batch_size: batch大小
        roi_size: ROI大小 (D, H, W)
        train_num_samples: 训练时每个样本的随机crop数量
        train_ratio: 训练集比例
        random_seed: 随机种子
        persistent: 是否使用PersistentDataset
        cache_dir: 缓存目录
        num_workers: DataLoader的worker数量

    Returns:
        DataLoader对象
    """

    # CT训练变换 - 分为可缓存部分和随机增强部分
    # 可缓存部分（非随机）：加载、预处理、padding
    ct_train_transforms_cache = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-125,
            a_max=275,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
    ])
    
    # 随机增强部分（不能缓存）：每次epoch都重新执行
    ct_train_transforms_aug = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=train_num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0,
            spatial_size=roi_size,
            rotate_range=(0, 0, np.pi / 30),
            scale_range=(0.1, 0.1, 0.1)
        ),
        ToTensord(keys=["image", "label"]),
    ])
    
    # 完整训练transform（用于非persistent模式）
    ct_train_transforms = Compose([
        ct_train_transforms_cache,
        ct_train_transforms_aug,
    ])

    # CT验证变换
    ct_val_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-125,
            a_max=275,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
        ToTensord(keys=["image", "label"]),
    ])

    # MR训练变换 - 分为可缓存部分和随机增强部分
    # 可缓存部分（非随机）：加载、预处理、padding
    mr_train_transforms_cache = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
    ])
    
    # 随机增强部分（不能缓存）：每次epoch都重新执行
    mr_train_transforms_aug = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=train_num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
    ])
    
    # 完整训练transform（用于非persistent模式）
    mr_train_transforms = Compose([
        mr_train_transforms_cache,
        mr_train_transforms_aug,
    ])

    # MR验证变换
    mr_val_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
        ToTensord(keys=["image", "label"]),
    ])

    # 设置缓存目录
    if cache_dir == "":
        cache_dir = os.path.join(root_dir, 'cache_bone_tumor')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # 对于训练集，使用新的缓存目录结构（分离缓存和增强）
    # 这样可以避免使用旧的包含随机增强的缓存
    if persistent and phase == 'train':
        # 使用新的缓存目录命名，避免与旧缓存冲突
        cache_subdir = os.path.join(cache_dir, f'{modality}_{phase}_cache_v2')
        if os.path.exists(cache_subdir):
            # 检查是否有旧的完整transform缓存（可能包含随机增强）
            old_cache_dir = os.path.join(cache_dir, f'{modality}_{phase}')
            if os.path.exists(old_cache_dir):
                print(f"[WARNING] Found old cache directory: {old_cache_dir}")
                print(f"[WARNING] Old cache may contain random augmentations and will be ignored.")
                print(f"[INFO] Using new separated cache: {cache_subdir}")

    # 获取数据字典
    data_dicts = get_bone_tumor_data_dict(
        root_dir=root_dir,
        modality=modality,
        phase=phase,
        train_ratio=train_ratio,
        random_seed=random_seed
    )

    print(f"[Bone Tumor Dataset] {phase.upper()} - Modality: {modality}, Samples: {len(data_dicts)}")
    
    if persistent and phase == 'train':
        print(f"[INFO] Using separated cache strategy for training:")
        print(f"  - Non-random transforms (Load, Spacing, etc.) will be cached")
        print(f"  - Random augmentations (Crop, Rotate, etc.) will be applied on-the-fly each epoch")
        print(f"  - This ensures data diversity across epochs")

    # 对于训练集，如果使用persistent，需要分离缓存和随机增强
    # 对于验证集，可以直接使用完整的transform
    if persistent and phase == 'train':
        # 训练集：先缓存非随机部分，再应用随机增强
        
        # 创建缓存数据集（只缓存非随机部分）
        # 使用v2后缀避免与旧缓存冲突
        cache_subdir = os.path.join(cache_dir, f'{modality}_{phase}_cache_v2')
        
        if modality == 'MIX':
            # MIX模式：为CT和MR分别创建缓存transform
            def ct_cache_transform(data_dict):
                if data_dict['modality'] == 'CT':
                    return ct_train_transforms_cache(data_dict)
                else:
                    return data_dict  # MR数据不在这里处理
            
            def mr_cache_transform(data_dict):
                if data_dict['modality'] == 'MR':
                    return mr_train_transforms_cache(data_dict)
                else:
                    return data_dict  # CT数据不在这里处理
            
            # 分别缓存CT和MR
            ct_data_dicts = [d for d in data_dicts if d['modality'] == 'CT']
            mr_data_dicts = [d for d in data_dicts if d['modality'] == 'MR']
            
            if len(ct_data_dicts) > 0:
                ct_cache_dataset = PersistentDataset(
                    data=ct_data_dicts,
                    transform=ct_train_transforms_cache,
                    cache_dir=os.path.join(cache_subdir, 'CT')
                )
            else:
                ct_cache_dataset = None
                
            if len(mr_data_dicts) > 0:
                mr_cache_dataset = PersistentDataset(
                    data=mr_data_dicts,
                    transform=mr_train_transforms_cache,
                    cache_dir=os.path.join(cache_subdir, 'MR')
                )
            else:
                mr_cache_dataset = None
            
            # 创建包装数据集，在访问时应用随机增强
            class AugmentedDataset(TorchDataset):
                def __init__(self, cache_dataset, aug_transform, modality_type):
                    super().__init__()
                    self.cache_dataset = cache_dataset
                    self.aug_transform = aug_transform
                    self.modality_type = modality_type
                
                def __len__(self):
                    if self.cache_dataset is None:
                        return 0
                    # RandCropByPosNegLabeld会返回num_samples个样本，所以实际长度需要乘以num_samples
                    # 但这里我们返回的是原始数据集的长度，DataLoader会通过list_data_collate处理
                    return len(self.cache_dataset)
                
                def __getitem__(self, index):
                    # 从缓存数据集获取数据（已经过非随机预处理）
                    data = self.cache_dataset[index]
                    # 应用随机增强（每次访问都会重新执行，确保随机性）
                    # RandCropByPosNegLabeld可能返回列表，list_data_collate会处理
                    augmented_data = self.aug_transform(data)
                    return augmented_data
            
            datasets = []
            if ct_cache_dataset:
                datasets.append(AugmentedDataset(ct_cache_dataset, ct_train_transforms_aug, 'CT'))
            if mr_cache_dataset:
                datasets.append(AugmentedDataset(mr_cache_dataset, mr_train_transforms_aug, 'MR'))
            
            from torch.utils.data import ConcatDataset
            dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        else:
            # 单一模态
            cache_transform = ct_train_transforms_cache if modality == 'CT' else mr_train_transforms_cache
            aug_transform = ct_train_transforms_aug if modality == 'CT' else mr_train_transforms_aug
            
            cache_dataset = PersistentDataset(
                data=data_dicts,
                transform=cache_transform,
                cache_dir=cache_subdir
            )
            
            # 创建包装数据集，在访问时应用随机增强
            class AugmentedDataset(TorchDataset):
                def __init__(self, cache_dataset, aug_transform):
                    super().__init__()
                    self.cache_dataset = cache_dataset
                    self.aug_transform = aug_transform
                
                def __len__(self):
                    return len(self.cache_dataset)
                
                def __getitem__(self, index):
                    # 从缓存数据集获取数据（已经过非随机预处理）
                    data = self.cache_dataset[index]
                    # 应用随机增强（每次访问都会重新执行，确保随机性）
                    # RandCropByPosNegLabeld可能返回列表，list_data_collate会处理
                    augmented_data = self.aug_transform(data)
                    return augmented_data
            
            dataset = AugmentedDataset(cache_dataset, aug_transform)
    else:
        # 验证集或非persistent模式：使用完整的transform
        if modality == 'MIX':
            # 创建一个transform函数，根据每个样本的modality选择对应的transform
            def mixed_transform(data_dict):
                mod = data_dict['modality']
                if phase == 'train':
                    if mod == 'CT':
                        return ct_train_transforms(data_dict)
                    else:  # MR
                        return mr_train_transforms(data_dict)
                else:  # val
                    if mod == 'CT':
                        return ct_val_transforms(data_dict)
                    else:  # MR
                        return mr_val_transforms(data_dict)

            transform = mixed_transform
        else:
            # 单一模态，直接选择对应的transform
            if modality == 'CT':
                transform = ct_train_transforms if phase == 'train' else ct_val_transforms
            else:  # MR
                transform = mr_train_transforms if phase == 'train' else mr_val_transforms

        # 创建数据集
        cache_subdir = os.path.join(cache_dir, f'{modality}_{phase}')

        if persistent:
            dataset = PersistentDataset(
                data=data_dicts,
                transform=transform,
                cache_dir=cache_subdir
            )
        else:
            dataset = CacheDataset(
                data=data_dicts,
                transform=transform,
                cache_rate=1.0
            )

    # 创建DataLoader
    shuffle = (phase == 'train')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if phase == 'train' else 0,
        collate_fn=list_data_collate,
        pin_memory=True
    )

    return loader
#非配对数据增强

def get_loader_paired_bone_tumor(root_dir: str,
                                  phase: str = 'train',
                                  batch_size: int = 1,
                                  roi_size: tuple = (96, 96, 96),
                                  train_num_samples: int = 2,
                                  train_ratio: float = 0.8,
                                  random_seed: int = 42,
                                  persistent: bool = True,
                                  cache_dir: str = "",
                                  num_workers: int = 8):
    """
    创建配对 CT+MR 的 DataLoader，空间增强严格同步。

    数据格式：每个 dict 包含 'ct', 'mr', 'label', 'name'，CT 和 MR 来自同一病例。

    增强策略：
    - 空间增强（RandCrop、RandAffine）：ct/mr/label 同一 dict，MONAI 自动共享随机参数，
      保证 CT 和 MR 裁剪位置、旋转角度完全一致。
    - 强度增强：CT 和 MR 各自独立（模态差异大，不应强制一致）。

    Args:
        root_dir: 数据集根目录
        phase: 'train' 或 'val'
        batch_size: batch 大小
        roi_size: ROI 大小 (D, H, W)
        train_num_samples: 训练时每个样本随机 crop 数量
        train_ratio: 训练集比例
        random_seed: 随机种子（用于 train/val 划分）
        persistent: 是否使用 PersistentDataset 缓存预处理结果
        cache_dir: 缓存目录（空字符串则自动设为 root_dir/cache_paired）
        num_workers: DataLoader 的 worker 数量

    Returns:
        DataLoader 对象，每个 batch 包含 'ct', 'mr', 'label', 'name'
    """
    if cache_dir == "":
        cache_dir = os.path.join(root_dir, 'cache_paired')
    os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 可缓存部分（确定性变换）：加载 → spacing → orientation → 强度归一化 → crop → pad
    # ct 和 mr 作为同一 dict 的不同 key 一起处理，确保 spacing/orientation/crop 对齐
    # ------------------------------------------------------------------
    cache_transforms = Compose([
        # 加载图像（确保通道优先，仅加载图像数据）
        LoadImaged(keys=["ct", "mr", "label"], ensure_channel_first=True, image_only=True),
        # 统一方向（RAS）
        Orientationd(keys=["ct", "mr", "label"], axcodes="RAS"),
        # CT 边缘连通零值背景 → -2048（先做再重采样，输出保留 MetaTensor）
        ReplaceZeroEdgeBackgroundd(keys=["ct"], black_value=-2048),
        # 统一间距（CT: 1.5mm, MR: 1.5mm, Label: 2.0mm）
        Spacingd(
            keys=["ct", "mr", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        # CT 窗宽归一化（骨肿瘤典型窗：-125~275 HU）
        ScaleIntensityRanged(
            keys=["ct"],
            a_min=-125,
            a_max=275,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # MR 零均值归一化
        # NormalizeIntensityd(keys=["mr"], nonzero=True, channel_wise=True),
        ScaleIntensityRangePercentilesd(
            keys=["mr"], 
            lower=0.5, 
            upper=99.5, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True, 
            channel_wise=True
        ),
        # 以 CT 为前景源裁剪背景，再 pad，使 RandCrop 在有效解剖区域内采样
        CropForegroundd(keys=["ct", "mr", "label"], source_key="ct"),
        SpatialPadd(keys=["ct", "mr", "label"], spatial_size=roi_size, mode='constant'),
    ])

    # ------------------------------------------------------------------
    # 随机增强部分（每次 epoch 重新执行，不缓存）
    # 空间增强：keys 同时包含 ct/mr/label → MONAI 用同一随机状态变换所有 key
    # 强度增强：ct 和 mr 分别指定，独立采样
    # ------------------------------------------------------------------
    if phase == 'train':
        aug_transforms = Compose([
            # 中心裁剪 —— ct/mr/label 取体积中心 roi_size 区域
            # CenterSpatialCropd(keys=["ct", "mr", "label"], roi_size=roi_size),
            RandCropByPosNegLabeld(
                keys=["ct", "mr", "label"], 
                label_key="label", 
                spatial_size=roi_size, 
                pos=3,                       # 肿瘤采样权重提升
                neg=1,                       # 背景采样权重
                num_samples=train_num_samples, # 使用函数参数传进来的变量 (默认是2)
                image_key="ct",              # 结合 threshold 使用
                image_threshold=0            # 强制系统不要去采纯黑的空气区域
            ),
            # 空间增强 —— ct/mr/label 共享同一随机 affine 参数（旋转/缩放）
            RandAffined(
                keys=["ct", "mr", "label"],
                mode=("bilinear", "bilinear", "nearest"),
                prob=0.8,
                spatial_size=roi_size,
                rotate_range=(0, 0, np.pi / 30),
                scale_range=(0.1, 0.1, 0.1),
            ),
            # 显式关闭 flip，避免破坏 left/right 等解剖侧别语义。
            # 版本 A 的空间先验依赖病例文本中的方向词，因此这里不做任何翻转增强。
            # 强度增强 —— CT 独立（HU 偏移）
            RandShiftIntensityd(keys=["ct"], offsets=0.10, prob=0.50),#offset=0.10 表示在原有强度基础上随机增加或减少最多10%的强度值，prob=0.50 表示有50%的概率应用这个增强
            # 强度增强 —— MR 独立（缩放 + 偏移）
            RandScaleIntensityd(keys=["mr"], factors=0.1, prob=0.5),#factors=0.1 表示在原有强度基础上随机缩放最多10%，prob=0.5 表示有50%的概率应用这个增强
            RandShiftIntensityd(keys=["mr"], offsets=0.1, prob=0.5),
            ToTensord(keys=["ct", "mr", "label"]),
        ])
    else:# 验证集：仅转换为 tensor，无随机增强，确保结果稳定可复现
        aug_transforms = Compose([
            ToTensord(keys=["ct", "mr", "label"]),
        ])

    data_dicts = get_paired_data_dicts(
        root_dir=root_dir,
        phase=phase,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    print(f"[Paired Loader] {phase.upper()} - {len(data_dicts)} cases, roi={roi_size}, "
          f"crop={'CenterSpatialCrop' if phase == 'train' else 'none'}")

    if persistent and phase == 'train':
        # 训练集：只缓存确定性预处理，每次动态应用随机增强
        cache_subdir = os.path.join(cache_dir, f'paired_{phase}_cache')
        cache_dataset = PersistentDataset(
            data=data_dicts,
            transform=cache_transforms,
            cache_dir=cache_subdir,
        )

        class _AugDataset(TorchDataset):
            def __init__(self, ds, aug):
                self.ds = ds
                self.aug = aug

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                # 从缓存取预处理结果，再做随机增强
                return self.aug(self.ds[idx])

        dataset = _AugDataset(cache_dataset, aug_transforms)

    elif persistent and phase == 'val':
        # 验证集：完整 transform 缓存（无随机增强，幂等）
        full_val_transforms = Compose([cache_transforms, aug_transforms])
        cache_subdir = os.path.join(cache_dir, f'paired_{phase}_cache')
        dataset = PersistentDataset(
            data=data_dicts,
            transform=full_val_transforms,
            cache_dir=cache_subdir,
        )
    else:
        full_transforms = Compose([cache_transforms, aug_transforms])
        dataset = CacheDataset(data=data_dicts, transform=full_transforms, cache_rate=1.0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=num_workers if phase == 'train' else 0,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    return loader
#配对数据增强的transform获取函数（可选，提供更灵活的接口）

def get_transforms_for_modality(modality: str, phase: str, roi_size: tuple, train_num_samples: int = 2):
    """
    获取指定模态和阶段的transform
    
    Args:
        modality: 'CT' 或 'MR'
        phase: 'train' 或 'val'
        roi_size: ROI大小 (D, H, W)
        train_num_samples: 训练时每个样本的随机crop数量
    
    Returns:
        (cache_transform, aug_transform) 或 (full_transform, None)
    """
    if modality == 'CT':
        if phase == 'train':
            # CT训练变换 - 分为可缓存部分和随机增强部分
            ct_train_transforms_cache = Compose([
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-125,
                    a_max=275,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
            ])
            
            ct_train_transforms_aug = Compose([
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=roi_size,
                    pos=1,
                    neg=1,
                    num_samples=train_num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0,
                    spatial_size=roi_size,
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)
                ),
                ToTensord(keys=["image", "label"]),
            ])
            return ct_train_transforms_cache, ct_train_transforms_aug
        else:  # val
            ct_val_transforms = Compose([
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-125,
                    a_max=275,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
                ToTensord(keys=["image", "label"]),
            ])
            return ct_val_transforms, None
    else:  # MR
        if phase == 'train':
            mr_train_transforms_cache = Compose([
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
            ])
            
            mr_train_transforms_aug = Compose([
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=roi_size,
                    pos=1,
                    neg=1,
                    num_samples=train_num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                ToTensord(keys=["image", "label"]),
            ])
            return mr_train_transforms_cache, mr_train_transforms_aug
        else:  # val
            mr_val_transforms = Compose([
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
                ToTensord(keys=["image", "label"]),
            ])
            return mr_val_transforms, None
