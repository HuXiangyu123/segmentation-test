# 项目路径与树
- 当前项目路径：`/home/glcuser/projhighcv/bone_tumor`
- 顶层结构（来自当前文件系统）：
```
/home/glcuser/projhighcv/bone_tumor
├── analyze.py
├── audit_ctreg/
│   ├── dataset_audit.csv
│   └── dataset_audit.json
├── dataset/
│   ├── 上海市一/
│   ├── 第1批/
│   ├── 第2批/
│   ├── 第3批/
│   ├── 第4批/
│   ├── 第5批/
│   └── 信息登记_实际生成.xlsx
├── export_dataset_registry_from_fs.py
├── MulModSeg_2024/
├── scan_dataset_ctreg.py
├── viz_case.py
├── .env
└── .git/
```

# dataset 文件夹构成
## 根目录概览
- 批次目录：`第1批`、`第2批`、`第3批`、`第4批`、`第5批`、`上海市一`
- 登记表：`信息登记_实际生成.xlsx`
- 病例目录总数：109（分布在第1-5批），另有 1 个“上海市一”样本直接放在批次根目录下
- 总文件数：393（`.nii.gz` 358，`.nii` 21，`.png` 13，`.xlsx` 1）

## 批次统计（按文件名规则统计：ct/mr/ct_reg/seg）
| 批次 | 病例目录数 | 文件数 | MR | CT_reg | CT_raw | Seg(标注) | PNG | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 上海市一 | 0 | 3 | 1 | 1 | 0 | 1 | 0 | 样本直接在批次根目录（无病例子目录） |
| 第1批 | 28 | 107 | 28 | 28 | 22 | 28 | 1 | 1 个病例含 `DF_CT/` 子目录 |
| 第2批 | 25 | 100 | 25 | 25 | 25 | 25 | 0 | — |
| 第3批 | 25 | 76 | 25 | 25 | 0 | 26 | 0 | 1 个病例出现双标注（`.nii` 与 `.nii.gz`） |
| 第4批 | 17 | 64 | 17 | 17 | 0 | 18 | 12 | 额外存在 `raw/` 与 `截图/` 子目录 |
| 第5批 | 14 | 42 | 14 | 14 | 0 | 14 | 0 | — |

## 结构与命名特征
- 病例层级：`dataset/批次/病例ID/` 为主结构；但“上海市一”样本文件直接放在批次目录。
- 主要模态/文件：
  - MR：文件名包含 `mr`
  - CT_raw：文件名包含 `ct`
  - CT_reg：文件名包含 `ct_reg`
  - Seg：文件名不含 `ct/mr` 的 `.nii/.nii.gz`
- 发现的特殊子目录：`DF_CT/`（仅 1 例）、`raw/`、`截图/`。

## 需要关注的异常文件
- `第3批/11419355/` 同时存在 `11419355.nii` 与 `11419355.nii.gz`（双标注文件）。
- `第4批/11642786/raw/11342786.nii.gz`：位于 `raw/` 子目录，且文件名与病例目录不一致。

# audit_ctreg（既有审计结果）摘要
> 说明：以下统计来自 `audit_ctreg/dataset_audit.csv`（此前脚本输出）。

- 样本总数：110（第1批 28、第2批 25、第3批 25、第4批 17、第5批 14、上海市一 1）
- 选择的 CT 类型：全部为 `ct_reg`（`chosen_ct_type=reg`）
- 缺失情况：
  - CT_raw 缺失 64 个样本
  - MR、Seg、CT_reg 均无缺失
- 主要质量/一致性问题（issues 字段拆分统计）：
  - `CTRAW_MR_SHAPE_MISMATCH`：46
  - `CTRAW_MR_AFFINE_DIFF`：46
  - `CTRAW_SEG_SHAPE_MISMATCH`：44
  - `CTRAW_SEG_AFFINE_DIFF`：44
  - `CTREG_MR_AFFINE_DIFF`：9
  - `MR_SEG_AFFINE_DIFF`：8
  - `MR_SEG_SHAPE_MISMATCH`：2
  - `CTREG_MR_SHAPE_MISMATCH`：2
  - `CTREG_SEG_AFFINE_DIFF`：1
- 分割标签值：109 个样本为 `0,1`，仅 1 个样本为 `0,1,2`（`第2批/10244353`）
- 患者重复出现：3 个患者 ID 在不同批次重复（共 6 条记录）
  - 11084154（第1批、第2批）
  - 11183936（第1批、第2批）
  - 11325149（第2批、第3批）
