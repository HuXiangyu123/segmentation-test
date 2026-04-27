import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

def classify_tumor_size(proportion, threshold_small, threshold_large):
    """根据占比划分大、中、小肿瘤"""
    if proportion < threshold_small:
        return 'Small'
    elif proportion > threshold_large:
        return 'Large'
    else:
        return 'Medium'

def calculate_nested_tumor_proportions(base_dir, output_csv, threshold_small=0.01, threshold_large=0.05):
    """
    遍历多级嵌套文件夹，匹配 GT 文件，计算占比并进行三分类
    """
    results = []
    
    print(f"{'Case ID':<15} | {'Batch':<10} | {'Proportion (%)':<15} | {'Size Class':<10}")
    print("-" * 65)
    
    for root, dirs, files in os.walk(base_dir):
        case_id_from_dir = os.path.basename(root)
        
        for filename in files:
            # 1. 必须是 nii 或 nii.gz 结尾
            if not (filename.endswith('.nii.gz') or filename.endswith('.nii')):
                continue
            
            # 2. 【核心修改】排除带有 _ct, _mr, _reg 等后缀的干扰文件
            # 如果文件名里包含这些字符串，直接跳过
            lower_name = filename.lower()
            if "_ct" in lower_name or "_mr" in lower_name or "_reg" in lower_name:
                continue
            
            # 去掉后缀，提取潜在的 case_id
            if filename.endswith('.nii.gz'):
                potential_case_id = filename[:-7]
            else:
                potential_case_id = filename[:-4]

            is_valid_gt = False
            
            # 情况1: 常规嵌套文件夹 (文件夹名 = 文件名前缀)
            # 有时文件名带 N (如 N12896530)，但文件夹名不带 (如 12896530)
            if potential_case_id == case_id_from_dir or potential_case_id.replace('N', '') == case_id_from_dir.replace('N', ''):
                is_valid_gt = True
            # 情况2: 散装存放 (比如直接在 /上海市一/ 文件夹下)
            # 只要是长串数字，或者是 'N' 开头带长数字，我们就认为它是合法的 GT
            elif potential_case_id.isdigit() or (potential_case_id.startswith('N') and potential_case_id[1:].isdigit()):
                 is_valid_gt = True

            if is_valid_gt:
                file_path = os.path.join(root, filename)
                batch_name = os.path.basename(os.path.dirname(root))
                
                # 处理直接挂在 "上海市一" 下的文件
                if os.path.dirname(root) == base_dir:
                     batch_name = os.path.basename(root)

                case_id = potential_case_id
                
                try:
                    gt_img = sitk.ReadImage(file_path)
                    gt_array = sitk.GetArrayFromImage(gt_img)
                    
                    tumor_voxels = np.sum(gt_array > 0)
                    total_voxels = gt_array.size
                    
                    proportion = tumor_voxels / total_voxels if total_voxels > 0 else 0
                    proportion_percent = proportion * 100
                    
                    size_class = classify_tumor_size(proportion, threshold_small, threshold_large)
                    
                    print(f"{case_id:<15} | {batch_name:<10} | {proportion_percent:.4f}%{'':<7} | {size_class}")
                    
                    results.append({
                        'Batch': batch_name,
                        'Case_ID': case_id,
                        'Tumor_Voxels': tumor_voxels,
                        'Total_Voxels': total_voxels,
                        'Proportion_Raw': proportion,
                        'Proportion_Percent': f"{proportion_percent:.4f}%",
                        'Size_Category': size_class, 
                        'GT_File_Path': file_path   
                    })
                    
                except Exception as e:
                    print(f"{case_id:<15} | {batch_name:<10} | 读取或计算失败: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=['Batch', 'Case_ID'])
        df.to_csv(output_csv, index=False)
        print("-" * 65)
        print(f"✅ 成功提取 {len(df)} 个 Case，结果已保存至: {output_csv}")
        
        print("\n=== 肿瘤大小分类汇总 ===")
        print(df['Size_Category'].value_counts().to_string())
    else:
        print("未找到任何符合规则的 GT 文件，请检查路径。")

if __name__ == "__main__":
    # ================= 配置区域 =================
    BASE_DIR = r"/root/autodl-tmp/segmentation-test-main/MulModSeg_2024/dataset"
    OUTPUT_CSV_FILE = r"/root/autodl-tmp/segmentation-test-main/tumor_each_case_proportion.csv"  
    
    THRESHOLD_S = 0.02 
    THRESHOLD_L = 0.05  
    # ============================================
    
    calculate_nested_tumor_proportions(BASE_DIR, OUTPUT_CSV_FILE, THRESHOLD_S, THRESHOLD_L)