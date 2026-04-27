#!/usr/bin/env python
"""
基于 prompts/prompts.csv 中每个患者的 caption，生成患者个性化的 CLIP 文本嵌入。

输出格式 (patient_embeddings.pth):
    'embeddings': tensor(N, 512)  # N 是所有患者、所有模态的总数
    'id_map': list[dict]         # 映射列表，每个元素对应 embeddings 中的一行
        [{'patient_id': '10075157', 'modality': 'CT'},
         {'patient_id': '10075157', 'modality': 'MR'},
         {'patient_id': '10180747', 'modality': 'CT'},
         ...
        ]
"""
import os
import re
import pandas as pd
import torch
import clip
import argparse # 引入 argparse

MODALITY_TEXT = {
    'CT': 'computed tomography imaging',
    'MR': 'magnetic resonance imaging',
}


def normalize_patient_id(reg_id: str) -> str:
    """与 dataloader_bone_tumor.py 中相同的 ID 提取逻辑。"""
    m = re.search(r'\d+', str(reg_id))
    return m.group() if m else str(reg_id)


def build_prompt(caption: str, modality: str) -> str:
    """构建 CLIP 文本提示。"""
    return f"A {MODALITY_TEXT[modality]} showing {caption}."


def main(csv_input_path: str, output_path: str):
    """
    主函数，接收输入 CSV 路径和输出文件路径。
    """
    if not os.path.exists(csv_input_path):
        raise FileNotFoundError(f"prompts.csv not found at: {csv_input_path}")

    df = pd.read_csv(csv_input_path)
    print(f"Loaded {len(df)} rows from {csv_input_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()

    all_embeddings = []
    id_map = []

    processed_pids = set() # 用于记录已经处理过的患者ID

    with torch.no_grad():
        # 确保我们遍历的是 DataFrame 的行，而不是潜在的重复 patient_id
        # 假设 reg_id 唯一标识一个患者的 caption
        # 如果一个 patient_id 有多个 reg_id 且 caption 不同，这里只会取第一个 reg_id 对应的 caption
        # 如果需要处理所有 caption，需要调整此处的去重逻辑
        unique_rows = df.drop_duplicates(subset=['reg_id'])

        for _, row in unique_rows.iterrows():
            pid     = normalize_patient_id(row['reg_id'])
            caption = str(row['caption'])

            # 跳过已经处理过的患者ID，避免重复生成
            if pid in processed_pids:
                continue

            for modality in ['CT', 'MR']:
                prompt = build_prompt(caption, modality)
                tokens = clip.tokenize([prompt]).to(device)
                emb    = clip_model.encode_text(tokens)[0].float().cpu()  # (512,)

                all_embeddings.append(emb)
                id_map.append({'patient_id': pid, 'modality': modality})

                print(f"  [{pid}] {modality}: \"{prompt}\"")

            processed_pids.add(pid) # 标记该患者ID已处理

    # 将所有嵌入合并成一个 Tensor
    if all_embeddings:
        embeddings_tensor = torch.stack(all_embeddings)
    else:
        embeddings_tensor = torch.empty(0, 512) # 如果没有数据，创建一个空的 Tensor

    output_data = {
        'embeddings': embeddings_tensor,
        'id_map': id_map
    }

    torch.save(output_data, output_path)
    print(f"\nSaved {len(id_map)} embeddings ({embeddings_tensor.shape[0]} total) → {output_path}")
    print(f"Mapping saved to {output_path} with keys 'embeddings' and 'id_map'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate personalized CLIP text embeddings for patients.")
    parser.add_argument(
        '--csv_input',
        type=str,
        required=True,
        help="Path to the prompts.csv file."
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help="Path to save the generated patient_embeddings.pth file."
    )

    args = parser.parse_args()

    main(args.csv_input, args.output_file)