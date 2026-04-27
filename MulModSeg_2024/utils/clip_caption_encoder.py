# 用于带 caption 训练时用 CLIP 编码文本，供 train 与 enhanced_validation 共用
import torch


def get_clip_text_encoder(device):
    """加载 CLIP 用于编码 caption，返回 model。"""
    try:
        import clip
        model, _ = clip.load('ViT-B/32', device=device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(
            f"CLIP 加载失败，带 caption 训练需安装: pip install git+https://github.com/openai/CLIP.git ；错误: {e}"
        )


def encode_captions_clip(captions, clip_model, device, max_len=77):
    """
    将 list of caption 字符串编码为 [B, 512]。空字符串用占位符 "tumor" 替代。
    """
    import clip
    if not captions:
        return None
    safe = [c.strip() if (c and str(c).strip()) else "tumor" for c in captions]
    tokens = clip.tokenize(safe, truncate=True, context_length=max_len).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens).float()
    return feats
