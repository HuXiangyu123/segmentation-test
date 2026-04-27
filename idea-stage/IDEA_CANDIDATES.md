# Idea Candidates

| # | Idea | Pilot Signal | Novelty | Reviewer Score | Status |
|---|------|-------------|---------|---------------|--------|
| 1 | BiomedCLIP + Progressive Freeze | Pending | ✅ Confirmed | 7/10 | RECOMMENDED |
| 2 | Decoder CBAM/SE Attention | Pending | ✅ Confirmed | 6/10 | BACKUP |
| 3 | LLM-Enhanced Text Descriptions | Pending | ✅ Confirmed | 6/10 | DEFERRED (API key needed) |
| 4 | Boundary Dice Loss | Pending | ❌ None | 5/10 | QUICK WIN |
| 5 | Modality-Aware Routing Bias | Pending | ⚠️ Weak | 5/10 | QUICK WIN |

## Active Idea: #1 — BiomedCLIP + Progressive Freeze
- **Hypothesis**: Medical-domain text embeddings + progressive encoder unfreezing stabilizes text-guided MoE routing and improves pelvis Dice by 3-5%
- **Key evidence**: BiomedCLIP outperforms generic CLIP by 6-15% on PubMed medical tasks (Zhang et al., 2023); progressive freezing reduces small-sample overfitting (validated across multiple medical imaging studies)
- **Next step**: Run P0 quick wins → download BiomedCLIP weights → run E1a → run E1c combined
- **⚠️ Manual**: BiomedCLIP weight download (~800MB), SwinUNETR pretrained weight verification
