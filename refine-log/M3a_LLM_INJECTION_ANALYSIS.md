# M3a: LLM Lexicon Construction Analysis

**Date**: 2026-05-02  
**Status**: Revised after code-path, metadata, and literature audit  
**Question**: How should GPT-5.4 or a strong vLLM-served instruct model be used to improve `case_text_embedding` for MulModSeg, given that the previous vocabulary strategy produced little gain?

---

## 1. Bottom-Line Conclusion

The previous M3a idea was too optimistic about what the LLM could add.

The code audit shows that the router ultimately consumes **one offline-computed 512-d text vector per case/modality**, fused as:

`static_tumor + alpha * case_text_embedding -> Linear(512->256) -> DynamicSemanticHead`

This means a large JSON schema does **not** become a richer token-level conditioning signal inside the model. It is still compressed to a single vector before routing.

At the same time, the available structured data contains only a small number of verified fields:

- site / side / compartment from `prompts_updated.csv`
- tumor size bucket from `tumor_each_case_proportion.csv`
- batch / file-variant from the registration log

It does **not** contain verified MR signal, enhancement, cortical destruction, periosteal reaction, or matrix pattern. Asking an LLM to fill those fields mostly injects hallucinated noise.

**Therefore the feasible M3a direction is not "let the LLM write a richer fake report".**  
It is:

1. **Plan A: Controlled descriptor bank**
   Use GPT-5.4 or a strong vLLM model to generate multiple short, discriminative, observation-bounded descriptors, then filter and average them with BiomedCLIP.
2. **Plan B: Mask-grounded attribute lexicon**
   First extract real morphology/shape bins from the tumor masks, then use the LLM only to normalize those verified attributes into compact biomedical phrases.

Plan A is lower effort and should be run first.  
Plan B is more work, but it is the only path that can introduce meaningful **within-site** variation without hallucination.

---

## 2. Current Pipeline Audit

### 2.1 What is generated today

`MulModSeg_2024/text_embedding/generate_bone_tumor_class_embeddings.py` builds one sentence per case:

```text
A multi-modal CT and MR scan showing a single {size_en} osteosarcoma located in the {site_en}.
```

That sentence is encoded by CLIP ViT-B/32 and saved as:

```python
{
  "embeddings": Tensor[N_cases, 2, 512],   # modality order: [MR, CT]
  "id_map": list[dict],
  "modality_order": ["MR", "CT"],
}
```

### 2.2 How it enters MulModSeg

From `MulModSeg.py::_get_fused_text_feature`:

1. Load static tumor embedding from `--word_embedding`
2. Load patient-level case embedding from `CaseTextEmbeddingStore`
3. Normalize and fuse:
   `static_tumor + case_text_alpha * case_tumor`
4. Project to 256-dim
5. Feed to `DynamicSemanticHead` router

### 2.3 Hard implication

Any M3a plan must produce a **stable single-vector representation** offline.

Without changing the model code, the following are **not** first-order gains:

- long free-form reports
- large nested JSON schemas
- token-level prompt chaining
- different prompt variants that are never aggregated or selected

The leverage point is the **geometry of the final 512-d vector**, not the apparent richness of the raw text.

---

## 3. Available Information vs. Missing Information

### 3.1 Verified fields available now

| Field | Source | Notes |
|------|--------|-------|
| `bone/site` | `prompts_updated.csv` | femur / pelvis with left-right refinement |
| `side` | `prompts_updated.csv` | left / right / bilateral / unspecified |
| `compartment` | `prompts_updated.csv` | mostly distal; one proximal case |
| `size_category` | `tumor_each_case_proportion.csv` | Small / Medium / Large |
| `tumor_voxels`, `proportion` | `tumor_each_case_proportion.csv` | can be reused for finer binning |
| `batch` | prompt metadata | useful as cohort/source tag, not anatomy itself |

### 3.2 Fields not available in structured metadata

- T1/T2 signal
- enhancement
- lytic/blastic matrix
- cortical erosion / penetration
- periosteal reaction
- soft-tissue mass
- pathological fracture
- shape / margin labels from radiologist review

### 3.3 Why the original schema underperformed

The previous plan asked the LLM to output radiology-style attributes that are not observed anywhere in the data pipeline.

That creates three problems:

1. **Hallucination noise**: generated fields vary because of language-model priors, not because of the patient.
2. **Uniform pseudo-signal**: since all cases are osteosarcoma, many fields collapse to nearly the same wording.
3. **Compression loss**: even if the JSON looks detailed, it is still collapsed into a single embedding vector before routing.

---

## 4. Additional Constraints from the Text Encoders

### 4.1 BiomedCLIP

Local config in `MulModSeg_2024/text_embedding/biomedclip/open_clip_config.json` shows:

- text encoder: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
- embedding dim: `512`
- `context_length: 256`

### 4.2 MedSigLIP

Local model card in `MulModSeg_2024/text_embedding/medsiglip/README.md` states:

- medical image-text shared embedding space
- text context length: `64`

### 4.3 Practical consequence

Even with BiomedCLIP, a single long report is not ideal. For MedSigLIP it is even less suitable.

The safer pattern is:

- generate **multiple short descriptors**
- encode them separately
- normalize and aggregate offline

This is also closer to what prompt-ensemble papers do in practice.

---

## 5. Paper-Backed Design Principles

### 5.1 CuPL: LLMs help when they generate discriminative descriptions, not just longer text

CuPL ("What does a platypus look like?") shows that prompting an LLM to produce descriptive sentences per class can outperform plain hand-written templates for CLIP-style zero-shot classification. The useful idea here is not "longer is better"; it is:

- produce **multiple descriptions**
- make them **class-discriminative**
- aggregate them as a prompt set

### 5.2 SuS-X: prompt ensembling remains useful even on top of CuPL-style generation

The SuS-X supplementary material explicitly keeps several short pre-prompts such as:

- `a photo of a {}.`
- `a blurry photo of a {}.`
- `a black and white photo of a {}.`

and averages the resulting embeddings. It also notes that **prompt ensembling and CuPL are complementary**.  
This is directly relevant to MulModSeg: one single case sentence is a weak baseline; a small bank of short descriptors is a more defensible design.

### 5.3 BiomedCLIP: domain-specific biomedical text space is the right default

BiomedCLIP is pretrained on PMC-15M biomedical figure-caption pairs and uses a biomedical BERT text encoder. For M3a, this matters more than fancy LLM wording because the downstream success depends on whether the final descriptors live in a biomedical embedding space rather than general CLIP space.

### 5.4 MedKLIP: descriptions should be grounded in structured knowledge, not invented

MedKLIP replaces raw disease names with description-based medical entities and reports large gains in zero-shot settings.  
The transferable lesson for this project is:

- use the LLM to **translate structured facts into standardized medical phrases**
- do **not** ask it to guess invisible findings

This supports Plan B especially strongly.

### 5.5 Shape/radiomics literature: if we want more case-specific signal, measure it first

The PyRadiomics paper is the practical reminder here: volume, elongation, compactness, and surface irregularity are measurable from masks and reproducible.  
If we need more text variation than site + size can provide, the right next move is to extract shape bins from the segmentation masks and verbalize them, not to fabricate MR findings.

---

## 6. Two Feasible Plans

### 6.1 Plan A: Controlled Descriptor Bank

**Goal**: Improve case-text geometry without changing model code and without introducing hallucinated findings.

#### Core idea

Generate a small descriptor bank for each **canonical metadata key**, not for each patient from scratch.

Recommended canonical key:

`(bone, side, compartment, size_category)`

This maps the current dataset to a small set of real groups such as:

- left distal femur + small
- right distal femur + medium
- left pelvis + large
- bilateral pelvis + medium

#### LLM role

Use GPT-5.4 or a strong OpenAI-compatible vLLM endpoint to generate **8-16 short English descriptors** per canonical key.

The prompt must be observation-bounded:

- only use provided fields
- no MR signal
- no enhancement
- no cortical invasion
- no matrix pattern
- no prognosis

#### Descriptor families

Generate descriptors in three families:

1. **Anatomy-heavy**
   Example: `left distal femoral osteosarcoma`
2. **Burden-heavy**
   Example: `medium-volume tumor centered in the distal femur`
3. **Clinical-normalized**
   Example: `osteosarcoma arising in the left distal femur with medium tumor burden`

Each descriptor should be short, ideally under 24 tokens.

#### Filtering rules

After generation:

1. Reject any descriptor containing unobserved findings:
   - `T1`, `T2`, `enhancement`, `marrow edema`, `periosteal`, `cortical destruction`, etc.
2. Remove near-duplicates by cosine similarity in BiomedCLIP space
3. Keep top `K=6` or `K=8` descriptors per canonical key
4. Normalize each descriptor embedding, then average

#### Offline aggregation

Because the current code expects one vector per case/modality, store:

```python
final_case_embedding = mean(normalize(E_1), ..., normalize(E_K))
```

and write back to the current `.pth` format:

```python
{
  "embeddings": Tensor[N_cases, 2, 512],
  "id_map": ...,
  "modality_order": ["MR", "CT"]
}
```

Cases sharing the same canonical key can share the same descriptor bank, which improves consistency and lowers API cost.

#### Why Plan A is plausible

- fits the current model without architecture changes
- supported by CuPL + SuS-X style prompt ensembling
- avoids hallucinated radiology findings
- easy to cache and reproduce

#### Main limitation

Plan A still derives almost all signal from **site + size**, so its gain may plateau quickly.

#### Recommended ablation for Plan A

| Exp ID | Description |
|-------|-------------|
| `M3aA_1_single_biomedclip` | Replace CLIP caption with BiomedCLIP single sentence only |
| `M3aA_2_descbank_k4` | 4 descriptors per canonical key, averaged |
| `M3aA_3_descbank_k8` | 8 descriptors per canonical key, averaged |
| `M3aA_4_descbank_k8_seed3` | repeat best Plan A setup for seed variance |

#### Success criteria

- pelvis Dice `>= +0.5` over M2c
- router seed variance reduced
- pelvis/femur centroid cosine gap increases in BiomedCLIP space

If all three fail, stop Plan A and move to Plan B.

---

### 6.2 Plan B: Mask-Grounded Attribute Lexicon

**Goal**: Introduce real case-specific variation instead of language-model variation.

#### Core idea

Use the tumor masks to extract measurable morphology, discretize them into a small ontology, and let the LLM only rewrite those verified bins into biomedical descriptors.

#### New signal sources

From `tumor_each_case_proportion.csv` and mask files:

- tumor volume
- tumor-to-image proportion
- bbox extent ratio `(dx, dy, dz)`
- elongation / flatness
- compactness / sphericity
- surface irregularity proxy

From metadata:

- bone
- side
- compartment
- batch

#### Closed-vocabulary attribute schema

Suggested bins:

```json
{
  "bone": "femur | pelvis",
  "side": "left | right | bilateral | unspecified",
  "compartment": "distal | proximal | unspecified",
  "burden": "small | medium | large | very_large",
  "elongation": "focal | elongated | diffuse",
  "compactness": "compact | intermediate | irregular",
  "extent_3d": "localized | moderately_extended | widely_extended"
}
```

Every field must come from a measured value or verified metadata field.

#### LLM role

The LLM does **not** infer new findings.  
It only converts the attribute dictionary into 4-8 short biomedical descriptions.

Example input:

```json
{
  "bone": "pelvis",
  "side": "left",
  "compartment": "unspecified",
  "burden": "large",
  "elongation": "diffuse",
  "compactness": "irregular",
  "extent_3d": "widely_extended"
}
```

Example output style:

- `large irregular left pelvic osteosarcoma`
- `widely extended osteosarcoma centered in the left pelvis`
- `diffuse high-burden pelvic bone tumor with irregular contour`

#### Why Plan B is stronger than the previous M3a idea

The extra fields are no longer hallucinated. They are derived from the mask, so the text vector can reflect real intra-class differences, especially among pelvis cases.

#### Why Plan B is slower

It requires:

1. shape-feature extraction script
2. discretization thresholds
3. QC on a few cases
4. one extra text-generation pass

#### Recommended ablation for Plan B

| Exp ID | Description |
|-------|-------------|
| `M3aB_1_shape_template` | no LLM; template directly from measured bins |
| `M3aB_2_shape_llm_restyle` | same bins, LLM rewrites into short biomedical phrases |
| `M3aB_3_shape_llm_best_seed3` | repeat best Plan B setup for seed variance |

#### Success criteria

- pelvis Dice `>= +1.0` over M2c
- improvement over Plan A best
- better within-site embedding spread without exploding seed variance

---

## 7. Recommended Run Order

### Stage 1: minimal-risk text upgrade

1. `M2a`: single-sentence BiomedCLIP swap
2. `M3aA_2_descbank_k4`
3. `M3aA_3_descbank_k8`

If no consistent gain appears here, do **not** spend time on larger LLM schemas.

### Stage 2: real attribute injection

4. extract shape attributes from masks
5. `M3aB_1_shape_template`
6. `M3aB_2_shape_llm_restyle`

### Stage 3: lock the best text route

7. compare best of:
   - baseline CLIP caption
   - single-sentence BiomedCLIP
   - Plan A best
   - Plan B best
8. only then combine with M3b attention plug-ins

---

## 8. Provider Strategy: GPT-5.4 vs. vLLM

This plan should be provider-agnostic.

Recommended interface:

- OpenAI-compatible chat completion API
- deterministic settings:
  - `temperature=0`
  - `top_p=1`
  - fixed seed if backend supports it
  - JSON output with schema validation

Recommended usage:

- **GPT-5.4**: best first pass for generating the canonical descriptor bank
- **vLLM-served instruct model**: good for local replication once the schema and prompt are frozen

Important: the backend model should be treated as a **descriptor generator**, not as a pseudo-radiologist.

---

## 9. What Should Be Removed from the Old M3a Proposal

The following should be explicitly deprioritized:

- large fixed JSON schema with MR signal fields
- differential diagnosis ranking for a cohort where diagnosis is uniform
- templated cortical involvement defaults
- case-specific free-text radiology reports generated from location + size only

These add wording, but not verified information.

---

## 10. Concrete Deliverables

### Plan A deliverables

- `refine-log/llm_descriptor_bank_planA.json`
- `refine-log/llm_descriptor_bank_planA.filtered.json`
- `MulModSeg_2024/text_embedding/case_text_embeddings_planA_biomedclip.pth`

### Plan B deliverables

- `refine-log/mask_shape_attributes_planB.csv`
- `refine-log/llm_descriptor_bank_planB.json`
- `MulModSeg_2024/text_embedding/case_text_embeddings_planB_biomedclip.pth`

---

## 11. References and Transferable Lessons

1. **CuPL / customized prompts for CLIP**  
   LLM-generated class descriptions can outperform plain prompt templates when they are short, discriminative, and aggregated.  
   https://openreview.net/forum?id=3ly9cG9Ql9h

2. **SuS-X supplementary**  
   Prompt ensembling with several short pre-prompts is still useful and complementary to CuPL-style descriptors.  
   https://openaccess.thecvf.com/content/ICCV2023/supplemental/Udandarao_SuS-X_Training-Free_Name-Only_ICCV_2023_supplemental.pdf

3. **BiomedCLIP**  
   Domain-specific biomedical text/image pretraining is a better default than general CLIP for medical vocabulary encoding.  
   https://huggingface.co/papers/2303.00915

4. **MedKLIP**  
   Converting structured medical entities into grounded descriptions is more defensible than using raw names alone, and much safer than inventing unobserved findings.  
   https://arxiv.org/abs/2301.02228

5. **PyRadiomics**  
   If extra intra-class signal is needed, measure morphology from masks first, then verbalize it.  
   https://pubmed.ncbi.nlm.nih.gov/28901788/

---

## 12. Final Recommendation

For this project, the best near-term path is:

1. **Run Plan A first** with BiomedCLIP and a controlled descriptor bank
2. If gains are weak, **do not scale up the schema**
3. Move to **Plan B**, where the LLM only verbalizes measured shape attributes

That is the cleanest way to use GPT-5.4 or a strong vLLM model to improve the text prior without repeating the previous low-yield vocabulary expansion attempt.
