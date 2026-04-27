# Prompt Template (caption)

## Caption template (EN)
- "{diagnosis_en} in {site_en}"

## Rules
- batch/sample_id are NOT part of caption; they are join keys only.
- Special: '左/右 + 骨远端/股骨远端' -> left/right distal femur.
- Other sites translated by SITE_MAP; unknown Chinese sites are kept as-is and listed in untranslated_sites.csv.

## Drop list (shape mismatch)
- 11687281
- 12298737
