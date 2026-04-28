"""
Final intersection using meta_xl_tot.csv as bridge (save_uid -> sha256).
"""
import pandas as pd
import ast
import os

BASE = "/local-ssd/TRELLIS.2"

# Load TRELLIS.2 metadata
sketchfab_meta = pd.read_csv(os.path.join(BASE, "trellis.2_data/ObjaverseXL_sketchfab/metadata.csv"))
github_meta = pd.read_csv(os.path.join(BASE, "trellis.2_data/ObjaverseXL_github/metadata.csv"))
sketchfab_sha256 = set(sketchfab_meta['sha256'].dropna().values)
github_sha256 = set(github_meta['sha256'].dropna().values)

# Build sketchfab model_id (lowered) -> sha256
sketchfab_model_id_to_sha256 = {}
for _, row in sketchfab_meta.iterrows():
    fi = str(row.get('file_identifier', ''))
    if 'sketchfab.com/3d-models/' in fi:
        model_id = fi.split('sketchfab.com/3d-models/')[-1].lower()
        sketchfab_model_id_to_sha256[model_id] = row['sha256']

print(f"Sketchfab metadata: {len(sketchfab_meta)}")
print(f"GitHub metadata: {len(github_meta)}")

# Load diffusion_4d lists
with open(os.path.join(BASE, "diffusion_4d_data_list/objaverseXL_curated_uuid_list.txt")) as f:
    uuid_set = set(line.strip() for line in f if line.strip())
with open(os.path.join(BASE, "diffusion_4d_data_list/ObjV1_curated.txt")) as f:
    objv1_list = [line.strip() for line in f if line.strip()]

print(f"uuid_list: {len(uuid_set)}")
print(f"objv1_list: {len(objv1_list)}")

# Build save_uid -> sha256 mapping from meta_xl_tot.csv
print("\nParsing meta_xl_tot.csv (3.6M rows, this may take a while)...")
uid_to_sha256 = {}
chunk_iter = pd.read_csv(os.path.join(BASE, "diffusion_4d_data_list/meta_xl_tot.csv"), chunksize=100000)
for i, chunk in enumerate(chunk_iter):
    for _, row in chunk.iterrows():
        meta_str = str(row.get('meta', ''))
        sha256 = str(row.get('sha256', ''))
        if 'save_uid' in meta_str:
            try:
                meta_dict = ast.literal_eval(meta_str)
                save_uid = meta_dict.get('save_uid', '')
                if save_uid:
                    uid_to_sha256[save_uid] = sha256
            except:
                pass
    if (i + 1) % 10 == 0:
        print(f"  processed {(i+1)*100000} rows, {len(uid_to_sha256)} uid mappings so far...")

print(f"Built save_uid -> sha256 mapping: {len(uid_to_sha256)} entries")

# ===== uuid_list intersection =====
uuid_sha256s = set()
for uid in uuid_set:
    if uid in uid_to_sha256:
        uuid_sha256s.add(uid_to_sha256[uid])

inter_uuid_sketchfab = uuid_sha256s & sketchfab_sha256
inter_uuid_github = uuid_sha256s & github_sha256

# ===== objv1_list intersection (via sketchfab model_id) =====
objv1_matched = [uid for uid in objv1_list if uid.lower() in sketchfab_model_id_to_sha256]

# ===== SUMMARY =====
print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"")
print(f"objaverseXL_curated_uuid_list ({len(uuid_set)} entries):")
print(f"  mapped to sha256 via save_uid: {len(uuid_sha256s)}")
print(f"  ∩ sketchfab: {len(inter_uuid_sketchfab)}")
print(f"  ∩ github:    {len(inter_uuid_github)}")
print(f"  ∩ total:     {len(inter_uuid_sketchfab | inter_uuid_github)}")
print(f"")
print(f"ObjV1_curated ({len(objv1_list)} entries):")
print(f"  ∩ sketchfab (model_id): {len(objv1_matched)}")

# Save intersection lists
os.makedirs(os.path.join(BASE, "claude_tmp"), exist_ok=True)

# uuid_list intersections - save sha256
with open(os.path.join(BASE, "claude_tmp/uuid_sketchfab_intersection_sha256.txt"), 'w') as f:
    for s in sorted(inter_uuid_sketchfab):
        f.write(s + '\n')
with open(os.path.join(BASE, "claude_tmp/uuid_github_intersection_sha256.txt"), 'w') as f:
    for s in sorted(inter_uuid_github):
        f.write(s + '\n')

# objv1 intersection - save uid
with open(os.path.join(BASE, "claude_tmp/objv1_sketchfab_intersection.txt"), 'w') as f:
    for uid in objv1_matched:
        f.write(uid + '\n')

print(f"\nSaved intersection lists to claude_tmp/")
