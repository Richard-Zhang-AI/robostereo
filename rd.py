import json
from pathlib import Path

root = Path("assets/action_conditioned/geometry/bridge/annotation/test")
count = 0
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    changed = False
    for v in data.get("videos", []):
        if isinstance(v, dict) and "video_path" in v:
            vp = v["video_path"]
            if vp.endswith("geometry.safetensors"):
                continue
            new_vp = vp.replace("rgb.mp4", "geometry.safetensors")
            if new_vp != vp:
                v["video_path"] = new_vp
                changed = True
    if changed:
        path.write_text(json.dumps(data, indent=4))
        count += 1

print(f"updated {count} json files")