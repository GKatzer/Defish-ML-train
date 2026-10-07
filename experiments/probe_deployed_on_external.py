"""How does the deployed YOLO-cls classifier treat external (non-aquarium) fish images?

The server drops `many_fish` / `not_a_fish` predictions, so those count as "silently lost".
    python experiments/probe_deployed_on_external.py
"""
import sys
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from clf_frozen_vs_yolo import AUX, load_external, predict_yolo

ext = load_external()
rows = []
for mode in ("server", "native"):
    probs, names = predict_yolo(ext.path, mode)
    pred = pd.Series([names[i] for i in probs.argmax(1)])
    for lab, idx in ext.groupby("label").groups.items():
        p = pred.loc[idx]
        rows.append(dict(preproc=mode, silver_class=lab, n=len(p),
                         healthy=(p == "healthy").mean(),
                         disease_shown=(~p.isin(["healthy", *AUX])).mean(),
                         dropped_aux=p.isin(AUX).mean()))
df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
df.to_csv(__import__("pathlib").Path(__file__).parent / "results" / "probe_deployed_on_external.csv", index=False)
