import json
from collections import defaultdict

linhas = []
for modo in ("motor", "original"):
    try:
        with open(f"test_100_{modo}.jsonl", encoding="utf-8") as f:
            linhas += [json.loads(l) for l in f]
    except FileNotFoundError:
        pass
print("total:", len(linhas))
for modo in ("motor", "original"):
    m = [l for l in linhas if l["modo"] == modo]
    ok = sum(1 for l in m if l["acertou"])
    t = sum(l["tempo_s"] for l in m) / len(m)
    print(f"{modo}: {ok}/{len(m)} = {100 * ok / len(m):.0f}%  "
          f"tempo medio {t:.1f}s")
    por_caso = defaultdict(list)
    for l in m:
        por_caso[l["caso"]].append(l["acertou"])
    print("  sempre:", sorted([c for c, v in por_caso.items() if all(v)]))
    print("  nunca:", sorted([c for c, v in por_caso.items() if not any(v)]))
    print("  parcial:", {c: f"{sum(v)}/5" for c, v in sorted(por_caso.items())
                         if any(v) and not all(v)})
