import json
from collections import defaultdict

ls = [json.loads(l)
      for l in open("test_100_resultados.jsonl", encoding="utf-8")]
print("total:", len(ls))
ok = sum(1 for l in ls if l["acertou"])
t = sum(l["tempo_s"] for l in ls) / len(ls)
print(f"original: {ok}/{len(ls)} = {100 * ok / len(ls):.0f}%  "
      f"tempo medio {t:.1f}s")
por_caso = defaultdict(list)
for l in ls:
    por_caso[l["caso"]].append(l["acertou"])
print("sempre:", sorted([c for c, v in por_caso.items() if all(v)]))
print("nunca:", sorted([c for c, v in por_caso.items() if not any(v)]))
print("parcial:", {c: f"{sum(v)}/5" for c, v in sorted(por_caso.items())
                   if any(v) and not all(v)})
# determinismo: repeticoes da mesma imagem dao o mesmo resultado?
from collections import Counter
det = 0
for c, v in por_caso.items():
    outs = Counter()
    for l in ls:
        if l["caso"] == c:
            outs[l["obtido"]] += 1
    if len(outs) > 1:
        det += 1
        print("  instavel:", c, dict(outs))
print("casos instaveis:", det, "/", len(por_caso))
