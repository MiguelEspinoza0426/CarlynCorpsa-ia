from pathlib import Path
from collections import Counter

raiz = Path("ml/datasets/weldingpe_v1")
nombres = ['Safety Shoes', 'Welding Apron', 'Welding Coverall', 'Welding Gloves', 'Welding Mask']

for split in ["train", "valid", "test"]:
    c = Counter()
    for f in (raiz / split / "labels").glob("*.txt"):
        for linea in f.read_text().splitlines():
            if linea.strip():
                c[int(linea.split()[0])] += 1
    n_img = len(list((raiz / split / "images").glob("*")))
    print(f"\n{split}: {n_img} imágenes")
    for i, n in sorted(c.items()):
        print(f"  {nombres[i]}: {n}")