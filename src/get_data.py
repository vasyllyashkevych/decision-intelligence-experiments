"""Download the PROMISE/Jureczko multi-release defect datasets (class-level CK + OO metrics, bug counts).
Source: https://github.com/klainfo/DefectData  (inst/extdata/terapromise/ck)
Original data: Jureczko & Madeyski (2010), PROMISE repository. Data are NOT redistributed here; cite the original authors.
"""
import hashlib, json, sys, urllib.request
from pathlib import Path
BASE = "https://raw.githubusercontent.com/klainfo/DefectData/master/inst/extdata/terapromise/ck/"
FILES = """ant-1.3 ant-1.4 ant-1.5 ant-1.6 ant-1.7 camel-1.0 camel-1.2 camel-1.4 camel-1.6 ivy-1.1 ivy-1.4 ivy-2.0
jedit-3.2 jedit-4.0 jedit-4.1 jedit-4.2 jedit-4.3 log4j-1.0 log4j-1.1 log4j-1.2 lucene-2.0 lucene-2.2 lucene-2.4
poi-1.5 poi-2.0 poi-2.5 poi-3.0 synapse-1.0 synapse-1.1 synapse-1.2 velocity-1.4 velocity-1.5 velocity-1.6
xalan-2.4 xalan-2.5 xalan-2.6 xalan-2.7 xerces-1.2 xerces-1.3 xerces-1.4""".split()
out = Path(__file__).resolve().parents[1] / "data" / "raw"
out.mkdir(parents=True, exist_ok=True)
manifest = {}
for f in FILES:
    p = out / f"{f}.csv"
    if not p.exists():
        with urllib.request.urlopen(BASE + f + ".csv", timeout=60) as r:
            p.write_bytes(r.read())
    manifest[f] = hashlib.sha256(p.read_bytes()).hexdigest()
(out.parent / "MANIFEST.sha256.json").write_text(json.dumps(manifest, indent=1))
print(len(manifest), "files; manifest written")
