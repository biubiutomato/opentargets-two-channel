#!/usr/bin/env python3
"""Two-channel (genetics / protein-oriented) re-ranking of Open Targets associations for AD and PD.

Reproduces the main tables and figures of the BIBM 2026 workshop paper (S20209)."""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).resolve().parent
DATA = DIR / "data"
RES = DIR / "results"
RES.mkdir(exist_ok=True)
OUT = DIR / "figures"
OUT.mkdir(exist_ok=True)

QUERY_TMPL = """
query {{
  disease(efoId: "{efo}") {{
    associatedTargets(page: {{index: 0, size: 150}}) {{
      rows {{
        score
        target {{ id approvedSymbol tractability {{ label modality value }} }}
        datatypeScores {{ id score }}
      }}
    }}
  }}
}}
"""

AD_CLASSIC = {
    "APP", "PSEN1", "PSEN2", "APOE", "TREM2", "SORL1", "CLU", "CR1", "PICALM",
    "BIN1", "ABCA7", "ADAM10", "CD33", "SPI1", "PLCG2", "MS4A4A", "MS4A6A",
    "EPHA1", "CD2AP", "INPP5D", "ACE", "GRN", "TMEM106B", "ABI3",
}
AD_DECOY = {
    "ACHE", "BCHE", "GRIN1", "GRIN2A", "GRIN2B", "GRIN2C", "GRIN2D", "GRIN3A",
    "GRIN3B", "SLC6A3", "SLC6A2",
}

PD_CLASSIC = {
    "SNCA", "LRRK2", "GBA", "GBA1", "PRKN", "PINK1", "PARK7", "VPS35", "ATP13A2",
    "DNAJC6", "SYNJ1", "FBXO7", "PLA2G6", "ATP1A3", "MAPT", "GCH1",
    "VPS13C", "CHCHD2", "DJ1",
}
PD_DECOY = {
    "DRD2", "DRD1", "DRD3", "SLC6A3", "COMT", "MAOB", "MAOA", "DDC",
    "HTR2A", "ADORA2A", "GRIN1", "GRIN2B",
}


def tract_score(flags: set[str]) -> float:
    s = 0.0
    if any(k.endswith("Approved Drug") for k in flags):
        s += 0.25
    if any("Advanced Clinical" in k or "Phase 1 Clinical" in k for k in flags):
        s += 0.20
    if any("Structure with Ligand" in k for k in flags):
        s += 0.25
    if any("High-Quality Pocket" in k or "High-Quality Ligand" in k for k in flags):
        s += 0.20
    if any("Druggable Family" in k for k in flags):
        s += 0.10
    if any(k.startswith("AB:") for k in flags):
        s += 0.10
    return min(s, 1.0)


def fetch(efo: str, cache: Path, offline_fallback: Path | None = None) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    q = QUERY_TMPL.format(efo=efo)
    req = urllib.request.Request(
        "https://api.platform.opentargets.org/api/v4/graphql",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        cache.write_text(json.dumps(data))
        return data
    except Exception as e:
        print("fetch failed", efo, e)
        if offline_fallback and offline_fallback.exists():
            return json.loads(offline_fallback.read_text())
        raise


def parse(data: dict, classic: set[str], decoy: set[str]) -> list[dict]:
    parsed = []
    rows = data["data"]["disease"]["associatedTargets"]["rows"]
    for r in rows:
        m = {x["id"]: x["score"] for x in r["datatypeScores"]}
        flags = {
            f"{t.get('modality')}:{t.get('label')}"
            for t in (r["target"].get("tractability") or [])
            if t.get("value")
        }
        g = r["target"]["approvedSymbol"]
        rec = {
            "gene": g,
            "ot": r["score"],
            "ga": m.get("genetic_association", 0.0),
            "gl": m.get("genetic_literature", 0.0),
            "dna": max(m.get("genetic_association", 0.0), m.get("genetic_literature", 0.0)),
            "clin": m.get("clinical", 0.0),
            "lit": m.get("literature", 0.0),
            "rna": m.get("rna_expression", 0.0),
            "animal": m.get("animal_model", 0.0),
            "somatic": m.get("somatic_mutation", 0.0),
            "pathway": m.get("affected_pathway", 0.0),
        }
        rec["prot"] = (
            0.35 * tract_score(flags)
            + 0.25 * rec["rna"]
            + 0.20 * rec["animal"]
            + 0.20 * min(rec["lit"], 0.9)
        )
        rec["both"] = 0.55 * rec["dna"] + 0.45 * rec["prot"]
        rec["ga_only"] = rec["ga"]
        rec["equal"] = 0.5 * rec["dna"] + 0.5 * rec["prot"]
        rec["dna_heavy"] = 0.70 * rec["dna"] + 0.30 * rec["prot"]
        rec["prot_heavy"] = 0.30 * rec["dna"] + 0.70 * rec["prot"]
        rec["critic"] = rec["both"]
        rec["attack"] = ""
        if rec["clin"] >= 0.8 and rec["dna"] < 0.25:
            rec["critic"] *= 0.35
            rec["attack"] = "symptomatic"
        elif rec["dna"] >= 0.45 and rec["prot"] < 0.12:
            rec["critic"] *= 0.85
            rec["attack"] = "genetics-only"
        elif rec["prot"] >= 0.45 and rec["dna"] < 0.15 and rec["clin"] < 0.2:
            rec["critic"] *= 0.7
            rec["attack"] = "protein-only"
        rec["classic"] = int(g in classic)
        rec["decoy"] = int(g in decoy)
        parsed.append(rec)
    return parsed


def ranks(parsed, key):
    order = sorted(parsed, key=lambda x: (-x[key], x["gene"]))
    return {e["gene"]: i + 1 for i, e in enumerate(order)}


def topk(parsed, key, k=20):
    return sorted(parsed, key=lambda x: (-x[key], x["gene"]))[:k]


def stats_at_k(parsed, key, k=20):
    lst = topk(parsed, key, k)
    return {
        "curated": sum(x["classic"] for x in lst),
        "decoy": sum(x["decoy"] for x in lst),
        "genetic": sum(x["dna"] >= 0.3 for x in lst),
        "genes": [x["gene"] for x in lst],
    }


def enrichment(parsed, key, label_field="classic"):
    """Mann-Whitney-like rank-sum via simple AUROC on ranks."""
    pos = [x[key] for x in parsed if x[label_field]]
    neg = [x[key] for x in parsed if not x[label_field]]
    if not pos or not neg:
        return None
    # AUROC = P(pos > neg)
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def mean_rank(parsed, key, mask_field):
    r = ranks(parsed, key)
    vals = [r[x["gene"]] for x in parsed if x[mask_field]]
    return float(np.mean(vals)) if vals else None


def write_csv(parsed, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(parsed[0].keys()))
        w.writewriterow = w.writeheader
        w.writeheader()
        w.writerows(parsed)


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_metrics(parsed, path):
    methods = [
        ("Litera-\nture", "lit"),
        ("OT\noverall", "ot"),
        ("Genetics\nchannel", "dna"),
        ("Protein\nchannel", "prot"),
        ("Fusion", "both"),
        ("Fusion\n+ rule", "critic"),
    ]
    curated, decoy, gen = [], [], []
    for _, key in methods:
        s = stats_at_k(parsed, key)
        curated.append(s["curated"])
        decoy.append(s["decoy"])
        gen.append(s["genetic"])
    labels = [m[0] for m in methods]
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.bar(x - w, curated, w, label="Curated AD genes", color="#1f4e79")
    ax.bar(x, decoy, w, label="Symptomatic controls", color="#c44e52")
    ax.bar(x + w, gen, w, label="Genetic support ≥0.3", color="#6b9ac4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("Count in top 20", fontsize=7)
    ax.set_ylim(0, 21)
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3, handlelength=1.2, columnspacing=0.8)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close()


def fig_scatter(parsed, path):
    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    ax.scatter(
        [e["dna"] for e in parsed],
        [e["prot"] for e in parsed],
        s=11,
        c="#b7c0c8",
        alpha=0.65,
        linewidths=0,
    )
    for e in parsed:
        if e["classic"] or e["decoy"]:
            c = "#1f4e79" if e["classic"] else "#c44e52"
            ax.scatter([e["dna"]], [e["prot"]], s=26, c=c, zorder=3, linewidths=0)
    for e in parsed:
        if e["gene"] in {"APP", "PSEN1", "APOE", "TREM2", "ACHE", "GRIN1", "GRIN2B", "CR1", "BIN1"}:
            off = {"GRIN2B": (4, 4), "GRIN1": (4, -1), "ACHE": (4, -6),
                   "APP": (-14, 2), "PSEN1": (3, 2), "TREM2": (3, -3),
                   "APOE": (3, 2), "BIN1": (3, -5), "CR1": (3, -3)}.get(e["gene"], (2.5, 2.5))
            ax.annotate(
                e["gene"],
                (e["dna"], e["prot"]),
                fontsize=5.6,
                xytext=off,
                textcoords="offset points",
            )
    ax.set_xlabel("Genetic-channel score")
    ax.set_ylabel("Protein-channel score")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close()


def fig_rank_shift(parsed, path):
    r_ot = ranks(parsed, "ot")
    r_c = ranks(parsed, "critic")
    genes_up = ["TREM2", "CR1", "PLCG2", "BIN1", "INPP5D", "MS4A6A"]
    genes_dn = ["ACHE", "GRIN1", "GRIN2B", "BCHE", "SLC6A3", "GRIN2A"]
    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    y = 0
    labels = []
    for g, color, marker in (
        *[(g, "#1f4e79", "o") for g in genes_up],
        *[(g, "#c44e52", "s") for g in genes_dn],
    ):
        if g not in r_ot:
            continue
        ax.plot([r_ot[g], r_c[g]], [y, y], color=color, lw=1.4, alpha=0.85)
        ax.scatter([r_ot[g]], [y], c="#888", s=18, zorder=3)
        ax.scatter([r_c[g]], [y], c=color, s=22, zorder=4, marker=marker)
        labels.append(g)
        y += 1
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Rank in top-150 list (lower is better)")
    ax.invert_yaxis()
    style_ax(ax)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close()


def print_report(name, parsed):
    methods = [
        ("Literature only", "lit"),
        ("Open Targets overall", "ot"),
        ("Genetic association only", "ga_only"),
        ("Genetic channel", "dna"),
        ("Protein channel", "prot"),
        ("Equal fusion", "equal"),
        ("DNA-heavy fusion", "dna_heavy"),
        ("Protein-heavy fusion", "prot_heavy"),
        ("Two-channel fusion", "both"),
        ("Fusion + rule", "critic"),
    ]
    print(f"\n==== {name} n={len(parsed)} classic={sum(x['classic'] for x in parsed)} decoy={sum(x['decoy'] for x in parsed)}")
    for label, key in methods:
        s = stats_at_k(parsed, key)
        auroc_c = enrichment(parsed, key, "classic")
        auroc_d = enrichment(parsed, key, "decoy")
        mr_c = mean_rank(parsed, key, "classic")
        mr_d = mean_rank(parsed, key, "decoy")
        print(
            f"{label:28} C@20={s['curated']:2d} D@20={s['decoy']:2d} G@20={s['genetic']:2d} "
            f"AUROC_c={auroc_c:.3f} AUROC_d={auroc_d:.3f} meanR_c={mr_c:.1f} meanR_d={mr_d:.1f}"
        )
        print("   ", ", ".join(s["genes"][:12]))
    r_ot = ranks(parsed, "ot")
    r_both = ranks(parsed, "both")
    r_c = ranks(parsed, "critic")
    print("rank shifts OT -> fusion -> rule")
    for g in ["TREM2", "CR1", "PLCG2", "BIN1", "INPP5D", "ACHE", "GRIN1", "GRIN2B", "BCHE", "SLC6A3", "APP", "PSEN1"]:
        if g in r_ot:
            print(f"  {g:8} OT={r_ot[g]:3d} fus={r_both[g]:3d} filt={r_c[g]:3d}")
    # overlap
    a = set(stats_at_k(parsed, "dna")["genes"])
    b = set(stats_at_k(parsed, "prot")["genes"])
    print("top20 overlap dna∩prot", len(a & b), "only dna", sorted(a - b), "only prot", sorted(b - a))


def main():
    ad = parse(
        fetch("MONDO_0004975", DATA / "ot_ad.json"),
        AD_CLASSIC,
        AD_DECOY,
    )
    write_csv(ad, RES / "scores_ad.csv")
    fig_metrics(ad, OUT / "fig_metrics.pdf")
    fig_scatter(ad, OUT / "fig_scatter.pdf")
    fig_rank_shift(ad, OUT / "fig_ranks.pdf")
    print_report("Alzheimer", ad)

    try:
        pd = parse(
            fetch("MONDO_0005180", DATA / "ot_pd.json"),
            PD_CLASSIC,
            PD_DECOY,
        )
        write_csv(pd, RES / "scores_pd.csv")
        print_report("Parkinson", pd)
    except Exception as e:
        print("PD skipped", e)


if __name__ == "__main__":
    main()
