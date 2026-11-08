#!/usr/bin/env python3
"""Genetics-only baseline, protein-channel ablation, and weight/threshold sensitivity (camera-ready additions)."""
import json, itertools, sys
import numpy as np
from pathlib import Path
D=str(Path(__file__).resolve().parent)+"/"
import importlib.util
spec=importlib.util.spec_from_file_location("ra",D+"run_analysis.py"); ra=importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)

def base(name):
    if name=="AD": return ra.parse(json.load(open(D+"data/ot_ad.json")),ra.AD_CLASSIC,ra.AD_DECOY), json.load(open(D+"data/ot_ad.json"))
    return ra.parse(json.load(open(D+"data/ot_pd.json")),ra.PD_CLASSIC,ra.PD_DECOY), json.load(open(D+"data/ot_pd.json"))

def add_tract(parsed, raw):
    rows=raw["data"]["disease"]["associatedTargets"]["rows"]
    tmap={}
    for r in rows:
        flags={f"{t.get('modality')}:{t.get('label')}" for t in (r["target"].get("tractability") or []) if t.get("value")}
        tmap[r["target"]["approvedSymbol"]]=ra.tract_score(flags)
    for e in parsed: e["t"]=tmap[e["gene"]]
    return parsed

def prot(e,w=(0.35,0.25,0.20,0.20),cap=0.9):
    return w[0]*e["t"]+w[1]*e["rna"]+w[2]*e["animal"]+w[3]*min(e["lit"],cap)

def evaluate(parsed,key):
    s=ra.stats_at_k(parsed,key)
    top=ra.topk(parsed,key)
    return dict(C=s["curated"],D=s["decoy"],G=s["genetic"],
        auc=ra.enrichment(parsed,key,"classic"),mr=ra.mean_rank(parsed,key,"classic"),
        mrd=ra.mean_rank(parsed,key,"decoy"),
        tract=np.mean([x["t"] for x in top]), ntr=sum(x["t"]>=0.5 for x in top),
        P=np.mean([x["prot"] for x in top]))

def rule(e,f,clin=0.8,g=0.25,m=0.35):
    v=f
    if e["clin"]>=clin and e["dna"]<g: v*=m
    elif e["dna"]>=0.45 and e["prot"]<0.12: v*=0.85
    elif e["prot"]>=0.45 and e["dna"]<0.15 and e["clin"]<0.2: v*=0.7
    return v

for dz in ["AD","PD"]:
    P,raw=base(dz); add_tract(P,raw)
    print(f"\n######## {dz}")
    # genetics vs fusion with protein-information metrics
    print("== genetics-only vs fusion (top20 mean t, #t>=0.5, mean sP)")
    for k in ["ga_only","dna","both","equal","dna_heavy","critic"]:
        r=evaluate(P,k); print(f"{k:10} C={r['C']} D={r['D']} AUC={r['auc']:.3f} mr={r['mr']:.1f} mrd={r['mrd']:.1f} tract={r['tract']:.3f} ntr={r['ntr']} P={r['P']:.3f}")
    # ablation of protein channel components
    print("== ablation (drop one component from protein channel, then 0.55/0.45 fusion)")
    names=["tract","rna","animal","lit"]
    for drop in [None,0,1,2,3]:
        w=[0.35,0.25,0.20,0.20]
        if drop is not None: w[drop]=0.0
        for e in P:
            e["_p"]=prot(e,w); e["_f"]=0.55*e["dna"]+0.45*e["_p"]
        r=evaluate(P,"_f")
        print(f"drop={names[drop] if drop is not None else 'none':7} C={r['C']} D={r['D']} AUC={r['auc']:.3f} mr={r['mr']:.1f} mrd={r['mrd']:.1f} ntr={r['ntr']}")
    # fusion weight sweep
    print("== fusion weight sweep wG")
    for wg in np.round(np.arange(0,1.0001,0.05),2):
        for e in P: e["_f"]=wg*e["dna"]+(1-wg)*e["prot"]
        r=evaluate(P,"_f")
        print(f"wG={wg:.2f} C={r['C']} D={r['D']} AUC={r['auc']:.3f} mr={r['mr']:.1f} ntr={r['ntr']}")
    # protein weight perturbation
    rng=np.random.default_rng(0)
    res=[]
    for i in range(1000):
        w=rng.dirichlet(np.array([0.35,0.25,0.20,0.20])*40)
        for e in P:
            pp=prot(e,w); e["_f"]=0.55*e["dna"]+0.45*pp
        r=evaluate(P,"_f"); res.append((r["C"],r["D"],r["auc"]))
    res=np.array(res)
    print("== protein-weight perturbation Dirichlet(conc=40) 1000 draws: C min/med/max",res[:,0].min(),np.median(res[:,0]),res[:,0].max(),
          " D min/max",res[:,1].min(),res[:,1].max(), " AUC min/max %.3f %.3f"%(res[:,2].min(),res[:,2].max()),
          " frac D==0 %.3f"%(res[:,1]==0).mean())
    res=[]
    for w in itertools.product([0.5,1.0,1.5],repeat=4):
        ww=np.array([0.35,0.25,0.20,0.20])*np.array(w); ww=ww/ww.sum()
        for e in P:
            pp=prot(e,ww); e["_f"]=0.55*e["dna"]+0.45*pp
        r=evaluate(P,"_f"); res.append((r["C"],r["D"],r["auc"]))
    res=np.array(res)
    print("== protein weights x{0.5,1,1.5} grid (81, renorm): C",res[:,0].min(),res[:,0].max()," D",res[:,1].min(),res[:,1].max()," AUC %.3f-%.3f"%(res[:,2].min(),res[:,2].max()))
    for cap in [0.8,0.9,1.0]:
        for e in P: e["_f"]=0.55*e["dna"]+0.45*prot(e,cap=cap)
        r=evaluate(P,"_f"); print(f"lit cap {cap}: C={r['C']} D={r['D']} AUC={r['auc']:.3f}")
    # rule sensitivity
    print("== rule sensitivity (clin thr, G thr, mult)")
    out=[]
    for clin in [0.6,0.7,0.8,0.9]:
        for g in [0.15,0.25,0.35]:
            for m in [0.2,0.35,0.5]:
                for e in P: e["_r"]=rule(e,e["both"],clin,g,m)
                r=evaluate(P,"_r"); out.append((clin,g,m,r["C"],r["D"],r["auc"],r["mrd"]))
    out=np.array(out)
    print("C range",out[:,3].min(),out[:,3].max(),"D range",out[:,4].min(),out[:,4].max(),"AUC %.3f-%.3f"%(out[:,5].min(),out[:,5].max()),"mean rank ctrl %.1f-%.1f"%(out[:,6].min(),out[:,6].max()))
    for row in out:
        if row[0] in (0.6,0.9) and row[2]==0.35: print("  ",row)
    # how many genes hit each rule
    for e in P: pass
    from collections import Counter
    print("rule hits", Counter(e["attack"] for e in P))
    print("rule1 hits:", sorted(e["gene"] for e in P if e["attack"]=="symptomatic"))
