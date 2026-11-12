#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMNET REVISION MASTER RUNNER
Manuscript: COMNET-D-26-06148
Purpose: one-pass/resumable execution of the new reviewer-driven experiments.

This runner does NOT overwrite the original Zenodo artifact.
It writes everything to MyDrive/COMNET_REVISION_RUN.

Key design changes vs initial submission:
- separates model_seed / pool_seed / canary_seed
- saves per-flow predictions for new runs
- implements explicit three-way abstention
- caches/checkpoints at phase level
- records exact environment
"""

from __future__ import annotations
import os, sys, glob, gc, json, time, math, random, shutil, zipfile, hashlib, platform, subprocess, traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    import lightgbm as lgb
except Exception:
    lgb=None

# --------------------------- CLI ---------------------------
import argparse
P=argparse.ArgumentParser()
P.add_argument("--profile", choices=["smoke","revision","full"], default="revision")
P.add_argument("--phases", default="all",
               help="comma-separated: audit,core,r9,pool,dependence,summary or all")
P.add_argument("--base", default="/content/drive/MyDrive")
P.add_argument("--out", default=None)
P.add_argument("--force", action="store_true")
ARGS=P.parse_args()

BASE=Path(ARGS.base)
OUT=Path(ARGS.out) if ARGS.out else BASE/"COMNET_REVISION_RUN"
OUT.mkdir(parents=True,exist_ok=True)
for sub in ["environment","logs","predictions","core","R4_score_degradation","R6_runtime",
            "R9_prevalence_budget","E1_detector_family","E2_metrics","E3_pool_size",
            "E4_strat_fp2_ratio","E5_dependence","tables","figures","checkpoints"]:
    (OUT/sub).mkdir(exist_ok=True)

PHASES=set(["audit","core","r9","pool","dependence","summary"] if ARGS.phases=="all"
           else [x.strip() for x in ARGS.phases.split(",") if x.strip()])

# --------------------------- CONFIG ---------------------------
NF_PATHS={
    "CSE": BASE/"APT_Data/NF-CSE-CIC-IDS2018-v2.zip",
    "UNSW": BASE/"APT_Data/NF-UNSW-NB15-v2",
    "BoT": BASE/"APT_Data/NF-BoT-IoT-v2.zip",
    "ToN": BASE/"APT_Data/NF-ToN-IoT-v2.rar",
}
UNR_DIR=BASE/"Unraveled/network-flows"

PAIRS_NF=[("CSE","UNSW"),("UNSW","CSE"),("ToN","BoT"),("CSE","ToN"),("ToN","UNSW"),("BoT","CSE")]
SRC_UNR=["CSE","ToN","UNSW","BoT"]

TAU=.75
DELTAS=[.02,.05,.10]
ALPHAS=[0.,.25,.5,.75,1.]
BUDGETS=[10,20,50,100]
CORE_BUDGET=50
KEY_STRATS=["random","coreset","strat_score_W","strat_fp2"]
CORE_STRATS=["random","uncertainty","coreset","strat_score_W","strat_fp2"]
MODEL_FAMILIES=["lightgbm","logreg","mlp"]
PREV_RATES=[.005,.01,.0144,.02,.05,.10]
FP2_RATIOS=[.25,.50,.75]
POOL_SIZES=[40_000,80_000,160_000]
EVAL_N=20_000

if ARGS.profile=="smoke":
    MODEL_SEEDS=[0]; POOL_SEEDS=[0]; CANARY_SEEDS=[0,1]
    MAX_ROWS=6_000; CORE_MODELS=["lightgbm","logreg"]; R9_CANARY_SEEDS=[0,1]
    DEP_REPS=250
elif ARGS.profile=="full":
    MODEL_SEEDS=list(range(5)); POOL_SEEDS=list(range(4)); CANARY_SEEDS=list(range(10))
    MAX_ROWS=80_000; CORE_MODELS=MODEL_FAMILIES; R9_CANARY_SEEDS=list(range(10))
    DEP_REPS=5000
else:  # recommended reviewer revision profile
    MODEL_SEEDS=list(range(3)); POOL_SEEDS=list(range(3)); CANARY_SEEDS=list(range(5))
    MAX_ROWS=80_000; CORE_MODELS=MODEL_FAMILIES; R9_CANARY_SEEDS=list(range(5))
    DEP_REPS=2000

DROP_NF=['IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT','Label','Attack','Dataset']
NF2COLS={'dur':'FLOW_DURATION_MILLISECONDS','inb':'IN_BYTES','outb':'OUT_BYTES',
         'inp':'IN_PKTS','outp':'OUT_PKTS','durin':'DURATION_IN','durout':'DURATION_OUT',
         'maxps':'LONGEST_FLOW_PKT','minps':'SHORTEST_FLOW_PKT'}
UNRCOLS={'dur':'bidirectional_duration_ms','inb':'src2dst_bytes','outb':'dst2src_bytes',
         'inp':'src2dst_packets','outp':'dst2src_packets','durin':'src2dst_duration_ms',
         'durout':'dst2src_duration_ms','maxps':'bidirectional_max_ps','minps':'bidirectional_min_ps'}
ATTACK_STAGES={'reconnaissance','lateral movement','establish foothold','data exfiltration'}
UNR_JUNK={'6','26','33'}

RUN_CFG=dict(profile=ARGS.profile, model_seeds=MODEL_SEEDS, pool_seeds=POOL_SEEDS,
             canary_seeds=CANARY_SEEDS, max_rows=MAX_ROWS, eval_n=EVAL_N,
             alphas=ALPHAS,budgets=BUDGETS,prevalence_rates=PREV_RATES,
             models=CORE_MODELS, tau=TAU,deltas=DELTAS,fp2_ratios=FP2_RATIOS)
(OUT/"run_config.json").write_text(json.dumps(RUN_CFG,indent=2),encoding="utf-8")

# --------------------------- UTILS ---------------------------
def log(msg):
    s=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(s,flush=True)
    with open(OUT/"logs/master.log","a",encoding="utf-8") as f: f.write(s+"\n")

def atomic_csv_append(path:Path, rows):
    if not rows: return
    df=pd.DataFrame(rows)
    header=not path.exists()
    df.to_csv(path,mode="a",header=header,index=False)
    rows.clear()

def done_keys(path:Path, cols):
    if not path.exists(): return set()
    try:
        d=pd.read_csv(path,usecols=cols)
        return set(map(tuple,d.drop_duplicates().itertuples(index=False,name=None)))
    except Exception:
        return set()

def safe_metric(fn,*args,**kwargs):
    try: return float(fn(*args,**kwargs))
    except Exception: return np.nan

def tpr_at_fpr(y,p,target=.01):
    if len(np.unique(y))<2: return np.nan
    fpr,tpr,_=roc_curve(y,p)
    idx=np.where(fpr<=target)[0]
    return float(tpr[idx].max()) if len(idx) else 0.0

def metric_bundle(y,p):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,int)
    return dict(
        auroc=safe_metric(roc_auc_score,y,p),
        auprc=safe_metric(average_precision_score,y,p),
        tpr_fpr_001=tpr_at_fpr(y,p,.01),
        brier=safe_metric(brier_score_loss,y,p),
        logloss=safe_metric(log_loss,y,p,labels=[0,1]),
    )

def three_way(est,tau=TAU,delta=.05):
    if est is None or not np.isfinite(est): return "insufficient"
    if abs(est-tau)<delta: return "insufficient"
    return "failed" if est<tau-delta else "operational"

def clean(X):
    if isinstance(X,np.ndarray):
        X=pd.DataFrame(X)
    return (X.astype(np.float64).replace([np.inf,-np.inf],np.nan)
              .clip(-1e30,1e30).fillna(0.0).astype(np.float32))

def mapped_space(df,cmap):
    Z=pd.DataFrame(index=df.index)
    for k,c in cmap.items():
        Z[k]=pd.to_numeric(df[c],errors="coerce") if c in df.columns else np.nan
    Z["bytes_ratio"]=Z["inb"]/(Z["outb"]+1.0)
    Z["pkts_ratio"]=Z["inp"]/(Z["outp"]+1.0)
    Z["mps_in"]=Z["inb"]/(Z["inp"]+1.0)
    Z["mps_out"]=Z["outb"]/(Z["outp"]+1.0)
    return clean(Z)

def extract_archive(src:Path,dst:Path):
    dst.mkdir(parents=True,exist_ok=True)
    if src.is_dir():
        for f in src.rglob("*.csv"):
            target=dst/f.name
            if not target.exists(): shutil.copy2(f,target)
    elif str(src).lower().endswith(".zip"):
        with zipfile.ZipFile(src) as z: z.extractall(dst)
    elif str(src).lower().endswith(".rar"):
        cmd=["7z","x","-y",str(src),f"-o{dst}"]
        subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    else:
        raise ValueError(f"Unsupported dataset source: {src}")

# --------------------------- ENV / AUDIT ---------------------------
def capture_environment():
    envdir=OUT/"environment"
    lines=[
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"processor={platform.processor()}",
    ]
    try:
        import sklearn, scipy, matplotlib
        lines += [f"numpy={np.__version__}",f"pandas={pd.__version__}",
                  f"scipy={scipy.__version__}",f"sklearn={sklearn.__version__}",
                  f"matplotlib={matplotlib.__version__}"]
    except Exception: pass
    if lgb is not None: lines.append(f"lightgbm={lgb.__version__}")
    try:
        lines.append(subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                                             text=True).strip())
    except Exception: lines.append("gpu=not_detected_or_nvidia_smi_unavailable")
    (envdir/"environment_revision.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    try:
        freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
        (envdir/"pip_freeze_revision.txt").write_text(freeze,encoding="utf-8")
    except Exception: pass

def audit_paths():
    rows=[]
    for name,p in {**NF_PATHS,"UNR":UNR_DIR}.items():
        rows.append(dict(name=name,path=str(p),exists=p.exists(),is_dir=p.is_dir()))
    pd.DataFrame(rows).to_csv(OUT/"environment/data_path_audit.csv",index=False)
    missing=[r for r in rows if not r["exists"]]
    if missing: raise FileNotFoundError("Missing dataset paths: "+str(missing))
    log("Data path audit PASS.")

# --------------------------- DATA LOADERS ---------------------------
def _nf_extract_dir(name):
    return Path("/content/_comnet_nf")/name

def load_nf(name,pool_seed,max_rows=MAX_ROWS):
    src=NF_PATHS[name]
    edir=_nf_extract_dir(name)
    if not list(edir.rglob("*.csv")):
        log(f"Extracting {name} -> {edir}")
        extract_archive(src,edir)
    csvs=[p for p in edir.rglob("*.csv") if p.stat().st_size>5000]
    if not csvs: raise RuntimeError(f"No CSV extracted for {name}")
    acc=[]; got=0
    rng=np.random.RandomState(pool_seed)
    for f in csvs:
        try:
            chunks=pd.read_csv(f,chunksize=400_000,low_memory=False)
            for chunk in chunks:
                need=max_rows-got
                if need<=0: break
                target=max(need//4,1)
                if len(chunk)>target:
                    chunk=chunk.sample(min(len(chunk),target),random_state=pool_seed+got)
                acc.append(chunk); got+=len(chunk)
            if got>=max_rows: break
        except Exception as e:
            log(f"WARN reading {f}: {e}")
    if not acc: raise RuntimeError(f"Could not load {name}")
    df=pd.concat(acc,ignore_index=True)
    if len(df)>max_rows:
        df=df.sample(max_rows,random_state=pool_seed).reset_index(drop=True)
    label_candidates=[c for c in df.columns if c.lower()=="label"]
    if not label_candidates: raise RuntimeError(f"No Label column in {name}")
    lc=label_candidates[0]
    y=(pd.to_numeric(df[lc],errors="coerce").fillna(0).values!=0).astype(int)
    Xfull=clean(df.drop(columns=[c for c in DROP_NF if c in df.columns],errors="ignore")
                 .select_dtypes(include=[np.number]))
    Xmap=mapped_space(df,NF2COLS)
    return Xfull.reset_index(drop=True),Xmap.reset_index(drop=True),y

def scan_unraveled(include_meta=False, benign_per_file=1500, seed=0):
    files=sorted(UNR_DIR.rglob("*.csv"))
    if not files: raise RuntimeError(f"No Unraveled CSV in {UNR_DIR}")
    need=list(UNRCOLS.values())
    meta_cols=["src_ip","dst_ip","bidirectional_first_seen_ms"]
    atk=[]; ben=[]; meta_atk=[]; meta_ben=[]; label_col=None
    for fi,f in enumerate(files):
        try: head=pd.read_csv(f,nrows=5,engine="python",on_bad_lines="skip")
        except Exception: continue
        if label_col is None or label_col not in head.columns:
            cands=[c for c in head.columns if c.lower() in ("label","labels","stage","attack_stage","class")]
            if not cands: cands=[c for c in head.columns if "label" in c.lower()]
            if cands: label_col=cands[0]
        if label_col is None or label_col not in head.columns: continue
        use=[c for c in need+[label_col]+(meta_cols if include_meta else []) if c in head.columns]
        if label_col not in use or len([c for c in need if c in use])<5: continue
        try: df=pd.read_csv(f,usecols=use,engine="c",on_bad_lines="skip")
        except Exception:
            try: df=pd.read_csv(f,usecols=use,engine="python",on_bad_lines="skip")
            except Exception: continue
        lab=df[label_col].astype(str).str.strip().str.lower()
        junk=lab.isin(UNR_JUNK)|lab.str.contains("/",regex=False)
        df=df.loc[~junk].copy(); lab=lab.loc[~junk]
        is_atk=lab.isin(ATTACK_STAGES).values
        if np.any(is_atk):
            atk.append(mapped_space(df.loc[is_atk],UNRCOLS))
            if include_meta:
                m=df.loc[is_atk,[c for c in meta_cols if c in df.columns]].copy()
                m["y"]=1; meta_atk.append(m)
        b=df.loc[~is_atk]
        if len(b):
            take=b.sample(min(benign_per_file,len(b)),random_state=seed+fi)
            ben.append(mapped_space(take,UNRCOLS))
            if include_meta:
                m=take[[c for c in meta_cols if c in take.columns]].copy()
                m["y"]=0; meta_ben.append(m)
        if fi%50==0: log(f"Unraveled scan {fi+1}/{len(files)}")
    Xa=pd.concat(atk,ignore_index=True); Xb=pd.concat(ben,ignore_index=True)
    if include_meta:
        ma=pd.concat(meta_atk,ignore_index=True); mb=pd.concat(meta_ben,ignore_index=True)
        return Xa,Xb,ma,mb
    return Xa,Xb

_UNR_CACHE={}
def get_unr_pools():
    if "basic" not in _UNR_CACHE:
        _UNR_CACHE["basic"]=scan_unraveled(False,1500,0)
    return _UNR_CACHE["basic"]

def make_prevalence_pool(Xa,Xb,rate,pool_seed,n=MAX_ROWS):
    rng=np.random.RandomState(pool_seed)
    na=int(round(n*rate)); nb=n-na
    ia=rng.choice(len(Xa),size=min(na,len(Xa)),replace=False)
    ib=rng.choice(len(Xb),size=min(nb,len(Xb)),replace=False)
    X=np.vstack([Xa.iloc[ia].values,Xb.iloc[ib].values]).astype(np.float32)
    y=np.r_[np.ones(len(ia),int),np.zeros(len(ib),int)]
    perm=rng.permutation(len(y))
    return X[perm],y[perm]

# --------------------------- MODELS ---------------------------
def model_factory(family,seed):
    if family=="lightgbm":
        if lgb is None: raise RuntimeError("lightgbm unavailable")
        return lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=64,
                                  subsample=.8,colsample_bytree=.8,random_state=seed,
                                  n_jobs=-1,verbose=-1)
    if family=="logreg":
        return LogisticRegression(max_iter=600,solver="lbfgs",random_state=seed,n_jobs=None)
    if family=="mlp":
        return MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",
                             max_iter=60,early_stopping=True,validation_fraction=.1,
                             random_state=seed,batch_size=512,learning_rate_init=.001)
    raise ValueError(family)

def fit_realization(Xs,ys,Xt,model_family,model_seed):
    idx=np.random.RandomState(model_seed).permutation(len(Xs)); h=int(.75*len(idx))
    tr,held=idx[:h],idx[h:]
    qt=QuantileTransformer(output_distribution="normal",n_quantiles=min(1000,h),
                           random_state=model_seed)
    Xtr=qt.fit_transform(Xs[tr]); Xhd=qt.transform(Xs[held]); Xt2=qt.transform(Xt)
    model=model_factory(model_family,model_seed)
    t0=time.perf_counter(); model.fit(Xtr,ys[tr]); fit_sec=time.perf_counter()-t0
    return model,qt,Xtr,ys[tr],Xhd,ys[held],Xt2,fit_sec

# --------------------------- ACQUISITION ---------------------------
def bins_of(p,e): return np.clip(np.digitize(p,e[1:-1]),0,len(e)-2)

def sel_strat(X,p,B,seed,weighted=False):
    rng=np.random.RandomState(seed); n=len(X)
    e=np.quantile(p,np.linspace(0,1,11)); bins=bins_of(p,e)
    per=max(1,B//10); idx=[]; w=[]
    for b in range(10):
        m=np.where(bins==b)[0]
        if not len(m): continue
        take=rng.choice(m,size=min(per,len(m)),replace=False)
        idx.extend(take.tolist())
        # same weighting convention as submitted code
        w.extend([(len(m)/n)/(len(take)/B)]*len(take))
    idx=np.asarray(idx,int); w=np.asarray(w,float)
    if len(idx)<B:
        rest=np.setdiff1d(np.arange(n),idx)
        extra=rng.choice(rest,size=min(B-len(idx),len(rest)),replace=False)
        idx=np.r_[idx,extra]; w=np.r_[w,np.ones(len(extra))]
    return idx[:B],(w[:B] if weighted else None)

def sel_coreset(X,B,seed):
    rng=np.random.RandomState(seed); n=len(X)
    km=MiniBatchKMeans(n_clusters=min(B,n),random_state=seed,n_init=3,batch_size=2048).fit(X)
    idx=[]
    for c in range(min(B,n)):
        m=np.where(km.labels_==c)[0]
        if not len(m): continue
        dist=np.linalg.norm(X[m]-km.cluster_centers_[c],axis=1)
        idx.append(m[np.argmin(dist)])
    idx=np.asarray(idx,int)
    if len(idx)<B:
        rest=np.setdiff1d(np.arange(n),idx)
        idx=np.r_[idx,rng.choice(rest,size=B-len(idx),replace=False)]
    return idx[:B],None

def sel_fp2(X,p,B,seed,rho=.5):
    n_strat=max(1,min(B-1,int(round(B*(1-rho)))))
    i1,_=sel_strat(X,p,n_strat,seed,False)
    chosen=list(i1)
    dmin=np.full(len(X),np.inf)
    for i in chosen: dmin=np.minimum(dmin,np.linalg.norm(X-X[i],axis=1))
    for _ in range(B-len(chosen)):
        dmin[np.asarray(chosen,int)]=-np.inf
        j=int(np.argmax(dmin)); chosen.append(j)
        dmin=np.minimum(dmin,np.linalg.norm(X-X[j],axis=1))
    return np.asarray(chosen[:B]),None

def select_rule(strategy,X,p,B,seed,fp2_rho=.5):
    rng=np.random.RandomState(seed)
    if strategy=="random": return rng.choice(len(X),size=min(B,len(X)),replace=False),None
    if strategy=="uncertainty": return np.argsort(np.abs(p-.5))[:B],None
    if strategy=="coreset": return sel_coreset(X,B,seed)
    if strategy=="strat_score_W": return sel_strat(X,p,B,seed,True)
    if strategy=="strat_score": return sel_strat(X,p,B,seed,False)
    if strategy=="strat_fp2": return sel_fp2(X,p,B,seed,fp2_rho)
    raise ValueError(strategy)

def canary_metrics(y,p,idx,w=None):
    yy=y[idx]; pp=p[idx]
    out=dict(n_attack=int(yy.sum()),n_benign=int(len(yy)-yy.sum()),computable=0,
             est_auroc=np.nan,est_auprc=np.nan,est_tpr_fpr_001=np.nan)
    if 0<yy.sum()<len(yy):
        out["computable"]=1
        out["est_auroc"]=safe_metric(roc_auc_score,yy,pp,sample_weight=w)
        # weighted AP isn't supported consistently across all versions; use when valid
        out["est_auprc"]=safe_metric(average_precision_score,yy,pp,sample_weight=w)
        out["est_tpr_fpr_001"]=tpr_at_fpr(yy,pp,.01)
    return out

# --------------------------- LABEL-FREE SIGNALS ---------------------------
def mahal_dist(X,mu,prec):
    D=X-mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,prec,D),0))

def label_free_stats(Xtr,Xeval,p,seed):
    sub=np.random.RandomState(seed).choice(len(Xtr),min(8000,len(Xtr)),replace=False)
    lw=LedoitWolf().fit(Xtr[sub])
    mu=Xtr[sub].mean(0); mean_src=Xtr.mean(0)
    pp=np.clip(p,1e-9,1-1e-9)
    msp=np.maximum(pp,1-pp)
    ent=-(pp*np.log(pp)+(1-pp)*np.log(1-pp))
    md=mahal_dist(Xeval,mu,lw.precision_)
    mm=float(md.mean()); dim=Xeval.shape[1]
    return dict(mean_msp=float(msp.mean()),pred_pos_rate=float((pp>=.5).mean()),
                mean_entropy=float(ent.mean()),mean_mahal=mm,
                mmd_lin=float(np.linalg.norm(Xeval.mean(0)-mean_src)),
                conf_shift=float(msp.mean()*np.exp(-mm/dim)))

# --------------------------- PREP CONFIGS ---------------------------
def prepare_nf_for_pool(pool_seed,max_rows=MAX_ROWS):
    nf={}
    for n in ["CSE","UNSW","BoT","ToN"]:
        nf[n]=load_nf(n,pool_seed,max_rows)
        log(f"Loaded {n} pool_seed={pool_seed}: n={len(nf[n][2])}, attack={nf[n][2].mean():.4f}")
    return nf

def common_full(nf,a,b):
    Xa,_,ya=nf[a]; Xb,_,yb=nf[b]
    cols=[c for c in Xa.columns if c in Xb.columns]
    return Xa[cols].values,ya,Xb[cols].values,yb

def iter_config_data(nf,pool_seed,unr_rate=.0144,max_rows=MAX_ROWS):
    for src,tgt in PAIRS_NF:
        Xs,ys,Xt,yt=common_full(nf,src,tgt)
        yield f"NF:{src}->{tgt}",src,tgt,Xs,ys,Xt,yt
    Xa,Xb=get_unr_pools()
    for src in SRC_UNR:
        Xs=nf[src][1].values; ys=nf[src][2]
        Xt,yt=make_prevalence_pool(Xa,Xb,unr_rate,10000+pool_seed,max_rows)
        yield f"UNR:{src}->UNR@{unr_rate}",src,"UNR",Xs,ys,Xt,yt

# --------------------------- CORE: E1/E2/R4/R6/E4 ---------------------------
def save_prediction_parquet(config,family,ms,ps,alpha,y,p):
    # per-flow predictions only for pure target to keep archive tractable
    if alpha!=0: return
    safe=config.replace(":","_").replace("->","_to_").replace("@","_")
    d=OUT/"predictions"/family/safe
    d.mkdir(parents=True,exist_ok=True)
    path=d/f"model{ms}_pool{ps}.parquet"
    if path.exists() and not ARGS.force: return
    pd.DataFrame(dict(row_id=np.arange(len(y)),y_true=y.astype(np.int8),
                      score=p.astype(np.float32),config=config,model_family=family,
                      model_seed=ms,pool_seed=ps,alpha=alpha,
                      attack_prevalence=float(y.mean()))).to_parquet(path,index=False)

def run_core():
    canary_path=OUT/"core/core_canary_results.csv"
    rq1_path=OUT/"E1_detector_family/label_free_by_family.csv"
    metrics_path=OUT/"E2_metrics/true_metrics_by_family.csv"
    runtime_path=OUT/"R6_runtime/acquisition_runtime.csv"
    ratio_path=OUT/"E4_strat_fp2_ratio/fp2_ratio_results.csv"
    done=done_keys(canary_path,["config","model_family","model_seed","pool_seed","alpha","strategy","budget","canary_seed"])
    done_sig=done_keys(rq1_path,["config","model_family","model_seed","pool_seed","alpha"])
    done_met=done_keys(metrics_path,["config","model_family","model_seed","pool_seed","alpha"])
    done_time=done_keys(runtime_path,["config","model_family","model_seed","pool_seed","alpha","strategy","budget","canary_seed"])
    done_ratio=done_keys(ratio_path,["config","model_seed","pool_seed","alpha","rho_fp","budget","canary_seed"])
    rows=[]; sigrows=[]; metrows=[]; timerows=[]; ratiorows=[]
    for ps in POOL_SEEDS:
        nf=prepare_nf_for_pool(ps,MAX_ROWS)
        for config,src,tgt,Xs,ys,Xt,yt in iter_config_data(nf,ps,.0144,MAX_ROWS):
            for fam in CORE_MODELS:
                for ms in MODEL_SEEDS:
                    basekey=(config,fam,ms,ps)
                    # if all main rows already exist, still metrics/rq1 may be missing;
                    # we keep rerun lightweight via phase-level checkpoints below.
                    log(f"CORE {config} family={fam} model_seed={ms} pool_seed={ps}")
                    try:
                        model,qt,Xtr,ytr,Xhd,yhd,Xt2,fit_sec=fit_realization(Xs,ys,Xt,fam,ms)
                    except Exception as e:
                        log(f"ERROR fit {basekey}: {e}"); traceback.print_exc(); continue
                    rng=np.random.RandomState(100000+ps)
                    tperm=rng.permutation(len(Xt2))[:min(EVAL_N,len(Xt2))]
                    sperm=rng.permutation(len(Xhd))
                    for alpha in ALPHAS:
                        ns=int(EVAL_N*alpha)
                        nt=EVAL_N-ns
                        ti=tperm[:min(nt,len(tperm))]
                        si=sperm[:min(ns,len(sperm))]
                        Xmix=np.vstack([Xt2[ti],Xhd[si]])[:EVAL_N]
                        ymix=np.r_[yt[ti],yhd[si]][:EVAL_N]
                        if len(np.unique(ymix))<2: continue
                        t0=time.perf_counter()
                        p=model.predict_proba(Xmix)[:,1]
                        infer_sec=time.perf_counter()-t0
                        tm=metric_bundle(ymix,p)
                        collapsed=int(tm["auroc"]<TAU)
                        if alpha==0:
                            save_prediction_parquet(config,fam,ms,ps,alpha,ymix,p)
                            mkey=(config,fam,ms,ps,alpha)
                            if mkey not in done_met:
                                metrows.append(dict(config=config,source=src,target=tgt,model_family=fam,
                                                    model_seed=ms,pool_seed=ps,alpha=alpha,
                                                    prevalence=float(ymix.mean()),fit_sec=fit_sec,
                                                    inference_sec=infer_sec,**tm))
                                done_met.add(mkey)
                        skey=(config,fam,ms,ps,alpha)
                        if skey not in done_sig:
                            sig=label_free_stats(Xtr,Xmix,p,ms)
                            sigrows.append(dict(config=config,source=src,target=tgt,model_family=fam,
                                                model_seed=ms,pool_seed=ps,alpha=alpha,
                                                true_auroc=tm["auroc"],collapsed=collapsed,**sig))
                            done_sig.add(skey)
                        # Main B=50 acquisition
                        for strat in CORE_STRATS:
                            for cs in CANARY_SEEDS:
                                key=(config,fam,ms,ps,alpha,strat,CORE_BUDGET,cs)
                                if key in done: continue
                                st=time.perf_counter()
                                idx,w=select_rule(strat,Xmix,p,CORE_BUDGET,cs)
                                sel_sec=time.perf_counter()-st
                                cm=canary_metrics(ymix,p,idx,w)
                                row=dict(config=config,source=src,target=tgt,model_family=fam,
                                         model_seed=ms,pool_seed=ps,canary_seed=cs,alpha=alpha,
                                         prevalence=float(ymix.mean()),strategy=strat,budget=CORE_BUDGET,
                                         true_auroc=tm["auroc"],true_auprc=tm["auprc"],
                                         true_tpr_fpr_001=tm["tpr_fpr_001"],true_brier=tm["brier"],
                                         collapsed_auroc=collapsed,selection_sec=sel_sec,**cm)
                                for delta in DELTAS:
                                    row[f"verdict_d{delta:.2f}"]=three_way(cm["est_auroc"],TAU,delta)
                                rows.append(row)
                                tkey=(config,fam,ms,ps,alpha,strat,CORE_BUDGET,cs)
                                if tkey not in done_time:
                                    timerows.append(dict(config=config,model_family=fam,model_seed=ms,
                                                          pool_seed=ps,alpha=alpha,strategy=strat,
                                                          budget=CORE_BUDGET,canary_seed=cs,
                                                          selection_sec=sel_sec))
                                    done_time.add(tkey)
                        # FP2 split sensitivity (LightGBM only)
                        if fam=="lightgbm":
                            for rho in FP2_RATIOS:
                                for cs in CANARY_SEEDS:
                                    st=time.perf_counter()
                                    idx,w=select_rule("strat_fp2",Xmix,p,CORE_BUDGET,cs,fp2_rho=rho)
                                    sec=time.perf_counter()-st
                                    cm=canary_metrics(ymix,p,idx,w)
                                    rkey=(config,ms,ps,alpha,rho,CORE_BUDGET,cs)
                                    if rkey not in done_ratio:
                                        ratiorows.append(dict(config=config,model_seed=ms,pool_seed=ps,
                                                             canary_seed=cs,alpha=alpha,budget=CORE_BUDGET,
                                                             rho_fp=rho,true_auroc=tm["auroc"],
                                                             collapsed_auroc=collapsed,selection_sec=sec,**cm))
                                        done_ratio.add(rkey)
                        if len(rows)>=500: atomic_csv_append(canary_path,rows)
                        if len(sigrows)>=100: atomic_csv_append(rq1_path,sigrows)
                        if len(metrows)>=100: atomic_csv_append(metrics_path,metrows)
                        if len(timerows)>=500: atomic_csv_append(runtime_path,timerows)
                        if len(ratiorows)>=500: atomic_csv_append(ratio_path,ratiorows)
                    del model,qt,Xtr,Xhd,Xt2
                    gc.collect()
        del nf; gc.collect()
    atomic_csv_append(canary_path,rows); atomic_csv_append(rq1_path,sigrows)
    atomic_csv_append(metrics_path,metrows); atomic_csv_append(runtime_path,timerows)
    atomic_csv_append(ratio_path,ratiorows)
    log("CORE phase complete.")

# --------------------------- R9 PREVALENCE x B ---------------------------
def run_r9():
    path=OUT/"R9_prevalence_budget/prevalence_budget_grid.csv"
    done=done_keys(path,["config","model_seed","pool_seed","rate","alpha","strategy","budget","canary_seed"])
    rows=[]
    Xa,Xb=get_unr_pools()
    for ps in POOL_SEEDS:
        nf=prepare_nf_for_pool(ps,MAX_ROWS)
        for src in SRC_UNR:
            Xs=nf[src][1].values; ys=nf[src][2]
            # train source once per model seed, but scaling target depends only on source
            for ms in MODEL_SEEDS:
                for rate in PREV_RATES:
                    Xt,yt=make_prevalence_pool(Xa,Xb,rate,30000+100*ps+int(rate*10000),MAX_ROWS)
                    try:
                        model,qt,Xtr,ytr,Xhd,yhd,Xt2,fit_sec=fit_realization(Xs,ys,Xt,"lightgbm",ms)
                    except Exception as e:
                        log(f"R9 fit error {src} r={rate}: {e}"); continue
                    rng=np.random.RandomState(200000+ps)
                    tperm=rng.permutation(len(Xt2))[:min(EVAL_N,len(Xt2))]
                    sperm=rng.permutation(len(Xhd))
                    for alpha in ALPHAS:
                        ns=int(EVAL_N*alpha); nt=EVAL_N-ns
                        ti=tperm[:min(nt,len(tperm))]; si=sperm[:min(ns,len(sperm))]
                        Xmix=np.vstack([Xt2[ti],Xhd[si]])[:EVAL_N]
                        ymix=np.r_[yt[ti],yhd[si]][:EVAL_N]
                        if len(np.unique(ymix))<2: continue
                        p=model.predict_proba(Xmix)[:,1]
                        true=safe_metric(roc_auc_score,ymix,p); coll=int(true<TAU)
                        config=f"UNR:{src}->UNR@{rate}"
                        for strat in KEY_STRATS:
                            for B in BUDGETS:
                                for cs in R9_CANARY_SEEDS:
                                    key=(config,ms,ps,rate,alpha,strat,B,cs)
                                    if key in done: continue
                                    idx,w=select_rule(strat,Xmix,p,B,cs)
                                    cm=canary_metrics(ymix,p,idx,w)
                                    rows.append(dict(config=config,source=src,model_seed=ms,pool_seed=ps,
                                                     canary_seed=cs,rate=rate,alpha=alpha,
                                                     strategy=strat,budget=B,true_auroc=true,
                                                     collapsed_auroc=coll,**cm))
                                    if len(rows)>=1000: atomic_csv_append(path,rows)
                    del model,qt,Xtr,Xhd,Xt2; gc.collect()
        del nf; gc.collect()
    atomic_csv_append(path,rows)
    log("R9 phase complete.")

# --------------------------- E3 POOL SIZE ---------------------------
def run_pool_sensitivity():
    path=OUT/"E3_pool_size/pool_size_results.csv"
    done=done_keys(path,["config","pool_size","model_seed","pool_seed","strategy","canary_seed"])
    rows=[]
    # representative rare-attack block: four sources -> UNR at 1.44%, alpha=0, B=50
    Xa,Xb=get_unr_pools()
    sizes=POOL_SIZES if ARGS.profile!="smoke" else [4_000,6_000,8_000]
    for size in sizes:
        for ps in POOL_SEEDS[:2 if ARGS.profile!="full" else len(POOL_SEEDS)]:
            nf=prepare_nf_for_pool(ps,size)
            for src in SRC_UNR:
                Xs=nf[src][1].values; ys=nf[src][2]
                Xt,yt=make_prevalence_pool(Xa,Xb,.0144,40000+ps,size)
                for ms in MODEL_SEEDS:
                    try:
                        model,qt,Xtr,ytr,Xhd,yhd,Xt2,fit_sec=fit_realization(Xs,ys,Xt,"lightgbm",ms)
                    except Exception as e:
                        log(f"E3 fit error {src} size={size}: {e}"); continue
                    n=min(EVAL_N,len(Xt2)); idxp=np.random.RandomState(ps).permutation(len(Xt2))[:n]
                    Xeval=Xt2[idxp]; yeval=yt[idxp]; p=model.predict_proba(Xeval)[:,1]
                    true=safe_metric(roc_auc_score,yeval,p); coll=int(true<TAU)
                    config=f"UNR:{src}->UNR@0.0144"
                    for strat in KEY_STRATS:
                        for cs in CANARY_SEEDS:
                            key=(config,size,ms,ps,strat,cs)
                            if key in done: continue
                            idx,w=select_rule(strat,Xeval,p,CORE_BUDGET,cs)
                            cm=canary_metrics(yeval,p,idx,w)
                            rows.append(dict(config=config,pool_size=size,model_seed=ms,pool_seed=ps,
                                             canary_seed=cs,strategy=strat,budget=CORE_BUDGET,
                                             true_auroc=true,collapsed_auroc=coll,**cm))
                    if len(rows)>=500: atomic_csv_append(path,rows)
                    del model,qt,Xtr,Xhd,Xt2; gc.collect()
            del nf; gc.collect()
    atomic_csv_append(path,rows)
    log("E3 pool sensitivity complete.")

# --------------------------- E5 DEPENDENCE ---------------------------
def draw_iid(y,B,rng):
    idx=rng.choice(len(y),size=min(B,len(y)),replace=False)
    z=y[idx]; return int(z.min()!=z.max())

def cluster_draw(y,clusters,B,rng):
    # sample whole clusters until >= B observations, then randomly truncate to B
    keys=np.array(pd.unique(clusters),dtype=object)
    order=rng.permutation(len(keys))
    chosen=[]
    for j in order:
        ids=np.where(clusters==keys[j])[0]
        chosen.extend(ids.tolist())
        if len(chosen)>=B: break
    if not chosen: return 0
    chosen=np.asarray(chosen,int)
    if len(chosen)>B: chosen=rng.choice(chosen,size=B,replace=False)
    z=y[chosen]
    return int(len(z)>1 and z.min()!=z.max())

def run_dependence():
    path=OUT/"E5_dependence/unraveled_observability_dependence.csv"
    if path.exists() and not ARGS.force:
        log("E5 already exists; skip.")
        return
    Xa,Xb,ma,mb=scan_unraveled(True,1500,0)
    meta=pd.concat([ma,mb],ignore_index=True)
    # clean required fields
    for c in ["src_ip","dst_ip"]:
        if c not in meta.columns: meta[c]="NA"
        meta[c]=meta[c].astype(str)
    if "bidirectional_first_seen_ms" not in meta.columns:
        raise RuntimeError("Unraveled timestamp metadata missing.")
    meta["ts"]=pd.to_numeric(meta["bidirectional_first_seen_ms"],errors="coerce")
    meta=meta[np.isfinite(meta.ts)].reset_index(drop=True)
    y=meta.y.to_numpy(int)
    empirical_r=float(y.mean())
    pair=(meta.src_ip+"|"+meta.dst_ip).to_numpy(object)
    rows=[]
    for B in BUDGETS:
        analytic=1-(1-empirical_r)**B-empirical_r**B
        for rep in range(DEP_REPS):
            rng=np.random.RandomState(500000+B*10000+rep)
            rows.append(dict(method="iid_empirical",window_min=0,budget=B,rep=rep,
                             both_classes=draw_iid(y,B,rng),empirical_r=empirical_r,
                             analytic_iid=analytic))
            rows.append(dict(method="ip_pair_cluster",window_min=0,budget=B,rep=rep,
                             both_classes=cluster_draw(y,pair,B,rng),empirical_r=empirical_r,
                             analytic_iid=analytic))
        for win in [1,5,15]:
            cluster=(meta.ts//(win*60*1000)).astype("Int64").astype(str).to_numpy(object)
            for rep in range(DEP_REPS):
                rng=np.random.RandomState(600000+win*100000+B*1000+rep)
                rows.append(dict(method="temporal_cluster",window_min=win,budget=B,rep=rep,
                                 both_classes=cluster_draw(y,cluster,B,rng),
                                 empirical_r=empirical_r,analytic_iid=analytic))
    pd.DataFrame(rows).to_csv(path,index=False)
    summary=(pd.DataFrame(rows).groupby(["method","window_min","budget"])
             .agg(empirical_both=("both_classes","mean"),
                  empirical_r=("empirical_r","first"),
                  analytic_iid=("analytic_iid","first"))
             .reset_index())
    summary["difference_vs_analytic"]=summary.empirical_both-summary.analytic_iid
    summary.to_csv(OUT/"E5_dependence/unraveled_observability_dependence_summary.csv",index=False)
    log("E5 dependence complete.")

# --------------------------- SUMMARIES ---------------------------
def summarize_outputs():
    lines=["# COMNET REVISION RUN SUMMARY","",f"Profile: **{ARGS.profile}**",""]
    checks=[
        ("core/core_canary_results.csv","Core detector-family/canary"),
        ("E1_detector_family/label_free_by_family.csv","E1 detector-family label-free"),
        ("E2_metrics/true_metrics_by_family.csv","E2 alternative metrics"),
        ("R6_runtime/acquisition_runtime.csv","R6 runtime"),
        ("E4_strat_fp2_ratio/fp2_ratio_results.csv","E4 fp2 ratio"),
        ("R9_prevalence_budget/prevalence_budget_grid.csv","R9 prevalence x budget"),
        ("E3_pool_size/pool_size_results.csv","E3 pool size"),
        ("E5_dependence/unraveled_observability_dependence_summary.csv","E5 dependence"),
    ]
    for rel,label in checks:
        p=OUT/rel
        if p.exists():
            try:
                n=sum(1 for _ in open(p,encoding="utf-8"))-1
            except Exception: n=-1
            lines.append(f"- ✅ {label}: `{rel}` ({n:,} rows)")
        else:
            lines.append(f"- ❌ {label}: missing `{rel}`")
    lines += ["","## Next step",
              "Zip the entire `COMNET_REVISION_RUN` folder and return it for reviewer-response analysis.",
              "Do not delete intermediate prediction parquet files."]
    (OUT/"REVISION_RUN_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    log("Summary written.")

# --------------------------- MAIN ---------------------------
def main():
    log("="*80)
    log(f"COMNET REVISION MASTER | profile={ARGS.profile} | phases={sorted(PHASES)}")
    log(f"Output: {OUT}")
    if "audit" in PHASES:
        capture_environment(); audit_paths()
    if "core" in PHASES:
        run_core()
    if "r9" in PHASES:
        run_r9()
    if "pool" in PHASES:
        run_pool_sensitivity()
    if "dependence" in PHASES:
        run_dependence()
    if "summary" in PHASES:
        summarize_outputs()
    log("ALL REQUESTED PHASES COMPLETE.")

if __name__=="__main__":
    main()
