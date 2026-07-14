# Extract phi_center vs colony radius across ALL snapshots of a run.
import sys, os, glob, re
sys.path.insert(0,"/Users/manuellerchner/git/ProliferatingCellCollectives-IDP/analysis")
import numpy as np, pandas as pd
from parse_vtu import vtu_to_dataframe

def sph_area(L, r=0.25): return 2*r*(L-2*r)+np.pi*r**2

def load_iter(datadir, it):
    fs=glob.glob(os.path.join(datadir,f"particles_{it:07d}_rank_*.vtu"))
    dfs=[vtu_to_dataframe(open(f).read()) for f in fs]
    return pd.concat(dfs, ignore_index=True)

def iters(datadir):
    fs=glob.glob(os.path.join(datadir,"particles_*_rank_*.vtu"))
    return sorted(set(int(re.search(r"particles_(\d+)_rank",f).group(1)) for f in fs))

def phi_center(df, rmax_bin=6.0, bin_size=2.0):
    r=np.sqrt(df["x"]**2+df["y"]**2).values
    L=df["lengths_x"].values; A=sph_area(L)
    edges=np.arange(0,rmax_bin+bin_size,bin_size); vals=[]
    for i in range(len(edges)-1):
        m=(r>=edges[i])&(r<edges[i+1])
        if m.sum()>0: vals.append(A[m].sum()/(np.pi*(edges[i+1]**2-edges[i]**2)))
    return np.mean(vals) if vals else np.nan

def colony_radius(df):
    r=np.sqrt(df["x"]**2+df["y"]**2).values
    return np.percentile(r,99)  # robust outer radius

def run_curve(datadir):
    rows=[]
    for it in iters(datadir):
        try:
            df=load_iter(datadir,it)
            if len(df)<5: continue
            rows.append((it, colony_radius(df), phi_center(df), len(df)))
        except Exception: pass
    return pd.DataFrame(rows, columns=["iter","R","phi_center","N"])

def phi_at_R(datadir, targetR):
    c=run_curve(datadir)
    if len(c)==0: return np.nan, np.nan
    i=(c["R"]-targetR).abs().idxmin()
    return c.loc[i,"R"], c.loc[i,"phi_center"]

if __name__=="__main__":
    d=sys.argv[1]; tR=float(sys.argv[2]) if len(sys.argv)>2 else None
    dd=glob.glob(os.path.join(d,"vtk_output_*"))[0]+"/data"
    c=run_curve(dd)
    print(c.to_string(index=False))
    if tR: print(f"\nphi_center at R~{tR}:", phi_at_R(dd,tR))
