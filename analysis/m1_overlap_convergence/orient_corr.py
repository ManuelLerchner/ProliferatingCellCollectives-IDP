# Nematic orientational correlation function C(r) = <cos 2(theta_i - theta_j)>
# and correlation length xi (C(xi)=1/e), from a colony snapshot.
import sys, os, glob
sys.path.insert(0,"/Users/manuellerchner/git/ProliferatingCellCollectives-IDP/analysis")
sys.path.insert(0,".")
import numpy as np, phi_vs_r as P
from scipy.spatial import cKDTree

def corr_curve(df, rmax=25.0, dr=0.5, core_frac=0.8):
    x=df["x"].values; y=df["y"].values; th=df["orientation_angle"].values
    R=np.sqrt(x**2+y**2); Rc=np.percentile(R,99)
    # restrict reference cells to the colony core to avoid boundary effects
    core=R < core_frac*Rc
    pos=np.column_stack([x,y])
    tree=cKDTree(pos)
    edges=np.arange(0,rmax+dr,dr); nb=len(edges)-1
    ssum=np.zeros(nb); cnt=np.zeros(nb)
    cx,cy,cth=x[core],y[core],th[core]
    for i in np.where(core)[0]:
        idx=tree.query_ball_point(pos[i],rmax)
        idx=[j for j in idx if j!=i]
        if not idx: continue
        rj=np.sqrt((x[idx]-x[i])**2+(y[idx]-y[i])**2)
        cc=np.cos(2*(th[i]-th[idx]))
        b=np.clip((rj/dr).astype(int),0,nb-1)
        np.add.at(ssum,b,cc); np.add.at(cnt,b,1)
    C=ssum/np.maximum(cnt,1)
    rc=0.5*(edges[:-1]+edges[1:])
    return rc, C, Rc

def xi_from(rc,C):
    # correlation length: first crossing of 1/e
    target=1/np.e
    below=np.where(C<target)[0]
    if len(below)==0: return np.nan
    k=below[0]
    if k==0: return rc[0]
    # linear interp between k-1 and k
    r0,r1=rc[k-1],rc[k]; c0,c1=C[k-1],C[k]
    return r0+(target-c0)*(r1-r0)/(c1-c0)

def xi_expfit(rc,C):
    # fit C(r) = exp(-r/xi) over the decaying part (C>0.05)
    from scipy.optimize import curve_fit
    m=(C>0.05)&(rc<=rc[np.argmax(C<0.05)] if np.any(C<0.05) else np.ones_like(rc,bool))
    m=C>0.05
    if m.sum()<3: return np.nan
    try:
        p,_=curve_fit(lambda r,xi: np.exp(-r/xi), rc[m], C[m], p0=[2.0], maxfev=5000)
        return float(p[0])
    except Exception: return np.nan

def analyze(run_dir, navg=3):
    dd=glob.glob(os.path.join(run_dir,"vtk_output_*"))[0]+"/data"
    its=P.iters(dd)
    Cs=[]; Rc=0; N=0
    for it in its[-navg:]:
        df=P.load_iter(dd,it)
        rc,C,Rc=corr_curve(df); Cs.append(C); N=len(df)
    C=np.mean(Cs,axis=0)
    return rc,C,Rc,N,xi_from(rc,C),xi_expfit(rc,C)

if __name__=="__main__":
    for run in sys.argv[1:]:
        rc,C,Rc,N,xi,xie=analyze(run)
        print(f"{os.path.basename(run)}: N={N} R={Rc:.1f} xi_1/e={xi:.2f} xi_expfit={xie:.2f}")
        print("  C(r):", " ".join(f"{c:.2f}" for c in C[:15]))
