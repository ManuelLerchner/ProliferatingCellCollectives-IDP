import sys, os, glob, re
sys.path.insert(0,"$SP".replace("$SP","."))
import phi_vs_r as P
import numpy as np, pandas as pd, glob as g

SPD="."

def datadir(run):
    d=g.glob(os.path.join(SPD,"runs",run,"vtk_output_*"))
    return d[0]+"/data" if d else None

# dt-sweep (fixed k=2e4) at R=50 + hard
conv={"soft_R50_cfl1.0_k2e4":("soft",1.0),"soft_R50_cfl0.5_k2e4":("soft",0.5),
      "soft_R50_cfl0.25_k2e4":("soft",0.25),"soft_R50_cfl0.1_k2e4":("soft",0.1),
      "hard_R50_cfl0.5":("hard",0.5)}
rows=[]
for run,(mode,cfl) in conv.items():
    dd=datadir(run)
    if not dd: continue
    c=P.run_curve(dd); c["run"]=run; c["mode"]=mode; c["cfl"]=cfl
    rows.append(c)
if rows:
    conv_df=pd.concat(rows,ignore_index=True)
    conv_df.to_csv(SPD+"/conv_curves.csv",index=False)
    print("conv_curves.csv:",len(conv_df),"rows,",conv_df.run.nunique(),"runs")

# kcc sweep: phi at common R across stiffness (stable runs)
kcc_runs={"soft_cfl0.1_k2e4":2e4,"soft_cfl0.1_k1e5":1e5,"soft_cfl0.05_k5e5":5e5,"hard_cfl0.5":2e4}
Rc=20.0; krows=[]
for run,k in kcc_runs.items():
    dd=datadir(run)
    if not dd: continue
    R,phi=P.phi_at_R(dd,Rc)
    krows.append(dict(run=run,kcc=k,R=R,phi_center=phi,mode=("hard" if "hard" in run else "soft")))
pd.DataFrame(krows).to_csv(SPD+"/kcc_phi.csv",index=False)
print("kcc_phi.csv written")
print(pd.DataFrame(krows).to_string(index=False))
