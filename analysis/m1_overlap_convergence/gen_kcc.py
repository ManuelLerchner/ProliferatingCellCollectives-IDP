import sys, os, glob
sys.path.insert(0,".")
import phi_vs_r as P
import numpy as np, pandas as pd
SPD="."
# one stable grid run per kcc (warns=0), all ran to R=20
picks={2e4:"stab_k2e4_c1.0",5e4:"stab_k5e4_c0.5",1e5:"stab_k1e5_c0.3",
       2e5:"stab_k2e5_c0.18",5e5:"stab_k5e5_c0.04"}
rows=[]
for k,run in picks.items():
    dd=glob.glob(os.path.join(SPD,"stab",run,"vtk_output_*"))
    if not dd: print("MISSING",run); continue
    R,phi=P.phi_at_R(dd[0]+"/data",20.0)
    rows.append(dict(kcc=k,R=R,phi_center=phi,mode="soft"))
# hard ref at R=20
dd=glob.glob(os.path.join(SPD,"runs","hard_cfl0.5","vtk_output_*"))
R,phi=P.phi_at_R(dd[0]+"/data",20.0); rows.append(dict(kcc=2e4,R=R,phi_center=phi,mode="hard"))
df=pd.DataFrame(rows); df.to_csv(SPD+"/kcc_phi.csv",index=False)
print(df.to_string(index=False))
