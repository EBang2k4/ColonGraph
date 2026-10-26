"""Locked Stage5 transfer and refinement schedules; no data-dependent choices."""
import math


def photometric(iteration, profile='J'):
    if profile in ('P','P1','P2'):
        from utils.stage8 import photometric as stage8_photometric
        return stage8_photometric(iteration,profile)
    if profile in ('M','N','O'):
        from utils.stage6 import photometric as stage6_photometric
        return stage6_photometric(iteration)
    if profile not in ('A','J','K','L'):
        raise ValueError(profile)
    start,final_l1,final_mse = (10000,.15,15.) if profile in ('K','L') else (15000,.20,12.)
    tau=min(1.,max(0.,(iteration-start)/(30000-start)))
    return (.80-.65*tau,.20,2.+13.*tau) if profile in ('K','L') else (.80-.60*tau,.20,2.+10.*tau)


def dynamics(iteration, profile='J'):
    if profile in ('P','P1','P2'):
        from utils.stage8 import dynamics as stage8_dynamics
        return stage8_dynamics(iteration,profile)
    if profile in ('M','N','O'):
        from utils.stage6 import dynamics as stage6_dynamics
        return stage6_dynamics(iteration)
    if profile not in ('A','J','K','L'):
        raise ValueError(profile)
    if profile=='A':return 1.,.25,.01
    start,floor=(10000,.08) if profile in ('K','L') else (15000,.10)
    t=min(1.,max(0.,(iteration-start)/(30000-start)))
    eta=floor+(1-floor)*.5*(1+math.cos(math.pi*t))
    lo,hi,depth,knn=(8000,22000,.01,.0005) if profile=='L' else (10000,25000,.05,.002)
    t=min(1.,max(0.,(iteration-lo)/(hi-lo)))
    return eta,.25+(.0 if profile=='A' else (depth-.25)*t),.01+(.0 if profile=='A' else (knn-.01)*t)
