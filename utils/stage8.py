"""Locked Stage8 schedules: KNN99 constant on both depth modalities."""
from utils.stage6 import photometric as original_photometric, dynamics as original_dynamics
PROFILES=('P','P1','P2')


def photometric(iteration,profile):
    if profile not in PROFILES:raise ValueError(profile)
    if profile!='P1':return original_photometric(iteration)
    t=min(1.,max(0.,(iteration-15000)/15000.))
    return .8-.4*t,.2,2.+4.*t


def dynamics(iteration,profile):
    if profile not in PROFILES:raise ValueError(profile)
    eta,estimated_depth,_=original_dynamics(iteration)
    return eta,estimated_depth if profile=='P' else .25,.01
