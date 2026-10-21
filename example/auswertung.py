# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:39:56 2019

@author: foersterronny
"""

# Standard Libraries
from __future__ import division, unicode_literals, print_function # For compatibility of Python 2 and 3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# own Library
import nanoobjectdetection_1_0 as nd
import time #track runtime

import atexit
import resource
import sys, threading

def _print_peak_memory():
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print("[PEAK MEMORY] Peak process RSS: {:.2f} GiB".format(peak_gib), flush=True)

atexit.register(_print_peak_memory)

def _meminfo_gb():
    """(total, available) system RAM in GB (Linux)"""
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return info["MemTotal"] / 1024**2, info["MemAvailable"] / 1024**2

def _rss_gb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS"):
                return int(l.split()[1]) / 1024**2
    return float("nan")

def _oom_message(prefix):
    total, avail = _meminfo_gb()
    return ("\n!!! {}: system RAM total {:.1f} GB, available {:.1f} GB, this process uses {:.1f} GB.\n"
            "!!! If the process is killed now, the Linux OOM killer did it (out of RAM; check 'dmesg | grep -i oom').\n"
            "!!! Reduce the data: set ROI frame_max / use_num_frame in the parameter file, or use a machine with more RAM.\n"
            ).format(prefix, total, avail, _rss_gb())

def _log_stage(stage):
    total, available = _meminfo_gb()
    print("[{}] {} | RAM {:.1f} GB available of {:.1f} GB; process RSS {:.1f} GB".format(
        time.strftime("%H:%M:%S"), stage, available, total, _rss_gb()
    ), flush=True)

def _memory_watchdog(min_avail_gb=1.0, interval=0.5):
    warned = False
    while True:
        if _meminfo_gb()[1] < min_avail_gb and not warned:
            print(_oom_message("LOW MEMORY WARNING"), file=sys.stderr, flush=True)
            warned = True
        time.sleep(interval)

def _excepthook(exc_type, exc, tb):
    if issubclass(exc_type, MemoryError):
        print(_oom_message("OUT OF MEMORY (MemoryError)"), file=sys.stderr, flush=True)
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = _excepthook
threading.Thread(target=_memory_watchdog, daemon=True).start()
print(_oom_message("Memory at start"))

start_time=time.time()
#%% path of parameter file
# this must be replaced by any json file
ParameterJsonFile = "/home/tim/onedrive/IPHT/Python/NanoObjectDetection/example/parameter.json"


#%% check if the python version and the library are good
_log_stage("Checking Python and NanoObjectDetection environment")
nd.CheckSystem.CheckAll(ParameterJsonFile)
settings = nd.handle_data.ReadJson(ParameterJsonFile)

#%% read in the raw data to numpy
_log_stage("Reading TIFF stack")
rawframes_np = nd.handle_data.ReadData2Numpy(ParameterJsonFile)
print("Loaded stack: shape={}, dtype={}, size={:.2f} GiB".format(
    rawframes_np.shape, rawframes_np.dtype, rawframes_np.nbytes / 1024**3
), flush=True)


#%% choose ROI (Region of Interest) if wanted
_log_stage("Applying ROI and supersampling")
rawframes_super = nd.handle_data.RoiAndSuperSampling(settings, ParameterJsonFile, rawframes_np)


#%% standard image preprocessing
_log_stage("Preprocessing image stack")
rawframes_pre, rawframes_int, static_background = nd.PreProcessing.Main(rawframes_super, ParameterJsonFile)


#%% help with the parameters for finding objects 
_log_stage("Adjusting particle-finding parameters")
settings = nd.handle_data.ReadJson(ParameterJsonFile)
nd.AdjustSettings.Main(rawframes_super, rawframes_pre, rawframes_int, ParameterJsonFile)
    

#%% find the objects
_log_stage("Finding particles in frames")
obj_all = nd.get_trajectorie.FindSpots(rawframes_np, rawframes_int, ParameterJsonFile)


#%% identify static objects
# find trajectories of very slow diffusing (maybe stationary) objects
_log_stage("Identifying stationary particles")
traj_fixed = nd.get_trajectorie.Link(obj_all, ParameterJsonFile, SearchFixedParticles = True)


#%% cut trajectories if a moving particle comes too close to a stationary object
_log_stage("Removing detections near stationary particles")
obj_moving = nd.get_trajectorie.RemoveSpotsInNoGoAreas(obj_all, traj_fixed, ParameterJsonFile)
 

#%% form trajectories of valid particle positions
_log_stage("Linking moving-particle detections")
traj_moving = nd.get_trajectorie.Link(obj_moving, ParameterJsonFile, SearchFixedParticles = False)


#%% drift correction
_log_stage("Correcting trajectory drift")
traj_no_drift = nd.Drift.Main(traj_moving, ParameterJsonFile, PlotGlobalDrift = True)


#%% calculate the MSD and process to diffusion and diameter - LONGITUDINAL
eval_dim = "x"

_log_stage("Analyzing longitudinal (x) trajectories")
sizes_df_lin_x, any_successful_check, traj_final_x  = nd.CalcDiameter.Main2(traj_no_drift, ParameterJsonFile, MSD_fit_Show = True, eval_dim = eval_dim)

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin_x, any_successful_check, eval_dim = eval_dim)


#%% repeat for TRANSVERSAL direction
eval_dim = "y"

_log_stage("Analyzing transverse (y) trajectories")
sizes_df_lin_y, any_successful_check, traj_final_y = nd.CalcDiameter.Main2(traj_no_drift, ParameterJsonFile, MSD_fit_Show = True, eval_dim = eval_dim)

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin_y, any_successful_check, eval_dim = eval_dim)


#%% repeat for 2d evaluation
eval_dim = "2d"

_log_stage("Analyzing 2D trajectories")
sizes_df_lin2, any_successful_check, traj_final2 = nd.CalcDiameter.Main2(traj_no_drift, ParameterJsonFile, MSD_fit_Show = True, eval_dim = eval_dim)

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin2, any_successful_check, eval_dim = eval_dim)


#%% visualize results - LONGITUDINAL
_log_stage("Writing longitudinal summary")
nd.CalcDiameter.SummaryEval(settings, rawframes_pre, obj_moving, traj_moving, traj_no_drift, traj_final_x, sizes_df_lin_x)


#%% Animation
_log_stage("Creating animation")
nd.Animate.AnimateDiameterAndRawData_Big(rawframes_np, sizes_df_lin_x.copy(), traj_final_x, ParameterJsonFile, DoSave = False)

end_time=time.time()
print("Total runtime: {:.2f} seconds".format(end_time-start_time))