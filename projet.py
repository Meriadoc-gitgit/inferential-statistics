#####
# INFORMATION DE BINOMES
#####
# Hoang Thuy Duong VU
# HalimatoudDIALLO
####
import numpy as np


#####
# QUESTION 1.1
#####

def getPrior(df) :
  std = np.std(df["target"])/np.sqrt(len(df))
  mu = np.mean(df["target"])

  return {
    "estimation" : mu,
    "min5pourcent" :  mu-1.96*std,
    "max5pourcent" : mu+1.96*std
  }