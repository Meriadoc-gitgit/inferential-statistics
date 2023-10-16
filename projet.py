#####
# INFORMATION DE BINOMES
#####
# Hoang Thuy Duong VU
# HalimatoudDIALLO
####

# Import necessary libraries
import numpy as np

# Import utils
from utils import *


#####
# CLASSIFICATION A PRIORI
#####
# QUESTION 1.1 - CALCUL DE PROBA A PRIORI
#####

def getPrior(df) :
  std = np.std(df["target"])/np.sqrt(len(df))
  mu = np.mean(df["target"])

  return {
    "estimation" : mu,
    "min5pourcent" :  mu-1.96*std,
    "max5pourcent" : mu+1.96*std
  }



#####
# QUESTION 1.2 - POO DANS LA HIERACHIE DES `CLASSIFIER`
#####
class AprioriClassifier(AbstractClassifier) : 
  pass




class ML2DClassifier(AprioriClassifier) : 
  pass

class MAP2DClassifier(AprioriClassifier) : 
  pass

class MLNaiveBayesClassifier(AprioriClassifier) : 
  pass

class MAPNaiveBayesClassifier(AprioriClassifier) : 
  pass

class MAPTANlassifier(AprioriClassifier) : 
  pass




class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier) : 
  pass

class ReducedMAPNaiveBayesClassifier(MLNaiveBayesClassifier) : 
  pass