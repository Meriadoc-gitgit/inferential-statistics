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
class APrioriClassifier(AbstractClassifier) : 
  
  def __init__(self) : 
    pass

  def estimClass(self, attrs) : 
    if attrs is None : 
      return
    estimation = getPrior(attrs)
    return estimation["min5pourcent"]<=estimation["estimation"]<=estimation["max5pourcent"]




class ML2DClassifier(APrioriClassifier) : 
  pass

class MAP2DClassifier(APrioriClassifier) : 
  pass

class MLNaiveBayesClassifier(APrioriClassifier) : 
  pass

class MAPNaiveBayesClassifier(APrioriClassifier) : 
  pass

class MAPTANlassifier(APrioriClassifier) : 
  pass




class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier) : 
  pass

class ReducedMAPNaiveBayesClassifier(MLNaiveBayesClassifier) : 
  pass