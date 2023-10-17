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
    """
    Prédire si le patient semble malade ou non en retournant soit 1 soit 0
    
    Paramètres
    ---------
      attrs : pandas.dataframe
    """
    if attrs is None : # Le cas où le dataframe prend en entrée est NULL alors on retourne vide
      return
    estimation = getPrior(attrs) # Estimer si le patient est malade en utilisant l'estimation et son intervalle de confiance
    return 1 if estimation["min5pourcent"]<=estimation["estimation"]<=estimation["max5pourcent"] else 0

  def statsOnDF(self, df) : 
    """
    Renvoyer les 4 valeurs : vrai positif, vrai négatif, faux positif, faux négatif, ainsi que la précision et le rappel
    
    Paramètres
    ---------
      attrs : pandas.dataframe
    """
    estimation = self.estimClass(df)
    VP = len([i for i in df["target"] if i==1 and i==estimation]) # vrai posotif
    VN = len([i for i in df["target"] if i==0 and i==estimation]) # vrai négatif
    FP = len([i for i in df["target"] if i==0 and i!=estimation]) # faux positif
    FN = len([i for i in df["target"] if i==1 and i!=estimation]) # faux négatif
    return {
      "VP" : VP, "VN" : VN, "FP" : FP, "FN" : FN,
      "Précision" : np.mean(df["target"]), "Rappel" : estimation
    }




#####
# CLASSIFICATION PROBABILISTE À 2 DIMENSIONS
#####
# QUESTION 2.1 - PROBABILITÉS CONDITIONNELLES
#####
def P2D_l(df, attr) : 
  """
  Renvoyer les 4 valeurs : vrai positif, vrai négatif, faux positif, faux négatif, ainsi que la précision et le rappel
    
  Paramètres
  ---------
    attrs : pandas.dataframe
  """
  liste_val_unique_attr = np.unique(df[attr])
  df_target_1 = df[df["target"]==1][[attr, "target"]]
  df_target_0 = df[df["target"]==0][[attr, "target"]]
  p_attr_1 = dict()
  p_attr_0 = dict()
  
  for i in liste_val_unique_attr : 
    p_attr_1[i] = len(df_target_1[attr][df_target_1[attr]==i]) / len(df_target_1)
    p_attr_0[i] = len(df_target_0[attr][df_target_0[attr]==i]) / len(df_target_0)

  return {1 : p_attr_1, 0 : p_attr_0}







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