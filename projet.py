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

  def estimClass(self, dic) : 
    """
    Prédire si le patient semble malade ou non en retournant soit 1 soit 0
    
    Paramètres
    ---------
      attrs : pandas.dataframe
    """
    if dic is None : # Le cas où le dataframe prend en entrée est NULL alors on retourne vide
      return
    estimation = getPrior(dic)["estimation"] # Estimer si le patient est malade en utilisant l'estimation et son intervalle de confiance
    return 0 if estimation<0.5 else 1

  def statsOnDF(self, df) : 
    """
    Renvoyer les 4 valeurs : vrai positif, vrai négatif, faux positif, faux négatif, ainsi que la précision et le rappel
    
    Paramètres
    ---------
      attrs : pandas.dataframe
    """
    estimation = self.estimClass(df)
    print(estimation)
    #print(estimation)
    VP = len([i for i in df["target"] if i==1 and i==estimation]) # vrai positif
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
  Calculer dans le dataframe la probabilité P(attr\ target) sous la forme d'un dictioinnaire asociant à la valeur t un dictionnaire associant à la valeur a la probabilité P(attr=a\ target=t)
    
  Paramètres
  ---------
    df : dataframe
    attrs : string
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

def P2D_p(df, attr) : 
  """
  Calculer dans le dataframe la probabilité P(target\ attr) sous la forme d'un dictionnaire associant à la valeur a un dictionnaire associant à la valeur t la probabilité P(target=t \ attr=a)
    
  Paramètres
  ---------
    df : dataframe
    attrs : string
  """
  liste_val_unique_attr = np.unique(df[attr])
  res = dict()

  for i in liste_val_unique_attr : 
    D = dict()
    data = df[df[attr]==i][[attr,"target"]]
    p_attr_1 = len([i for i in data["target"] if i==1]) / len(data)
    p_attr_0 = len([i for i in data["target"] if i==0]) / len(data)
    D[1] = p_attr_1
    D[0] = p_attr_0
    res[i] = D
  return res






class ML2DClassifier(APrioriClassifier) : 
  def __init__(self, df, attr) : 
    self.attr = attr
    self.p2dl = P2D_l(df, attr)
    print(self.p2dl)

  def estimClass(self, dic) : 
    l = dic[self.attr]
    print(l)
    print(self.p2dl[0][l])
    print(self.p2dl[1][l])
    if self.p2dl[0][l] < self.p2dl[1][l] : return 1
    return 0
    
  







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