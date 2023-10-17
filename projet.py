#####
# INFORMATION DE BINOMES
#####
# Hoang Thuy Duong VU
# HalimatoudDIALLO
####

# Import necessary libraries
import numpy as np
import pandas as pd

# Import utils
import utils
from utils import *

train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")


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
    #print("dic",dic)
    estimation = getPrior(dic)["estimation"] # Estimer si le patient est malade en utilisant l'estimation et son intervalle de confiance
    return 0 if estimation<0.5 else 1

  def statsOnDF(self, df) : 
    """
    Renvoyer les 4 valeurs : vrai positif, vrai négatif, faux positif, faux négatif, ainsi que la précision et le rappel
    
    Paramètres
    ---------
      attrs : pandas.dataframe
    """
    VP = 0; VN = 0; FP = 0; FN = 0
    for i in range(len(df)) : 

      estimation = self.estimClass(utils.getNthDict(df, i))
      if df["target"][i]==1 and df["target"][i]==estimation : 
        VP+=1
      if df["target"][i]==0 and df["target"][i]==estimation : 
        VN+=1
      if df["target"][i]==0 and df["target"][i]!=estimation : 
        FP+=1
      if df["target"][i]==1 and df["target"][i]!=estimation : 
        FN+=1
    return {
      "VP" : VP, "VN" : VN, "FP" : FP, "FN" : FN,
      "Précision" : VP/(VP+FP), "Rappel" : VP/(VP+FN)
    }




#####
# CLASSIFICATION PROBABILISTE À 2 DIMENSIONS
#####
# QUESTION 2.1 - PROBABILITÉS CONDITIONNELLES
#####
def P2D_l(df, attr) : 
  """
  Calculer dans le dataframe la probabilité P(attr \ target) sous la forme d'un dictioinnaire asociant à la valeur t un dictionnaire associant à la valeur a la probabilité P(attr=a \ target=t)
    
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
  Calculer dans le dataframe la probabilité P(target \ attr) sous la forme d'un dictionnaire associant à la valeur a un dictionnaire associant à la valeur t la probabilité P(target=t \ attr=a)
    
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





#####
# QUESTION 2.2 - CLASSIFIEURS 2D PAR MAXIMUM DE VRAISEMBLANCE
#####
class ML2DClassifier(APrioriClassifier) : 
  def __init__(self, df, attr) : 
    self.attr = attr
    self.p2dl = P2D_l(df, attr)

  def estimClass(self, df) : 
    l = df[self.attr]
    if self.p2dl[0][l] < self.p2dl[1][l] : return 1
    return 0


#####
# QUESTION 2.3 - PROBABILITÉS CONDITIONNELLES
#####
class MAP2DClassifier(APrioriClassifier) : 
  def __init__(self, df, attr) : 
    self.attr = attr
    self.p2dp = P2D_p(df, attr)

  def estimClass(self, df) :
    l = df[self.attr]
    if self.p2dp[l][0] < self.p2dp[l][1] : return 1
    return 0



#####
# QUESTION 2.4 - COMPARAISON
#####





#####
# COMPLEXITÉS
#####
# FONCTION POUR L'AFFICHAGE
#####
def convert_octets_to_gbmokb(octets):
  if octets<1024 : 
    return 0,0,0,0
  # Conversion factors
  bytes_per_kb = 1024
  bytes_per_mb = bytes_per_kb * 1024
  bytes_per_gb = bytes_per_mb * 1024

  # Calculate the values
  gigabytes = octets // bytes_per_gb
  octets %= bytes_per_gb
  megabytes = octets // bytes_per_mb
  octets %= bytes_per_mb
  kilobytes = octets // bytes_per_kb
  octets %= bytes_per_kb

  return gigabytes, megabytes, kilobytes, octets


#####
# QUESTION 3.1 - COMPLEXITÉ EN MÉMOIRE
#####
def nbParams(df, attr=list(train.columns)) : 
  comp = 1
  for i in attr : 
    comp*=len(np.unique(df[i]))
  
  gb, mb, kb, octets = convert_octets_to_gbmokb(comp*8)

  string = str(len(attr)) + " variable(s) : "+str(comp*8)+" octets"

  # Affichage des octets
  if gb !=0 or mb!=0 or kb!=0: 
    string += " = "
  if gb!=0 : 
    string+=str(gb)+"go "
  if mb!=0 : 
    string+=str(mb)+"mo "
  if kb!=0 : 
    string+=str(kb)+"ko "
  if octets!=0 or (octets==0 and (kb!=0 or mb!=0 or gb!=0)) : 
    string+=str(octets)+"o"
  print(string)
  return comp*8



#####
# QUESTION 3.2 - COMPLEXITÉ EN MÉMOIRE SOUS HYPOTHÈSE D'INDÉPENDANCE COMPLÈTE
#####
def nbParamsIndep(df) : 
  attr = list(df.columns)
  comp = np.sum([len(np.unique(df[i])) for i in attr])*8

  gb, mb, kb, octets = convert_octets_to_gbmokb(comp*8)

  string = str(len(attr)) + " variable(s) : "+str(comp*8)+" octets"

  # Affichage des octets
  if gb !=0 or mb!=0 or kb!=0: 
    string += " = "
  if gb!=0 : 
    string+=str(gb)+"go "
  if mb!=0 : 
    string+=str(mb)+"mo "
  if kb!=0 : 
    string+=str(kb)+"ko "
  if octets!=0 or (octets==0 and (kb!=0 or mb!=0 or gb!=0)) : 
    string+=str(octets)+"o"
  print(string)
  return comp




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