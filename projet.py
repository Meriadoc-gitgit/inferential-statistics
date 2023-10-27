#####
# INFORMATION DE BINOMES
#####
# Hoang Thuy Duong VU | 21110221
# Halimatou DIALLO | 21114613
####

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import utils
import utils
from utils import * # Import les classes dans utils pour l'héritage de classes présentes dans ce fichier

# Importation de base de données afin de faciliter l'implémentation des noms de colonnes uniques sans devoir tout réécrire
train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")


#####
# 1 - CLASSIFICATION A PRIORI
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

  def estimClass(self, dic=None) : 
    """
    Prédire si le patient semble malade ou non en retournant soit 1 soit 0
    
    Paramètres
    ---------
      attrs : pandas.dataframe
    """
    #print("dic",dic)
    #estimation = getPrior(dic)["estimation"] # Estimer si le patient est malade en utilisant l'estimation et son intervalle de confiance
    #return 0 if estimation<0.5 else 1
    return 1

  def statsOnDF(self, df=None) : 
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
# 2 - CLASSIFICATION PROBABILISTE À 2 DIMENSIONS
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
# Voici notre réponsi ci-desssous :
"""
REDO !!!
À partir des résultats des tests de fonctions ci-dessus, on a obtenu : 
- APrioriClassifier avec le taux de précision et de rappel qui varie entre [0.7,0.9] et le nombre d'erreur supérieur à 10
- ML2DClassifier avec le taux de précision et de rappel qui varie entre [0.7,0.9] et le nombre d'erreur supérieur à 10
- MAP2DClassifier avec le taux de précision et de rappel qui varie entre [0.8,0.9]
et le nombre d'erreur supérieur à 15

On trouve bien que APrioriClassifier semble être le meilleur choix, car il a la meilleure précision en comparant avec les 2 autres méthodes de classification. Plus le nombre d'erreur augmente, plus le taux de précision diminue, et à partir de cette relation on peut déterminer le meilleur classificateur à choisir. 
"""





#####
# 3 - COMPLEXITÉS
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
  comp = np.sum([len(np.unique(df[i])) for i in attr])

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
# QUESTION 3.3 - INDÉPENDANCE PARTIELLE
#####
# 3.3.a. PREUVE
#####
# Voici notre réponsi ci-desssous :
"""
Pour prouver l'indépendance conditionnelle de A par rapport à C sachant B, on montre que :
  P(A | B, C) = P(A | B)
Cela signifie que l'information sur C n'a pas d'impact sur A une fois que l'on connaît B. 
"""



#####
# 3.3.b. COMPLEXITÉ EN INDÉPENDANCE PARTIELLE
#####
# Voici notre réponsi ci-desssous :
"""
Si les 3 variables A, et C ont 5 valeurs, 
- Pour savoir la taille mémoire en octet nécessaire pour représenter cette distribution sans l'indépendance conditionnelle, on applique la fonction nbParams qui renvoie 5^3*8 octet au total.
- Pour savoir la taille mémoire en octet nécessaire pour représenter cette distribution avec l'indépendance conditionnelle, on applique la fonction nbParamsIndep qui renvoie len(np.unique([(i,j,k) avec i dans A, j dans B, k dans C]))*8 octet au total, qui devra être inférieur ou égale à celle sans l'indépendance conditionnelle.
"""





#####
# 4 - MODÈLES GRAPHIQUES
#####
# QUESTION 4.1 - EXEMPLES
#####
# Voici notre réponsi ci-desssous :
"""
Supposons 5 variables A, B, C, D, E. 
Dans une représentation graphique de la relation entre ces 5 variables, s'ils sont complètement indépandantes conditionnellement, il n'existe aucun circuit. En revanche, aucun indépendance est observé, il existe un circuit passant par ces 5 variables. 
À noter que plusieurs représentations graphiques sont possibles pour ces 2 cas principals. Pour les graphes de version plus simple, l'indépendance des 5 variables peut être représentés par une arborescence d'une ligne droite passant de A à E. Or, pour la représentation la plus simple d'un circuit, il suffit d'ajouter un arc arrière de E à A, dont un graphe orienté ayant un arc arrière contient un circuit. 

Fonctions d'affichage de graphe d'indépendance :
utils.drawGraphHorizontal("A->B;B->C;C->D;D->E")
utils.drawGraphHorizontal("A->B;B->C;C->D;D->E;E->A")
"""



#####
# QUESTION 4.3 - MODÈLE GRAPHIQUE ET NAIVE BAYES
#####
# 4.3.a
#####
def drawNaiveBayes(df, attr) : 
  heading_attr = list(df.columns)
  string = ""
  for i in range(len(heading_attr)) : 
    if heading_attr[i]!=attr : 
      if i==0 : 
        string += attr + "->" + str(heading_attr[i])
      else : string += ";" + attr + "->" + str(heading_attr[i])
  return utils.drawGraph(string)




#####
# 4.3.b
#####
def nbParamsNaiveBayes(df, attr, headings=list(train.columns)) : 
  if len(headings)==0 : 
    comp = 2
  else : 
    comp = -1

    for i in headings : 
      data = df.loc[:, [attr,i]]
      comp+=len(np.unique(data[i]))

    comp*=2

  gb, mb, kb, octets = convert_octets_to_gbmokb(comp*8)

  string = str(len(headings)) + " variable(s) : "+str(comp*8)+" octets"

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
# QUESTION 4.4 - CLASSIFIER NAIVE BAYES
#####
class MLNaiveBayesClassifier(APrioriClassifier) : 
  def __init__(self, df) : 
    list_attr = list(df.columns)
    pd2l = dict()
    for attr in list_attr : 
      if attr!="target" :
        pd2l[attr] = P2D_l(df, attr) 
    self.p2dl = pd2l
    self.df = df

  def estimProbas(self, data) : 
    """Calcule la vraisemblance"""
    prob_0 = 1; prob_1 = 1
    for (k,v) in data.items() :
      if k in list(self.df.columns) and k!="target": 
        if v in self.p2dl[k][0] or v in self.p2dl[k][1]: 
          prob_0*=self.p2dl[k][0][v]
          prob_1*=self.p2dl[k][1][v]
        else : 
          prob_0*=0
          prob_1*=0
    return {0 : prob_0, 1 : prob_1}

  def estimClass(self, data) : 
    prob = self.estimProbas(data)
    return 1 if prob[1]>prob[0] else 0


class MAPNaiveBayesClassifier(APrioriClassifier) : 
  def __init__(self, df) : 
    list_attr = list(df.columns)
    pd2l = dict()
    for attr in list_attr : 
      if attr!="target" :
        pd2l[attr] = P2D_l(df, attr) 
    self.p2dl = pd2l
    self.df = df
    self.p = np.sum(df["target"]) / len(df)

  def estimProbas(self, data) : 
    """Calcule la vraisemblance"""
    prob_0, prob_1 = 1-self.p, self.p
    for (k,v) in data.items() : 
      if k in list(self.df.columns) and k!="target":
        prob_0*=self.p2dl[k][0][v] if v in self.p2dl[k][0] else 0
        prob_1*=self.p2dl[k][1][v] if v in self.p2dl[k][1] else 0

    pa = prob_0 + prob_1
    if pa!=0 : 
      prob_0/=pa; prob_1/=pa
    return {0 : prob_0, 1 : prob_1}

  def estimClass(self, data) : 
    prob = self.estimProbas(data)
    return 1 if prob[1]>prob[0] else 0
    







#####
# 5 - FEATURE SELECTION DANS LE CADRE DU CLASSIFIER NAIVE BAYES
# (Cours8 - page 11->15)
#####
# QUESTION 5.1 
#####
# Importer scipy.stats.chi2_contigency
from scipy.stats import chi2_contingency

def isIndepFromTarget(df, attr, x) :
  """vérifie si `attr` est indépendant de `target` au seuil de x%."""
  data = pd.crosstab(df[attr], df['target'])
  _, p, _, _ = chi2_contingency(data)
  return 1 if p>x else 0
  


class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier) : 
  def __init__(self, df, x) : 
    liste_attr = list(df.columns)
    liste_attr_indep = [attr for attr in liste_attr if not isIndepFromTarget(df, attr, x)]
    data = df.loc[:, liste_attr_indep]
    self.df = data
    super(ReducedMLNaiveBayesClassifier, self).__init__(data)
    self.x = x

  def draw(self) : 
    return drawNaiveBayes(self.df, "target")

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier) : 
  def __init__(self, df, x) : 
    liste_attr = list(df.columns)
    liste_attr_indep = [attr for attr in liste_attr if not isIndepFromTarget(df, attr, x)]
    data = df.loc[:, liste_attr_indep]
    self.df = data
    super(ReducedMAPNaiveBayesClassifier, self).__init__(data)
    self.x = x

  def draw(self) : 
    return drawNaiveBayes(self.df, "target")






#####
# 6 - EVALUATION DES CLASSIFIEURS
#####
# QUESTION 6.1 
#####
# Voici notre réponsi ci-desssous :
"""
insert response
"""



#####
# Import les libraries graphiques necessaires
#####
import matplotlib as mpl
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="hls", font="sans-serif", font_scale=1.4) 

mpl.rcParams['figure.figsize'] = (12, 10)


#####
# QUESTION 6.2
#####
def mapClassifiers(dic, df) : 
  for (k,v) in dic.items() : 
    cl = v
    stat = cl.statsOnDF(df)
    precision = stat["Précision"]
    rappel = stat["Rappel"]
    plt.scatter(precision, rappel, label=k, marker='x')
    plt.annotate(k, (precision+.001, rappel), fontsize=15, ha='left', va='center')






#####
# 7 - SOPHISTICATION DU MODÈLE (BONUS)
#####
# QUESTION 7.1 - CALCUL DES INFORMATION MUTUELLES
#####
def MutualInformation(df, x, y) : 
  joint_counts = df.groupby([y, x]).size().reset_index(name="count")
  joint_counts["P(y, x)"] = joint_counts["count"] / len(df)
  p_y = []; p_x = []
  for i in range(len(joint_counts)) : 
    p_y.append(len([j for j in df[y] if j==joint_counts[y][i]])/len(df))
    p_x.append(len([j for j in df[x] if j==joint_counts[x][i]])/len(df))
  joint_counts["P(y)"] = p_y
  joint_counts["P(x)"] = p_x

  return np.sum([joint_counts["P(y, x)"][i]*np.log2(joint_counts["P(y, x)"][i]/(joint_counts["P(y)"][i]*joint_counts["P(x)"][i])) for i in range(len(joint_counts))])
  

def ConditionalMutualInformation(df, x, y, z) : 
  joint_counts = df.groupby([x,y,z]).size().reset_index(name="count")
  joint_counts["P(x, y, z)"] = joint_counts["count"] / len(df)
  p_z = []
  p_x_z = []; p_y_z = []
  for i in range(len(joint_counts)) : 
    p_z.append(len([j for j in train[z] if j==joint_counts[z][i]])/len(train))

  joint_counts["P(z)"] = p_z

  joint_counts2 = train.groupby([x,z]).size().reset_index(name="count")
  joint_counts2["P(x, z)"] = joint_counts2["count"] / len(train)
  

  p_x_z = []
  for i in range(len(joint_counts)) : 
    for j in range(len(joint_counts2)) : 
      if joint_counts[x][i]==joint_counts2[x][j] and joint_counts[z][i]==joint_counts2[z][j] : 
        p_x_z.append(joint_counts2["P(x, z)"][j])

  joint_counts["P(x, z)"] = p_x_z

  joint_counts3 = train.groupby([y,z]).size().reset_index(name="count")
  joint_counts3["P(y, z)"] = joint_counts3["count"] / len(train)

  p_y_z = []
  for i in range(len(joint_counts)) : 
    for j in range(len(joint_counts3)) : 
      if joint_counts[y][i]==joint_counts3[y][j] and joint_counts[z][i]==joint_counts3[z][j] : 
        p_y_z.append(joint_counts3["P(y, z)"][j])

  joint_counts["P(y, z)"] = p_y_z

  return np.sum([joint_counts["P(x, y, z)"][i]*np.log2((joint_counts["P(z)"][i]*joint_counts["P(x, y, z)"][i])/(joint_counts["P(x, z)"][i]*joint_counts["P(y, z)"][i])) for i in range(len(joint_counts))])




#####
# QUESTION 7.2 - CALCUL DE LA MATRICE DES POIDS
#####
def MeanForSymetricWeights(cmis) : 
  if (len(cmis)!=len(cmis[0])) : print("Warning : The given matrix is not symetric"); return
  return np.sum([np.sum(i) for i in cmis])/(len(cmis)*len(cmis[0])-len(cmis[0]))


def SimplifyConditionalMutualInformationMatrix(cmis) : 
  mean = MeanForSymetricWeights(cmis)
  for i in range(len(cmis)) : 
    for j in range(len(cmis[i])) : 
      cmis[i][j] = 0 if cmis[i][j]<mean else cmis[i][j]




class MAPTANClassifier(APrioriClassifier) : 
  pass
