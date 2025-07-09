"""
plan pour le fichier: 

1. importer les bibliothèques nécessaires comme pandas, readcsv, geosoft, etc. 

2. ouvrir les fichiers data (peut être aussi changer dans les bons formats)

3. cibler les colonnes/lignes nécesaires dans les fichiers data (IP) et extraire

4. créer un nouveau fichier pour accueillir les données extraites

"""

# étape 1: importer les bonnes bibliotèques
import pandas as pd
import csv 
import numpy as np
import matplotlib.pyplot as plt

 # test pour voir si le script fonctionne
print("Début du script de planification")

# étape 2: ouvrir les fichiers data
# il faut probablement changer les format des fichiers 
df = pd.read_csv('IN0048-0556@041824_083500.IAB') # ne fonctionne pas mais c'est pour le plan

# étape 3: cibler les colonnes/lignes nécessaires
# je veux passer dans chaque ligne et cibler les colonnes IP et par la suite les ajouter au np array IP
# probablement pas la meilleure façon de faire mais c'est pour le plan
# initialiser un tableau pour les données IP
IP = np.zeros((20, len(df)))  
# faire une liste pour les colonnes IP qui va aller dans le tableau
IP_list = []
# extraire les colonnes IP
for i in range(0, 20):
    IP_list.append(df[f"IP_{i}"])
# convertir à csv
df_IP = pd.DataFrame(IP_list).T

# étape 4: créer un nouveau fichier pour accueillir les données extraites
# encore une fois ça ne vas pas fct mais c'est pour le plan
df_IP.to_csv('extracted_IP_data.csv')