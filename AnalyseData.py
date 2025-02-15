import pandas as pd
import matplotlib.pyplot as plt

# Lit le fichier Excel et le stocke dans la variable data sous forme de DataFrame.
data = pd.read_excel('Dataset_Controle_Qualite.xlsx')
# Afficher les premières lignes du dataset 
print(data.head())
# Calcul des KPIs
total_pieces = len(data)
conform_pieces = data[data['Résultat QC'] == 'Conforme'].shape[0]
non_conform_pieces = data[data['Résultat QC'] == 'Non conforme'].shape[0]
taux_conformite = (conform_pieces / total_pieces) * 100
taux_defaut = (non_conform_pieces / total_pieces) * 100

# Affichage des KPIs
print(f"Taux de conformité : {taux_conformite:.2f}%")
print(f"Taux de défaut : {taux_defaut:.2f}%")

# Visualisation des défauts par type
if 'Défaut' in data.columns:
    defauts_par_type = data['Défaut'].value_counts()
    defauts_par_type.plot(kind='bar', title='Répartition des défauts par type')
    plt.show()
else:
    print("La colonne 'Défaut' n'existe pas dans le dataset.")

# Temps de cycle de livraison (exemple fictif)
temps_preparation = 2  # heures
temps_transport = 3    # heures
temps_controle_qualite = 1  # heures
temps_cycle_livraison = temps_preparation + temps_transport + temps_controle_qualite
print(f"Temps de cycle de livraison : {temps_cycle_livraison} heures")

# Exemple de taux de retours (données fictives)
retours = 2  # nombre de produits retournés
taux_retours = (retours / total_pieces) * 100
print(f"Taux de retours des produits : {taux_retours:.2f}%")

# Satisfaction client (données fictives)
scores_satisfaction = [4, 5, 3, 4, 5]  # scores sur 5
taux_satisfaction = sum(scores_satisfaction) / len(scores_satisfaction)
print(f"Taux de satisfaction client : {taux_satisfaction:.2f}/5")

# Exemple de temps moyen de résolution des non-conformités (données fictives)
temps_resolution_non_conformites = [5, 3, 4]  # heures pour chaque non-conformité
temps_moyen_resolution = sum(temps_resolution_non_conformites) / len(temps_resolution_non_conformites)
print(f"Temps moyen de résolution des non-conformités : {temps_moyen_resolution:.2f} heures")
