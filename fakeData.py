import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

# Generate 10,000 records
data = []
for i in range(1, 10001):
    date = fake.date_between(start_date='-2y', end_date='today')
    produit = f"P-{random.randint(1, 50):03}"
    lot = random.randint(100, 999)
    poids = random.randint(480, 520)
    longueur = random.randint(95, 105)
    largeur = random.randint(45, 55)
    hauteur = random.randint(20, 30)
    temperature = random.randint(10, 30)
    pression = random.randint(99000, 105000)
    defauts = ["Aucun", "Rayure", "Défaut de forme", "Problème d'emballage"]
    defaut = random.choice(defauts)
    qc_result = "Conforme" if defaut == "Aucun" else "Non conforme"
    quality = random.choice(["Excellent", "Bon", "Moyen", "Mauvais"])
    retour = random.choice(["Oui", "Non"])
    date_retour = fake.date_between(start_date=date, end_date='today') if retour == "Oui" else None
    commentaire = "Defaut de fabrication" if defaut != "Aucun" else None
    humidite = random.choice([10, 85])
    emplacement = "Production"
    etat = "En cours"
    
    data.append([i, date, produit, lot, poids, longueur, largeur, hauteur, temperature, pression, 
                 defaut, qc_result, quality, retour, date_retour, commentaire, humidite, emplacement, etat])

# Create a DataFrame
df = pd.DataFrame(data, columns=[
    "ID", "Date", "Produit", "Lot", "Poids (g)", "Longueur (mm)", "Largeur (mm)", "Hauteur (mm)",
    "Température (°C)", "Pression (Pa)", "Défaut Inspection", "Résultat QC", "Quality", "Retour",
    "Date_Retour", "Commentaire", "Humidité (%)", "Emplacement", "État"
])

# Save to Excel
df.to_excel("generated_data.xlsx", index=False)

print("10,000 records successfully generated and saved as 'generated_data.xlsx'")
