import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest  # Détection des anomalies
import streamlit as st
from fpdf import FPDF

# Fonction pour générer le rapport en PDF
def generer_rapport(data, taux_conformite, taux_defaut, defauts_par_type, tendances, anomalies, suggestions):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rapport de Contrôle Qualité", ln=True, align='C')

    pdf.cell(200, 10, txt=f"Taux de conformité : {taux_conformite:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Taux de défaut : {taux_defaut:.2f}%", ln=True)

    pdf.cell(200, 10, txt="Répartition des défauts : ", ln=True)
    for defaut, count in defauts_par_type.items():
        pdf.cell(200, 10, txt=f"{defaut}: {count}", ln=True)

    pdf.cell(200, 10, txt="Anomalies détectées : ", ln=True)
    for index, row in anomalies.iterrows():
        pdf.cell(200, 10, txt=f"ID: {index}, Type: {row.get('Type', 'N/A')}", ln=True)

    pdf.cell(200, 10, txt="Suggestions de correction : ", ln=True)
    for suggestion in suggestions:
        pdf.cell(200, 10, txt=suggestion, ln=True)

    file_path = "rapport_controle_qualite.pdf"
    pdf.output(file_path)
    return file_path

# Fonction pour charger le fichier Excel
def charger_dataset(uploaded_file):
    return pd.read_excel(uploaded_file)

# Fonction pour corriger les anomalies automatiquement
def corriger_anomalies(data):
    """
    Corrige les anomalies dans le dataset en remplaçant les valeurs anormales par des valeurs acceptables.
    """
    # Correction des anomalies d'humidité
    if 'Humidité (%)' in data.columns:
        data.loc[data['Humidité (%)'] < 6, 'Humidité (%)'] = 10  # Augmenter à une valeur par défaut
        data.loc[data['Humidité (%)'] > 90, 'Humidité (%)'] = 85  # Réduire à une valeur acceptable

    # Correction des anomalies de température
    if 'Température (°C)' in data.columns:
        data.loc[data['Température (°C)'] > 30, 'Température (°C)'] = 25  # Réduire à une valeur acceptable

    return data

# Fonction pour détecter les anomalies spécifiques
def detecter_anomalies(data):
    anomalies = pd.DataFrame()
    if 'Humidité (%)' in data.columns:
        anomalies = pd.concat([anomalies, data[data['Humidité (%)'] < 6]])
    if 'Température (°C)' in data.columns:
        anomalies = pd.concat([anomalies, data[data['Température (°C)'] > 30]])
    return anomalies

# Fonction pour suggérer des corrections pour les anomalies
def suggester_correction(anomalies):
    suggestions = []
    for index, row in anomalies.iterrows():
        if row['Humidité (%)'] < 6:
            suggestions.append(f"ID {index}: Augmenter l'humidité à 10%.")
        if row['Température (°C)'] > 30:
            suggestions.append(f"ID {index}: Réduire la température à 25°C.")
    return suggestions

# Charger le dataset (exemple avec un fichier téléchargé)
uploaded_file = st.file_uploader("Téléchargez le fichier Excel", type=['xlsx'])
if uploaded_file is not None:
    data = charger_dataset(uploaded_file)

    # Étape 1 : Corriger les anomalies automatiquement
    data = corriger_anomalies(data)

    # Étape 2 : Calcul des indicateurs clés de performance (KPIs)
    total_pieces = len(data)
    conform_pieces = data[data['Résultat QC'] == 'Conforme'].shape[0]
    non_conform_pieces = data[data['Résultat QC'] == 'Non conforme'].shape[0]
    taux_conformite = (conform_pieces / total_pieces) * 100
    taux_defaut = (non_conform_pieces / total_pieces) * 100

    # Étape 3 : Visualisation des défauts
    defauts_par_type = {}
    if 'Défaut' in data.columns:
        defauts_par_type = data['Défaut'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=defauts_par_type.index, y=defauts_par_type.values, ax=ax)
        ax.set_title('Répartition des défauts par type')
        ax.set_ylabel('Nombre de défauts')
        ax.set_xlabel('Type de défaut')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Étape 4 : Analyse des tendances dans le temps
    tendances = pd.DataFrame()
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])  # Conversion en format datetime
        tendances = data.groupby('Date')['Résultat QC'].value_counts().unstack()
        fig, ax = plt.subplots()
        tendances.plot(kind='line', figsize=(10, 6), ax=ax)
        ax.set_title('Tendances des résultats QC dans le temps')
        ax.set_ylabel('Nombre')
        ax.set_xlabel('Date')
        st.pyplot(fig)

    # Étape 5 : Détection des anomalies spécifiques (avant correction)
    anomalies_specifiques = detecter_anomalies(data)

    # Suggestions de correction pour les anomalies détectées
    suggestions = suggester_correction(anomalies_specifiques)

    # Tableau de bord interactif avec Streamlit
    st.title('Contrôle Qualité des Produits')
    st.write("### Indicateurs Clés de Performance")
    st.metric(label="Taux de conformité", value=f"{taux_conformite:.2f}%")
    st.metric(label="Taux de défaut", value=f"{taux_defaut:.2f}%")

    # Répartition des défauts
    st.write("### Répartition des défauts")
    if not defauts_par_type.empty:
        st.bar_chart(defauts_par_type)
    else:
        st.write("La colonne 'Défaut' n'existe pas dans le dataset.")

    # Analyse des tendances
    st.write("### Tendances des résultats QC dans le temps")
    if not tendances.empty:
        st.line_chart(tendances)
    else:
        st.write("La colonne 'Date' n'existe pas dans le dataset.")

    # Visualisation des anomalies
    st.write("### Anomalies spécifiques et suggestions de correction")
    if not anomalies_specifiques.empty:
        st.dataframe(anomalies_specifiques)
        st.write("### Suggestions de correction")
        for suggestion in suggestions:
            st.write(suggestion)
    else:
        st.write("Aucune anomalie spécifique détectée.")

    # Générer le rapport en PDF
    pdf_file = generer_rapport(data, taux_conformite, taux_defaut, defauts_par_type, tendances, anomalies_specifiques, suggestions)

    # Ajouter une fonctionnalité de téléchargement pour le fichier PDF
    st.write("### Télécharger le rapport")
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Télécharger le rapport PDF",
            data=f,
            file_name=pdf_file,
            mime='application/octet-stream',
        )

    # Téléchargement du fichier corrigé
    corrected_file_path = "dataset_corrige.xlsx"
    data.to_excel(corrected_file_path, index=False)
    with open(corrected_file_path, "rb") as f:
        st.download_button(
            label="Télécharger le fichier corrigé",
            data=f,
            file_name="dataset_corrige.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# Instructions pour exécuter Streamlit
print("\nPour lancer le tableau de bord Streamlit, exécutez la commande suivante :")
print("streamlit run votre_script.py")
