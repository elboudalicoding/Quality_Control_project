import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # Détection des anomalies et modèle de prédiction
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
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
    data = pd.read_excel(uploaded_file)
    if 'Emplacement' not in data.columns:
        data['Emplacement'] = 'Production'  # Initialiser l'emplacement à 'Production'
    if 'État' not in data.columns:
        data['État'] = 'En cours'  # Initialiser l'état à 'En cours'
    return data

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

# Fonction pour mettre à jour l'emplacement et l'état des produits
def mettre_a_jour_etat(data, ids_produits, nouvel_emplacement, nouvel_etat):
    for id_produit in ids_produits:
        data.loc[data['ID'] == id_produit, 'Emplacement'] = nouvel_emplacement
        data.loc[data['ID'] == id_produit, 'État'] = nouvel_etat
    return data

# Fonction pour entraîner un modèle de machine learning pour prédire les défauts
def entrainer_modele(data):
    # Préparation des données
    features = data.drop(columns=['Résultat QC'])
    
    # Convertir les colonnes de type datetime en format numérique
    for col in features.select_dtypes(include=['datetime64']).columns:
        features[col] = features[col].astype('int64') / 10**9  # Convertir en secondes depuis l'époque
    
    # Sélectionner uniquement les colonnes numériques
    features = features.select_dtypes(include=[np.number])
    
    target = data['Résultat QC'].apply(lambda x: 1 if x == 'Non conforme' else 0)
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    modele = RandomForestClassifier(random_state=42)
    modele.fit(X_train, y_train)
    
    # Évaluation du modèle
    y_pred = modele.predict(X_test)
    rapport = classification_report(y_test, y_pred, target_names=['Conforme', 'Non conforme'], output_dict=True)
    
    # Affichage du rapport de classification sous forme de tableau
    st.write("### Rapport de classification")
    st.table(pd.DataFrame(rapport).transpose())
    
    # Affichage de la matrice de confusion
    st.write("### Matrice de confusion")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(modele, X_test, y_test, display_labels=['Conforme', 'Non conforme'], ax=ax)
    st.pyplot(fig)
    
    return modele

# Charger le dataset (exemple avec un fichier téléchargé)
uploaded_file = st.file_uploader("Téléchargez le fichier Excel", type=['xlsx'])
if uploaded_file is not None:
    data = charger_dataset(uploaded_file)

    # Étape 1 : Corriger les anomalies automatiquement
    data = corriger_anomalies(data)

    # Calcul des indicateurs clés de performance (KPIs)
    total_pieces = len(data)
    conform_pieces = data[data['Résultat QC'] == 'Conforme'].shape[0]
    non_conform_pieces = data[data['Résultat QC'] == 'Non conforme'].shape[0]
    taux_conformite = (conform_pieces / total_pieces) * 100
    taux_defaut = (non_conform_pieces / total_pieces) * 100

    # Détection des anomalies spécifiques (avant correction)
    anomalies_specifiques = detecter_anomalies(data)

    # Suggestions de correction pour les anomalies détectées
    suggestions = suggester_correction(anomalies_specifiques)

    # Visualisation des défauts
    defauts_par_type = {}
    if 'Défaut Inspection' in data.columns:
        defauts_par_type = data['Défaut Inspection'].value_counts()

    # Analyse des tendances dans le temps
    tendances = pd.DataFrame()
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])  # Conversion en format datetime
        tendances = data.groupby('Date')['Résultat QC'].value_counts().unstack()

    # Onglets pour naviguer entre les sections
    st.write("## Navigation")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Accueil", "Mise à jour de l'emplacement et de l'état des produits", "Calcul des indicateurs clés de performance", "Visualisation des défauts", "Analyse des tendances", "Détection des anomalies spécifiques", "Entraînement du modèle de prédiction des défauts"])

    with tab1:
        st.write("### Accueil")
        st.write("Bienvenue dans l'application de contrôle qualité des produits.")

    with tab2:
        st.write("### Mise à jour de l'emplacement et de l'état des produits")
        ids_produits = st.multiselect("Sélectionnez les IDs des produits", data['ID'].unique())
        nouvel_emplacement = st.selectbox("Sélectionnez le nouvel emplacement", ['Production', 'Transport', 'Client'])
        nouvel_etat = st.selectbox("Sélectionnez le nouvel état", ['En cours', 'En transit', 'Livré'])

        if st.button("Mettre à jour"):
            data = mettre_a_jour_etat(data, ids_produits, nouvel_emplacement, nouvel_etat)
            st.success(f"Produits {ids_produits} mis à jour avec succès : Emplacement = {nouvel_emplacement}, État = {nouvel_etat}")

            # Afficher le tableau de bord après la mise à jour
            st.write("### Tableau de bord après mise à jour")
            st.write("#### Indicateurs Clés de Performance")
            st.metric(label="Taux de conformité", value=f"{taux_conformite:.2f}%")
            st.metric(label="Taux de défaut", value=f"{taux_defaut:.2f}%")

            st.write("#### Répartition des défauts")
            if 'Défaut Inspection' in data.columns:
                fig = px.bar(defauts_par_type, x=defauts_par_type.index, y=defauts_par_type.values, labels={'x': 'Type de défaut', 'y': 'Nombre de défauts'}, title='Répartition des défauts par type')
                st.plotly_chart(fig, key="defauts_par_type")
            else:
                st.write("La colonne 'Défaut Inspection' n'existe pas dans le dataset.")

            st.write("#### Analyse des tendances dans le temps")
            if not tendances.empty:
                fig = px.line(tendances, labels={'value': 'Nombre', 'Date': 'Date'}, title='Tendances des résultats QC dans le temps')
                st.plotly_chart(fig, key="tendances")
            else:
                st.write("La colonne 'Date' n'existe pas dans le dataset.")

        # Visualisation du suivi des produits
        st.write("### Suivi des produits à travers la chaîne logistique")
        if 'Emplacement' in data.columns and 'État' in data.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='Emplacement', hue='État', ax=ax)
            ax.set_title('Suivi des produits à travers la chaîne logistique')
            ax.set_ylabel('Nombre de produits')
            ax.set_xlabel('Emplacement')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("Les colonnes 'Emplacement' et 'État' n'existent pas dans le dataset.")

    with tab3:
        st.write("### Calcul des indicateurs clés de performance (KPIs)")
        st.metric(label="Taux de conformité", value=f"{taux_conformite:.2f}%")
        st.metric(label="Taux de défaut", value=f"{taux_defaut:.2f}%")

    with tab4:
        st.write("### Visualisation des défauts")
        if 'Défaut Inspection' in data.columns:
            fig = px.bar(defauts_par_type, x=defauts_par_type.index, y=defauts_par_type.values, labels={'x': 'Type de défaut', 'y': 'Nombre de défauts'}, title='Répartition des défauts par type')
            st.plotly_chart(fig, key="defauts_par_type_tab4")
        else:
            st.write("La colonne 'Défaut Inspection' n'existe pas dans le dataset.")

    with tab5:
        st.write("### Analyse des tendances dans le temps")
        if not tendances.empty:
            fig = px.line(tendances, labels={'value': 'Nombre', 'Date': 'Date'}, title='Tendances des résultats QC dans le temps')
            st.plotly_chart(fig, key="tendances_tab5")
        else:
            st.write("La colonne 'Date' n'existe pas dans le dataset.")

    with tab6:
        st.write("### Détection des anomalies spécifiques")
        if not anomalies_specifiques.empty:
            st.dataframe(anomalies_specifiques)
            st.write("### Suggestions de correction")
            for suggestion in suggestions:
                st.write(suggestion)
        else:
            st.write("Aucune anomalie spécifique détectée.")

    with tab7:
        st.write("### Entraînement du modèle de prédiction des défauts")
        if st.button("Entraîner le modèle"):
            modele = entrainer_modele(data)
            st.success("Modèle entraîné avec succès")

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