import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration du style (recommandé)
sns.set_theme(style="whitegrid")

# --- GÉNÉRATION DES DONNÉES SYNTHÉTIQUES (1000 clients) ---
np.random.seed(42)
n_clients = 1000

data = {
    'ID': range(1, n_clients + 1),
    'Score_Credit': np.random.normal(650, 100, n_clients).astype(int),
    'Age': np.random.randint(18, 80, n_clients),
    'Solde': np.random.uniform(0, 200000, n_clients),  # 0..200k MAD
    'Salaire': np.random.uniform(3000, 40000, n_clients),
    'Ville': np.random.choice(['Casablanca', 'Rabat', 'Tanger'], n_clients),
    'Genre': np.random.choice(['Homme', 'Femme'], n_clients),
    'Exited': np.random.choice([0, 1], n_clients, p=[0.8, 0.2])  # 0=Resté, 1=Parti
}
df = pd.DataFrame(data)

# Correction : clip Score_Credit entre 300 et 850
df['Score_Credit'] = df['Score_Credit'].clip(300, 850)

print("Aperçu des données :")
print(df.head())

# ==========================================
# Partie 1 : Distributions (Univariée)
# ==========================================

# Ex 1.1 Histogramme + KDE (Solde) couleur teal
plt.figure(figsize=(10, 5))
sns.histplot(df['Solde'], bins=12, kde=True, color="teal")
plt.title("Distribution des Soldes Clients")
plt.xlabel("Solde (MAD)")
plt.ylabel("Count")
plt.show()

# Ex 1.2 KDE comparatif Score_Credit selon Exited
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x="Score_Credit", hue="Exited", common_norm=False, fill=True, alpha=0.3)
plt.title("Densité du Score de Crédit (Clients Partis vs Restés)")
plt.xlabel("Score_Credit")
plt.ylabel("Densité")
plt.show()

# ==========================================
# Partie 2 : Variables Catégorielles
# ==========================================

# Ex 2.1 Countplot : clients partis vs restés selon Genre
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Genre", hue="Exited")
plt.title("Nombre de clients partis vs restés par Genre")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.show()

# Ex 2.2 Boxplot : salaires par Ville
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Ville", y="Salaire")
plt.title("Comparaison des Salaires par Ville (Boxplot)")
plt.xlabel("Ville")
plt.ylabel("Salaire")
plt.show()

# Ex 2.3 Violinplot : salaires par Ville
plt.figure(figsize=(9, 5))
sns.violinplot(data=df, x="Ville", y="Salaire", inner="quartile")
plt.title("Comparaison des Salaires par Ville (Violin Plot)")
plt.xlabel("Ville")
plt.ylabel("Salaire")
plt.show()

# ==========================================
# Partie 3 : Relations (Bivariée)
# ==========================================

# Ex 3.1 Scatter : Age vs Solde, coloré par Ville
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Age", y="Solde", hue="Ville", alpha=0.7)
plt.title("Relation Âge vs Solde par Ville")
plt.xlabel("Âge")
plt.ylabel("Solde")
plt.show()

# Ex 3.2 Regplot : Age vs Score_Credit (tendance linéaire)
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x="Age", y="Score_Credit", scatter_kws={"alpha": 0.6})
plt.title("Régression : Âge vs Score de Crédit")
plt.xlabel("Âge")
plt.ylabel("Score_Credit")
plt.show()

# ==========================================
# Partie 4 : Matrices & Vue d'ensemble
# ==========================================

# Ex 4.1 Heatmap : matrice de corrélation des variables numériques
num_cols = df.select_dtypes(include=np.number)
corr = num_cols.corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de Corrélation")
plt.show()

# Ex 4.2 Pairplot : vue d'ensemble colorée par Exited
sns.pairplot(df, vars=["Age", "Solde", "Score_Credit", "Salaire"], hue="Exited", corner=True)
plt.suptitle("Vue d'ensemble (Pairplot)", y=1.02)
plt.show()
