import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================
# --- Dataset 1 : Ventes ---
data_ventes = {
    'Date': pd.to_datetime([
        '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05',
        '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
        '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'
    ]),
    'Ville': [
        'Agadir', 'Casablanca', 'Tanger', 'Marrakech', 'Agadir',
        'Casablanca', 'Tanger', 'Marrakech', 'Agadir', 'Casablanca',
        'Marrakech', 'Agadir', 'Casablanca', 'Tanger', 'Marrakech'
    ],
    'Categorie': [
        'Électronique', 'Vêtements', 'Électronique', 'Meubles', 'Vêtements',
        'Électronique', 'Vêtements', 'Meubles', 'Électronique', 'Vêtements',
        'Meubles', 'Électronique', 'Vêtements', 'Électronique', 'Meubles'
    ],
    'Montant': [12000, 450, 3400, 8000, 300, 15000, 600, 7200, 2500, 500, 9000, 4000, 700, 3200, 5500],
    'Age_Client': [22, 45, 34, 58, 19, 36, 25, 62, 28, 41, 55, 23, 33, 29, 60],
    'Genre': ['H', 'F', 'H', 'F', 'F', 'H', 'F', 'H', 'H', 'F', 'F', 'H', 'F', 'H', 'F']
}
df = pd.DataFrame(data_ventes)

# --- Dataset 2 : Performance RH ---
data_perf = {
    'Metrique': ['Vitesse', 'Précision', 'Satisfaction Client', 'Assiduité', 'Innovation'],
    'Employe_A': [4, 5, 3, 5, 2],
    'Employe_B': [3, 4, 5, 4, 5]
}
df_perf = pd.DataFrame(data_perf)

print("--- Données Chargées ---")

# ==========================================
# 2. PARTIE A : DATA ENGINEERING (TRANSFORMATION)
# ==========================================

# 1. Logique Métier (Apply)
def calcul_commission(montant):
    if montant > 10000:
        return 500
    elif 5000 <= montant <= 10000:
        return 200
    else:
        return 50

df['Commission'] = df['Montant'].apply(calcul_commission)

# 2. Discrétisation (Cut)
bins = [0, 30, 50, 100]
labels = ['Jeune', 'Adulte', 'Senior']
df['Segment_Age'] = pd.cut(df['Age_Client'], bins=bins, labels=labels)

print(df[['Montant', 'Commission']].head(10))
print(df[['Age_Client', 'Segment_Age']].head(10))

# 3. Agrégation (GroupBy)
df_agg = df.groupby('Ville')['Montant'].agg(['sum', 'mean']).reset_index()
print("\n--- Performance par Ville ---")
print(df_agg)

# 4. Tableau Croisé (Pivot Table)
df_pivot = df.pivot_table(index='Ville', columns='Categorie', values='Montant', aggfunc='sum', fill_value=0)
print("\n--- Pivot Table (Ville x Categorie) ---")
print(df_pivot)

# ==========================================
# 3. PARTIE B : VISUALISATION (REPORTING)
# ==========================================
plt.style.use('ggplot')

# --- AXE 1 : ANALYSE TEMPORELLE ---
df_date = df.groupby('Date')['Montant'].sum().reset_index()

# 1. Line Plot (Tendance)
plt.figure(figsize=(10, 5))
plt.plot(df_date['Date'], df_date['Montant'], marker='o', linestyle='-', color='blue')
plt.title("Évolution du Chiffre d'Affaires")
plt.xlabel("Date")
plt.ylabel("Montant (DH)")
plt.grid(True)
plt.show()

# 2. Area Plot (Volume)
plt.figure(figsize=(10, 5))
plt.fill_between(df_date['Date'], df_date['Montant'], color='skyblue', alpha=0.4)
plt.plot(df_date['Date'], df_date['Montant'], color='Slateblue', alpha=0.6)
plt.title("Volume Cumulé des Ventes")
plt.show()

# 3. Histogramme (Distribution)
plt.figure(figsize=(8, 5))
plt.hist(df['Montant'], bins=5, color='orange', edgecolor='black')
plt.title("Distribution des Montants de Transaction")
plt.xlabel("Montant (DH)")
plt.ylabel("Fréquence")
plt.show()

# --- AXE 2 : COMPARAISONS ---
df_cat = df.groupby('Categorie')['Montant'].sum()
plt.figure(figsize=(8, 5))
plt.bar(df_cat.index, df_cat.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title("CA Total par Catégorie")
plt.show()

# 5. Scatter Plot (Age vs Montant)
plt.figure(figsize=(8, 5))
plt.scatter(df['Age_Client'], df['Montant'], c='purple', alpha=0.6, s=100)
plt.title("Corrélation : Âge vs Montant")
plt.xlabel("Âge")
plt.ylabel("Montant Acheté")
plt.show()

# 6. Pie Chart (Segments)
counts_age = df['Segment_Age'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(counts_age, labels=counts_age.index, autopct='%1.1f%%', startangle=140,
        colors=['gold', 'lightgreen', 'lightcoral'])
plt.title("Répartition des Ventes par Segment d'Âge")
plt.show()

# --- AXE 3 : AVANCÉ ---
cross_tab = pd.crosstab(df['Ville'], df['Genre'])
cross_tab.plot(kind='bar', stacked=True, color=['#e74c3c', '#3498db'], figsize=(8, 6))
plt.title("Répartition H/F par Ville")
plt.ylabel("Nombre de transactions")
plt.xticks(rotation=45)
plt.show()

# 8. Radar Plot (Performance RH)
labels_r = df_perf['Metrique']
num_vars = len(labels_r)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += [angles[0]]

val_a = df_perf['Employe_A'].tolist() + [df_perf['Employe_A'].tolist()[0]]
val_b = df_perf['Employe_B'].tolist() + [df_perf['Employe_B'].tolist()[0]]

plt.figure(figsize=(6, 6))
ax = plt.subplot(polar=True)

ax.plot(angles, val_a, linewidth=1, linestyle='solid', label='Employé A')
ax.fill(angles, val_a, 'b', alpha=0.1)

ax.plot(angles, val_b, linewidth=1, linestyle='solid', label='Employé B')
ax.fill(angles, val_b, 'r', alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_r)
plt.title("Comparaison des Compétences")
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.show()
