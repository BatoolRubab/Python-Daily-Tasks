import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# =======================
# 1. Load FMRI Dataset
# =======================
fmri = sns.load_dataset("fmri")
print(fmri.head())

# Plot a basic line plot for signal over time
sns.lineplot(x="timepoint", y="signal", data=fmri)
plt.show()

# Line plot using hue to separate by event type
sns.lineplot(x="timepoint", y="signal", data=fmri, hue="event")
plt.show()

# Line plot with both hue and line style for different events
sns.lineplot(x="timepoint", y="signal", data=fmri, hue="event", style="event")
plt.show()

# Line plot with markers added for each point
sns.lineplot(x="timepoint", y="signal", data=fmri, hue="event", style="event", markers=True)
plt.show()

# =======================
# 2. Bar Plot with Pokémon Dataset
# =======================
sns.set_style("whitegrid")
pokemon = pd.read_csv('pokemon.csv')
print(pokemon.head())

# Barplot: Speed by Legendary status and Generation
sns.barplot(x="is_legendary", y="speed", hue="generation", data=pokemon)
plt.show()

# Barplot with custom color palette
sns.barplot(x="is_legendary", y="speed", palette="rocket", data=pokemon)
plt.show()

# =======================
# 3. Scatter Plot with Iris Dataset
# =======================
iris = pd.read_csv("Iris.csv")
print(iris.head())

# Scatter plot: SepalLength vs PetalLength by Species
sns.scatterplot(x="SepalLengthCm", y="PetalLengthCm", data=iris, hue="Species", style="Species")
plt.show()

# =======================
# 4. Histogram and Joint Plot with Diamonds Dataset
# =======================
dia = pd.read_csv("diamonds.csv")
print(dia.head())

# Distribution plot for Price with KDE
sns.displot(dia["price"], kde=True, bins=10, color="green")
plt.show()

# Joint plot with regression line for Carat vs Price
sns.jointplot(x="carat", y="price", color="olive", data=dia, kind="reg")
plt.show()

# =======================
# 5. Boxplot for Diamond Prices by Carat Range
# =======================
dia['carat_bin'] = pd.cut(dia['carat'], bins=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5],
                          labels=['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-5'])

# Boxplot of price by carat range
plt.figure(figsize=(10, 6))
sns.boxplot(x="carat_bin", y="price", data=dia)
plt.title("Diamond Price by Carat Range")
plt.xlabel("Carat Range")
plt.ylabel("Price")
plt.show()

# =======================
# 6. Game of Thrones Battles Dataset (EDA)
# =======================
battle = pd.read_csv("battles.csv")

# Rename columns for clarity
battle.rename(columns={"attacker_1": "primary_attacker", "defender_1": "primary_defender"}, inplace=True)

# Battle analysis: attacker king vs attacker size
sns.set(rc={"figure.figsize": (13, 5)})
sns.barplot(x="attacker_king", y="attacker_size", data=battle)
plt.show()

# Battle analysis: defender king vs defender size
sns.barplot(x="defender_king", y="defender_size", data=battle)
plt.show()

# Count plot of battles by attacker king and battle type
sns.countplot(x="attacker_king", hue="battle_type", data=battle)
plt.show()

# =======================
# 7. Character Deaths Dataset
# =======================
death = pd.read_csv("character-deaths.csv")
print(death)
print(death.shape)

# Count values by gender and nobility
print(death["Gender"].value_counts())
print(death["Nobility"].value_counts())

# Count plot of deaths by year
sns.countplot(x="Death Year", data=death)
plt.show()

# Count plot of allegiances (rotated for readability)
sns.set(rc={'figure.figsize': (30, 10)})
sns.countplot(x="Allegiances", data=death)
plt.title("Character Deaths by Allegiance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
