import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка датасета
df = pd.read_csv('marvel_characters_dataset.csv')  # убедись, что файл с таким именем есть

# Пример анализа
print("Количество строк в датасете:", len(df))
print(df.head())

# Построим график (если есть колонка 'name')
if 'name' in df.columns:
    top_chars = df['name'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_chars.values, y=top_chars.index, palette='mako')
    plt.title('Топ-10 персонажей Marvel по количеству упоминаний')
    plt.xlabel('Количество')
    plt.ylabel('Имя')
    plt.tight_layout()
    plt.show()
else:
    print("Колонка 'name' не найдена в датасете")