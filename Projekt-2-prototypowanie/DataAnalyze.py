import os
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns


#pkt 1.
annotations_path = './annotations/Annotation'  # zmień ścieżkę na rzeczywistą

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = {
        'filename': root.find('filename').text,
        'width': int(root.find('size/width').text),
        'height': int(root.find('size/height').text),
        'depth': int(root.find('size/depth').text),
        'breed': root.find('object/name').text,
        'xmin': int(root.find('object/bndbox/xmin').text),
        'ymin': int(root.find('object/bndbox/ymin').text),
        'xmax': int(root.find('object/bndbox/xmax').text),
        'ymax': int(root.find('object/bndbox/ymax').text)
    }
    return data



data = []
for root_dir, _, files in os.walk(annotations_path):
    for filename in files:
        file_path = os.path.join(root_dir, filename)
        data.append(parse_xml(file_path))


df = pd.DataFrame(data)


print("Podstawowe informacje o danych:")
print(df.info())


plt.figure(figsize=(12, 6))
df['breed'].value_counts().plot(kind='bar')
plt.title("Rozkład ras psów")
plt.xlabel("Rasa")
plt.ylabel("Liczba obrazów")
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['width'], bins=20)
plt.title("Rozkład szerokości obrazów")
plt.xlabel("Szerokość")

plt.subplot(1, 2, 2)
sns.histplot(df['height'], bins=20)
plt.title("Rozkład wysokości obrazów")
plt.xlabel("Wysokość")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['width', 'height']])
plt.title("Wykres pudełkowy szerokości i wysokości obrazów")
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df[['width', 'height', 'depth', 'xmin', 'ymin', 'xmax', 'ymax']].corr(), annot=True, cmap='coolwarm')
plt.title("Macierz korelacji zmiennych numerycznych")
plt.show()


print("Brakujące wartości w danych:")
print(df.isnull().sum())



import sweetviz as sv


report = sv.analyze(df)
report.show_html("dogs_dataset_report.html")
