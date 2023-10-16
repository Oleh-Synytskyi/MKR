import requests
import os
import csv
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


lattitude = 50.958817  # широта                        27.248347161385226, 33.840686962333194
longitude = 10.952814  # довгота
direction = 90  # азимут
map_width = 5000 # довжина мапи
map_height = 5000 # висота мапи
key = "AtAGNqgREpRZd8E9X8m0kDZILfIRClZUYNYVbv1Lnj0RjvJuD0gJqakFvH7sfFkl"  # Зареєструватись на Bing Maps і отримати ключ

url = f"https://dev.virtualearth.net/REST/V1/Imagery/Map/Aerial/" \
      f"{lattitude},{longitude}" \
      f"/20?dir={direction}&ms={map_width},{map_height}&" \
      f"key={key}"
response = requests.get(url)

if response.status_code == 200:
    with open("Pictures 3x3/road.jpg", "wb") as file:
        file.write(response.content)
        print("Success.")
else:
    print("Bruh:", response.status_code)

## FEATURE EXTRACTION (Використання нейромереж)

abs_path = 'Pictures 3x3/parking.jpg'  #УВАГА!  Директорія може відрізнятись!

img = cv2.imread(abs_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = img.reshape(-1)
data_frame = pd.DataFrame()
data_frame['Original Pixel Values'] = img2

#Накладання фільтру Entropy
entropy_img = entropy(img, disk(1))
entropy1 = entropy_img.reshape(-1)
data_frame['Entropy'] = entropy1

#Накладання фільтру Gaussian через сігму 3
gauss_img = nd.gaussian_filter(img, sigma=3)
gauss1 = gauss_img.reshape(-1)
data_frame['Gaussian'] = gauss1

sobel_img = sobel(img)
sobel1 = sobel_img.reshape(-1)
data_frame['Sobel'] = sobel1

print(data_frame)

#Оригінальне відображення знімку
cv2.imshow('Original Image', img)

#Гаусівське відображення
cv2.imshow('Gaussian', gauss_img)

#Відображення через нейронку використовуючи фільтр Entropy
cv2.imshow('Entropy', entropy_img)

#Відображення знімку через sobel
cv2.imshow('Sobel Image', sobel_img)

cv2.waitKey()
cv2.destroyAllWindows()


## Write to CSV file (Імпорт та експорт даних файлу CSV)

# Read file CSV(Читання файлу CSV)
with open("Data.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=";")
    for row in reader:
        print(row['Name_picture'], '|', row['coordinate'])

# Write file CSV(Записування файлу CSV)
with open("../new_data.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    writer.writerow(["El 1", "El 2", "El 3"])
    writer.writerow(["El 2", "El 2", "El 3"])
    writer.writerow(["El 3", "El 2", "El 3"])