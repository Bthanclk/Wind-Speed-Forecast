import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_predict(data, user_input):
    # Tarih sütununu datetime formatına çevirme, hatalı tarihleri atla
    data['tarih'] = pd.to_datetime(data['tarih'], format='%d-%m-%Y %H:%M', errors='coerce')

    # Hatalı tarihleri filtrele
    data = data.dropna(subset=['tarih'])

    # Tahmin yapmak için kullanılacak tarih aralığını belirle
    start_tarih = datetime(2023, 8, 4)
    end_tarih = datetime(2023, 8, 7)

    # Belirtilen tarih aralığındaki önceki verileri seç
    previous_data = data[data['tarih'] < start_tarih]

    # Belirtilen tarih aralığındaki tahmin verilerini seç
    prediction_data = data[(data['tarih'] >= start_tarih) & (data['tarih'] <= end_tarih)]

    # Tahmin yapmak için kullanılacak özellikleri seç
    features = ['nem', 'basinc', 'sicaklik']

    # Modeli oluştur
    model = LinearRegression()

    # Her bir özellik için ayrı grafik oluştur
    for feature in features:
        # Modeli eğit
        model.fit(previous_data[[feature]], previous_data['ruzgar_hizi'])

        # Tahmin yap
        predictions = model.predict(prediction_data[[feature]])

        # Tahmin sonuçlarını veri setine ekle
        prediction_data[f'predicted_wind_speed_{feature}'] = predictions

        # Grafik oluştur
        plt.figure(figsize=(12, 6))
        plt.plot(previous_data['tarih'], previous_data['ruzgar_hizi'], marker='o', linestyle='-', color='b', label='Önceki Rüzgar Hızları')
        plt.plot(prediction_data['tarih'], prediction_data[f'predicted_wind_speed_{feature}'], marker='o', linestyle='-', label=f'Tahmin Edilen Rüzgar Hızı - {feature}', color='red')
        plt.title(f'Rüzgar Hızı Tahminleri {feature}')
        plt.xlabel('Tarih')
        plt.ylabel('Rüzgar Hızı (m/s)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Üç özellikle birlikte hesaplanan grafik
    plt.figure(figsize=(12, 6))
    plt.plot(previous_data['tarih'], previous_data['ruzgar_hizi'], marker='o', linestyle='-', color='b', label='Önceki Rüzgar Hızları')
    plt.plot(prediction_data['tarih'], prediction_data['predicted_wind_speed_nem'], marker='o', linestyle='-', label='Tahmin Edilen Rüzgar Hızı - Nem')
    plt.plot(prediction_data['tarih'], prediction_data['predicted_wind_speed_basinc'], marker='o', linestyle='-', label='Tahmin Edilen Rüzgar Hızı - Basınç')
    plt.plot(prediction_data['tarih'], prediction_data['predicted_wind_speed_sicaklik'], marker='o', linestyle='-', label='Tahmin Edilen Rüzgar Hızı - Sıcaklık')
    plt.title('Rüzgar Hızı Tahminleri - Tüm Özellikler')
    plt.xlabel('Tarih')
    plt.ylabel('Rüzgar Hızı (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(previous_data['tarih'], previous_data['ruzgar_hizi'], marker='o', linestyle='-', color='b', label='Önceki Rüzgar Hızları')
    plt.plot(prediction_data['tarih'], (prediction_data['predicted_wind_speed_nem'] + prediction_data['predicted_wind_speed_basinc'] + prediction_data['predicted_wind_speed_sicaklik']) / 3, marker='o', linestyle='-', color='r', label='Tahmin Edilen Rüzgar Hızı')
    plt.title('Rüzgar Hızı Tahminleri - Ortalama')
    plt.xlabel('Tarih')
    plt.ylabel('Rüzgar Hızı (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Kullanıcı girişi
    humidity_input = float(input("Nem oranını giriniz: "))
    pressure_input = float(input("Basınç değerini giriniz: "))
    temp_input = float(input("Sıcaklık değerini giriniz: "))

    user_input = [humidity_input, pressure_input, temp_input]

    # Gelecek Gün Hesaplanması
    X = data[['nem', 'basinc', 'sicaklik']]
    y = data['ruzgar_hizi']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli oluştur ve eğit

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Tahmin yap

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Kullanıcı girişi
    user_pred = model.predict([user_input])
    print(f"Tahmini Rüzgar Hızı: {user_pred[0]}")

# Veri setini yükle
wind_data = pd.read_csv('wind_data.csv')

# Fonksiyonu çağır
train_and_predict(wind_data, [])
