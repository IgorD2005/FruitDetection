#!/usr/bin/env python3
"""
LAB5: Konwolucyjne sieci neuronowe
Klasyfikacja owoców: banan, pomarańcza, cytryna
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# Stałe globalne
# ============================================================

IMG_SIZE = (128, 128)


# ============================================================
# TODO 1: Przygotowanie danych
# ============================================================

def wczytaj_dane(sciezka_danych):
    """
    Wczytuje dane treningowe i walidacyjne z podkatalogów przy użyciu
    ImageDataGenerator z parametrem validation_split.

    Struktura katalogów:
    dataset/
        banana/   (zdjęcia bananów)
        orange/   (zdjęcia pomarańczy)
        lemon/    (zdjęcia cytryn)

    Zastosowana normalizacja: rescale=1./255 (piksele [0,255] -> [0,1])
    Podział: 80% trening / 20% walidacja

    Dwa oddzielne generatory:
    - train_datagen: z augmentacją (transformacje losowe)
    - val_datagen:   tylko normalizacja — uczciwa ocena modelu

    Zwraca:
        train_gen - generator zbioru treningowego (z augmentacją)
        val_gen   - generator zbioru walidacyjnego (bez augmentacji)
    """

    train_datagen = przygotuj_augmentacje()
    train_datagen.rescale = 1. / 255
    train_datagen.validation_split = 0.2

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        sciezka_danych,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        sciezka_danych,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print(f"\n[INFO] Klasy i ich indeksy: {train_gen.class_indices}")
    print(f"[INFO] Zbiór treningowy:  {train_gen.samples} obrazów")
    print(f"[INFO] Zbiór walidacyjny: {val_gen.samples} obrazów")

    return train_gen, val_gen


# ============================================================
# TODO 2: Augmentacja danych
# ============================================================

def przygotuj_augmentacje():
    """
    Tworzy generator augmentacji danych.

    Zastosowane transformacje:
    - rotation_range=40 -> obrót do 40°
    - width/height_shift=20% -> przesunięcie w pionie i poziomie
    - shear_range=20% -> ścinanie (zmiana perspektywy)
    - zoom_range=30% -> przybliżenie/oddalenie
    - horizontal_flip=True -> odbicie lustrzane
    - brightness_range=[0.7,1.3] -> losowa zmiana jasności
    - fill_mode='nearest' -> uzupełnianie pikseli po transformacji

    Cel augmentacji:
    - zwiększenie różnorodności danych
    - poprawa uogólniania modelu
    - zmniejszenie overfittingu

    Zwraca:
    - ImageDataGenerator skonfigurowany do augmentacji
    """
    generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    print("[INFO] Generator augmentacji został utworzony")

    return generator


# ============================================================
# TODO 3: Budowa modelu CNN
# ============================================================

def zbuduj_model(input_shape, liczba_klas):
    """

    Buduje konwolucyjną sieć neuronową.

    Architektura:
    - Warstwy Conv2D (16, 32, 64, 64 filtrów) z aktywacją ReLU
    - Warstwy MaxPooling2D (2x2) po każdej konwolucji
    - Warstwa Flatten (spłaszczenie map cech)
    - Warstwa Dense (128 neuronów) -> Dropout(0.3)
    - Warstwa wyjściowa: Dense(liczba_klas) -> Softmax

    Uzasadnienie wyborów:
    - ReLU: Standardowa funkcja aktywacji dla warstw ukrytych.
    - Softmax: Przekształca wynik w prawdopodobieństwa (suma = 1).
    - Dropout(0.3): Zapobiega overfittingowi poprzez losowe wyłączanie 30% neuronów.
    - MaxPooling: Redukuje wymiary obrazu, zachowując najważniejsze cechy.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(liczba_klas, activation='softmax')
    ])
    return model


# ============================================================
# TODO 4: Kompilacja modelu
# ============================================================

def skompiluj_model(model):
    """
    Kompiluje model.

    Uzasadnienie wyborów:
    - categorical_crossentropy: Strata dla klasyfikacji wieloklasowej (one-hot).
    - Adam (lr=0.0001): Wydajny optymalizator z adaptacyjnym tempem uczenia.
    - accuracy: Metryka określająca odsetek poprawnych klasyfikacji.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# ============================================================
# TODO 5: Trenowanie modelu
# ============================================================

def trenuj_model(model, train_gen, val_gen):
    """
    Trenuje model z użyciem EarlyStopping i ModelCheckpoint.

    Parametry:
    - Epoki: max 20
    - Dane: train_gen (z augmentacją), val_gen (bez augmentacji)

    Techniki:
    - EarlyStopping: Zatrzymuje naukę po 5 epokach bez poprawy val_loss.
    - ModelCheckpoint: Zapisuje tylko najlepsze wagi (best_model.keras).

    Uzasadnienie:
    - 20 epok: wystarczające do zbieżności modelu
    - EarlyStopping: zapobiega przeuczeniu
    - patience=5: kompromis między stabilnością a szybkością
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            "best_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    historia = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    return historia


# ============================================================
# TODO 6: Klasyfikacja testowa
# ============================================================

def klasyfikuj_obraz(model, sciezka_obrazu, class_indices):
    """
    Klasyfikuje pojedynczy obraz.

    Parametry:
    - model: wytrenowany model CNN
    - sciezka_obrazu: ścieżka do obrazu testowego
    - class_indices: słownik mapujący nazwy klas na indeksy

    Funkcja:
    - wczytuje obraz
    - przeskalowuje i normalizuje dane
    - wykonuje predykcję
    - wyświetla wynik oraz prawdopodobieństwa klas
    """
    if not os.path.isfile(sciezka_obrazu):
        print(f"[BŁĄD] Plik nie istnieje: {sciezka_obrazu}")
        return

    img = cv2.imread(sciezka_obrazu)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    predictions = model.predict(img_input, verbose=0)[0]

    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_idx = int(np.argmax(predictions))
    predicted_class = idx_to_class[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    print("\n" + "=" * 40)
    print(f"WYNIK KLASYFIKACJI: {sciezka_obrazu}")
    print("=" * 40)
    print(f"Przewidywana klasa: {predicted_class.upper()}")
    print(f"Pewność: {confidence:.2f}%")
    print("\nPrawdopodobieństwa wszystkich klas:")
    for idx in sorted(idx_to_class.keys()):
        name = idx_to_class[idx]
        prob = predictions[idx]
        bar = "█" * int(prob * 30)
        print(f"  {name:<10}: {prob * 100:6.2f}%  {bar}")
    print("=" * 40)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title(f"Predykcja: {predicted_class} ({confidence:.1f}%)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("wynik_klasyfikacji.png", dpi=150)
    plt.show()
    print("[INFO] Wynik zapisany jako 'wynik_klasyfikacji.png'")

    return predicted_class, predictions


# ============================================================
# Wizualizacja historii treningu
# ============================================================

def rysuj_wykresy(historia):
    """
    Rysuje wykresy dokładności i straty dla zbioru treningowego i walidacyjnego.
    Zapisuje wykres do pliku 'historia_treningu.png'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(historia.history["accuracy"], label="Train accuracy")
    axes[0].plot(historia.history["val_accuracy"], label="Val accuracy")
    axes[0].set_title("Dokładność (Accuracy)")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(historia.history["loss"], label="Train loss")
    axes[1].plot(historia.history["val_loss"], label="Val loss")
    axes[1].set_title("Strata (Loss)")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("historia_treningu.png", dpi=150)
    plt.show()
    print("[INFO] Wykres zapisany jako 'historia_treningu.png'")


# ============================================================
# Funkcja główna
# ============================================================

def main():
    sciezka_danych = "dataset"

    input_shape = (128, 128, 3)
    liczba_klas = 3

    print("[KROK 1] Wczytywanie danych i przygotowanie generatorów...")
    train_gen, val_gen = wczytaj_dane(sciezka_danych)

    print("\n[KROK 2] Augmentacja danych...")
    przygotuj_augmentacje()

    print("\n[KROK 3] Budowa modelu CNN...")
    model = zbuduj_model(input_shape, liczba_klas)

    print("\n[KROK 4] Kompilacja modelu...")
    skompiluj_model(model)

    model.summary()

    print("\n[KROK 5] Trenowanie modelu...")
    historia = trenuj_model(model, train_gen, val_gen)

    print("\n[KROK 6] Zapisywanie modelu...")
    model.save("model_owoce.keras")
    print("[INFO] Model zapisany jako 'model_owoce.keras'")

    print("\n[KROK 7] Rysowanie wykresów...")
    rysuj_wykresy(historia)

    print("\n[KROK 8] Klasyfikacja testowa...")
    przykladowy_obraz = "test.jpg"
    if os.path.isfile(przykladowy_obraz):
        klasyfikuj_obraz(model, przykladowy_obraz, train_gen.class_indices)
    else:
        print(f"[INFO] Brak pliku '{przykladowy_obraz}'.")
        print("[INFO] Umieść zdjęcie owocu jako 'test.jpg' i uruchom ponownie.")
        print("[INFO] Możesz też wywołać:")
        print("       klasyfikuj_obraz(model, 'ścieżka/do/obrazu.jpg', train_gen.class_indices)")


if __name__ == "__main__":
    main()
