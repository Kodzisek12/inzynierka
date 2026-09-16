# Rozpoznawanie czynności w filmie

Projekt klasyfikuje filmy do jednej ze 101 klas UCF101 za pomocą wytrenowanej sieci 3D CNN.

## Aplikacja lokalna

Aplikacja umożliwia przesłanie własnego pliku wideo lub zebranie próbki z kamery. Korzysta z modelu `best_model.keras` i z tego samego przygotowania danych co trening: 32 równomiernie wybrane klatki, rozmiar 32 × 32 px, RGB i wartości pikseli od 0 do 1.

W PowerShellu, w katalogu projektu, utwórz środowisko Pythona dla Windows i zainstaluj pakiety:

```powershell
py -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Jeżeli polecenie `py` nie jest dostępne, użyj `python` zamiast niego w pierwszej linii.

Po pojawieniu się komunikatu o serwerze otwórz [http://127.0.0.1:5000](http://127.0.0.1:5000). Wybierz plik AVI, MP4, MOV, MKV lub WebM albo kliknij **Włącz kamerę** i zaakceptuj zgodę przeglądarki. Kamera zbiera 32 klatki przez około 8 sekund, a następnie wyświetla pięć najbardziej prawdopodobnych klas.

## Skrypty projektu

- `preprocessing.py` — tworzy deterministyczny podział danych na trening, walidację i test, zachowując całe grupy źródłowych nagrań UCF101 w jednym zbiorze.
- `train.py` — trenuje model i zapisuje najlepsze wagi.
- `test.py` — mierzy jakość modelu na zbiorze testowym.
- `app.py` — uruchamia lokalną aplikację WWW.
