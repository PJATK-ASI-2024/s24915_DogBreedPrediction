# Projekt: Klasyfikacja Ras Psów na Podstawie Obrazów i Atrybutów Fizycznych

## Opis Tematu i Problemu Biznesowego/Technicznego

### Temat
Celem projektu jest stworzenie modelu uczenia maszynowego, który będzie rozpoznawał rasy psów na podstawie ich zdjęć oraz dodatkowych cech fizycznych. Tego typu model może znaleźć zastosowanie w aplikacjach mobilnych oraz serwisach online, wspierających właścicieli psów, schroniska oraz organizacje adopcyjne w rozpoznawaniu ras psów.

### Problem
W rzeczywistych scenariuszach, takich jak adopcje, identyfikacja rasy psa może być trudna bez specjalistycznej wiedzy. Właściwe określenie rasy jest jednak istotne, ponieważ różne rasy mają odmienne wymagania zdrowotne, żywieniowe, oraz potrzeby dotyczące aktywności. Projekt ma na celu automatyzację tego procesu, co pozwoli na szybką i dokładną klasyfikację rasy psów na podstawie zdjęcia oraz danych fizycznych, takich jak waga czy wzrost.

## Źródła Danych i Ich Charakterystyka

### Źródło Danych Obrazowych
Do części wizualnej projektu wykorzystamy **Stanford Dogs Dataset**, który zawiera ponad 6400 obrazów 40 ras psów. Zbiór ten jest otwarty i często stosowany w projektach związanych z rozpoznawaniem obrazów, co czyni go odpowiednim wyborem ze względu na wysoką jakość i szczegółowe etykiety.

Aby spełnić wymagania dotyczące metadanych z co najmniej pięcioma atrybutami numerycznymi, można wzbogacić powyższe zbiory o dodatkowe informacje, takie jak:

Wymiary obrazu: Szerokość i wysokość w pikselach.
Rozmiar pliku: W kilobajtach lub megabajtach.
Dominujące kolory: Średnie wartości RGB dla obrazu.
Jasność: Średnia jasność obrazu.
Kontrast: Wskaźnik kontrastu obrazu.

## Cele Projektu

1. **Stworzenie modelu klasyfikacyjnego**: Model, który rozpozna rasę psa na podstawie obrazu i atrybutów fizycznych.
2. **Budowa pipeline'u przetwarzania danych**: Zautomatyzowany proces przygotowania, przetwarzania i podziału danych na zbiór treningowy i testowy.
3. **Opracowanie dokumentacji użytkownika**: Przygotowanie instrukcji korzystania z modelu i omówienie jego ograniczeń.
4. **Weryfikacja i doszkalanie modelu**: Zbadanie skuteczności modelu oraz jego doskonalenie, aby zapewnić dokładność klasyfikacji.

