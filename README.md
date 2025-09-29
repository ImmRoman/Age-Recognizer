# Struttura del progetto

## 📌 Main
- Contiene i loop di training.
- Qui viene importato direttamente il modello `mobilenet_v3_large`.

## 🗄️ Dataset
- Implementa il **Database** e le funzioni affini.

## 🧠 Cnn
- Contiene il modello **AgeCNN** con cui sperimento custom cnn.
- Contiente la funzione per definire le classi di riconoscimenti.

## 📂 Test_on_folder
- Script per testare un modello su una **cartella di immagini**.
- Genera la **matrice di confusione** per valutare le prestazioni.

## 🖼️ Test_on_single_pictures
- Script per testare un modello su una **singola immagine**.
