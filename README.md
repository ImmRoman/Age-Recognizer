# Struttura del progetto

##  Main
- Contiene i loop di training per l'Age Detection
- Nel branch Gender-Detection il main si allena per riconoscere il genere.
## Dataset
- Implementa il **Database** per estrarre dal database UTK faces le foto e le loro label. 

##  Cnn
- Contiene il modello **AgeCNN** con cui sperimento custom cnn.
- Contiente la funzione per definire le classi di riconoscimenti.

## FinalScript
- Esegue in pipeline MediaPipe e le 2 CNN selezionate per il riconoscimento di età e genere

## Models
- Contiene i modelli e le matrici di confusione presentati nella tesina

## Test_on_folder
- Script per testare un modello su una **cartella di immagini**.
- Genera la **matrice di confusione** per valutare le prestazioni.

## Test_on_single_pictures
- Script per testare un modello su una **singola immagine**.
