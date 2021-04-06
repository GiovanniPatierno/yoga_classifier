# Yoga Classifier

Il progetto è composto da 4 file:

- Main.py è il file che principale che esegue il programma;
- extract_feature contiene tutte le funzioni per estrarre le feature;
- utility.py contiene la funzione per tracciare una matrice di confusione, insieme alle funzioni per eseguire delle predizioni per i vari classificatori
- Cnn_function crea e restituisce la rete neurale a convoluzione

Nel repository ci sono anche i pesi della rete neurale già addestrata: "h.5.data-00000-of-00001" e "h.5.index"

La documentazione completa si trova nella cartella "Documentazione"

N.B Al momento dell'esecuzione il classificatore SVM potrebbe richiedere alcuni minuti

### Istruzioni

1. Scaricare il dataset  da qui: https://drive.google.com/drive/folders/1zarNtQtoqLA-YsLPIrK0vO43SYqjQsh0

2.  Inserirlo nella cartella dataset_file

3. Inserire il path assoluto del training e del test set, nella variabile TRAIN_DIR e TEST_DIR(riga 27 e 28)

4. Modificare il path assoluto per eseguire le predizioni
