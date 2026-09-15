# MeArm Forza 4

![Demo del Robot](/media/demo.gif)

## Introduzione
Il progetto consiste nello sviluppo di un simulatore fisico end-to-end di un braccio robotico (MeArm) in grado di giocare a Forza 4 contro un essere. Il sistema sfrutta il motore fisico PyBullet per una simulazione realistica di gravità, collisioni e cinematica dei motori, unendolo a un algoritmo Minimax per garantire al robot di scegliere la mossa migliore.

---

## Software
L'architettura è modulata per separare la simulazione fisica dall'interfaccia utente, prevenendo blocchi nel rendering:
*   **Interfaccia e Logica (`simulazione.py`)**: Genera la GUI con cui interagire, gestisce lo stato della partita, il controllo della vittoria e l'algoritmo Minimax per il calcolo della mossa ottimale del robot.
*   **Fisica e Ambiente (`mearm_real_env.py`)**: Utilizza un ambiente Gymnasium personalizzato per caricare il modello URDF del robot, gestire la fisica della griglia di Forza 4 e coordinare la cinematica complessa dei tiranti (mimic joints, per maggiori informazioni guarda repository MeArm-RL-PyBullet).
*   **Multiprocessing**: Sfrutta la libreria `multiprocessing` di Python per far comunicare asincronamente la GUI e l'ambiente PyBullet a 60fps tramite variabili condivise (`mp.Value`) come `shared_target_col` e `shared_user_col`.

---

## Calibrazione
Per garantire precisione millimetrica nell'inserimento dei gettoni senza incappare in bug della cinematica inversa, si usa una tecnica di calibrazione pre-calcolata.

### Teach & Play Elettromeccanico
Tramite lo script `calibra.py`, è stata creata una mappatura in cui per ognuna delle 7 colonne del tabellone è stata salvata una configurazione sicura di angoli (Base, Spalla, Gomito). 
*   **Ologrammi Guida**: Durante l'esecuzione di `calibra.py`, il sistema genera dei cubi olografici azzurri alle altezze di sgancio ideali. Tramite appositi slider, muovi i servomotori finché un blocco rosso sulla pinza del robot non si sovrappone perfettamente all'ologramma.

---

## Problem Solving

### Le collisioni con la Griglia
Il movimento simultaneo dei 3 assi (da punto a punto) causava inevitabilmente lo schianto del braccio contro la griglia centrale.
*   **Soluzione (Macchina a Stati e Clearance Plane):** Sviluppo di una macchina a stati per simulare il comportamento dei bracci industriali. Il movimento è suddiviso in fasi rigide a forma di "L" inversa: `LIFT_ARM` (il braccio si alza tenendo ferma la base), `ROTATE_TO_COL` (il robot ruota sorvolando in sicurezza la griglia) e `DROP` (rilascio del gettone), seguito dal ritorno `ROTATE_TO_HOME` prima di riabbassarsi (`LOWER_ARM`).

### Incastri e rimbalzi dei gettoni
L'utilizzo di blocchetti cubici portava a frequenti incastri nelle paratie, mentre le sfere standard tendevano a rimbalzare o vibrare all'infinito a causa del motore fisico.
*   **Soluzione (Ottimizzazione Sfere e Attrito):** I gettoni sono stati sostituiti con sferette perfette da 13mm. Al momento del rilascio, la dinamica della sfera viene sovrascritta tramite `p.changeDynamics`, impostando l'elasticità a zero (`restitution=0.0`) e aumentando l'attrito rotazionale (`rollingFriction=0.1`, `spinningFriction=0.1`). 

---

## Conclusione
Il progetto dimostra la perfetta sinergia tra sviluppo UI, calcolo fisico e logica algoritmica. Risolvendo le complessità della simulazione meccanica tramite macchine a stati rigide.

---

## Installazione e Avvio
Per eseguire il progetto, assicurati di installare le seguenti dipendenze:

```bash
pip install pybullet gymnasium customtkinter numpy