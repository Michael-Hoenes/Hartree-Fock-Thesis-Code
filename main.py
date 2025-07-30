import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as scl
import numba 

#Dieser Code wurde von Gemini 2.5 Pro auskommentiert. Der Code selbst wurde vom Autor Michael Hönes geschrieben.


# --- Matplotlib Konfiguration für LaTeX-Plots ---
# Diese Einstellungen sorgen für eine hochwertige grafische Darstellung,
# die für wissenschaftliche Publikationen und Abschlussarbeiten geeignet ist.
# 'text.usetex = True' aktiviert die LaTeX-Engine für alle Textelemente im Plot.
mpl.rcParams['text.usetex'] = True
# 'text.latex.preamble' ermöglicht das Laden von LaTeX-Paketen,
# hier für mathematische Symbole und Schriftarten.
mpl.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{physics}
'''
# Setzt die Schriftfamilie auf eine Serifenschrift, passend zu den meisten LaTeX-Dokumenten.
mpl.rcParams['font.family'] = 'serif'

#--------------------------- Funktionen ----------------------------

@numba.jit(nopython=True)
def compute_F(D_up, D_down, w0):
    """
    Berechnet die Interaktions-Anteile der Fock-Matrix im k-Raum.
    Diese Funktion ist der rechenintensivste Teil der SCF-Schleife und wird
    daher mit Numba JIT (Just-In-Time) kompiliert, um die Ausführung erheblich 
    zu beschleunigen.

    Args:
        D_up (np.array): Die Dichtematrix der Spin-Up-Elektronen im k-Raum.
                         D[k, k'] = <c_k^dagger c_k'>
        D_down (np.array): Die Dichtematrix der Spin-Down-Elektronen im k-Raum.
        w0 (float): Die Stärke der Wechselwirkung.

    Returns:
        J (np.array): Die Coulomb-Matrix. Repräsentiert die klassische elektrostatische
                      Abstoßung zwischen Elektronen. Wirkt auf alle Elektronen unabhängig vom Spin.
        K_up (np.array): Die Austausch-Matrix für Spin-Up-Elektronen. Ein rein
                         quantenmechanischer Effekt (Pauli-Prinzip), der nur zwischen
                         Elektronen mit gleichem Spin wirkt.
        K_down (np.array): Die Austausch-Matrix für Spin-Down-Elektronen.
    """
    # M wird aus der Dimension der Dichtematrix abgeleitet. M ist die Anzahl 
    # der diskreten k-Punkte in der Brillouinzone.
    M = D_up.shape[0]
    
    # Initialisierung der Matrizen mit Nullen.
    J = np.zeros((M,M))      # Coulomb-Matrix
    K_up = np.zeros((M,M))   # Austausch-Matrix für Spin-Up
    K_down = np.zeros((M,M)) # Austausch-Matrix für Spin-Down

    # Die Schleifen iterieren über alle k-Punkte, um die Matrixelemente zu berechnen.
    # Dies repräsentiert die Streuung von zwei Elektronen.
    for k1 in range(M):
        for k2 in range(M):
            for k3 in range(M):
                # Impulserhaltung im Gitter: k1 + k3 = k2 + k4 (mod G), wobei G ein
                # reziproker Gittervektor ist. Im diskreten Fall wird dies zu:
                k4 = (k1 + k3 - k2) % M  # Das Modulo M berücksichtigt Umklapp-Prozesse.
                
                # Berechnung des Coulomb-Matrixelements J[k1, k2].
                # Es hängt von der Gesamt-Elektronendichte (up + down) ab.
                # Der Vorfaktor w0/(2*pi^2) ist modellspezifisch.
                J[k1, k2] += np.divide(w0, 2*np.pi**2) * (D_up[k3, k4] + D_down[k3, k4])
                
                # Berechnung des Austausch-Matrixelements K_up[k1, k2].
                # Es hängt nur von der Dichte der Spin-Up-Elektronen ab.
                K_up[k1, k2] +=  np.divide(w0, 2*np.pi**2) * D_up[k3, k4]
                
                # Berechnung des Austausch-Matrixelements K_down[k1, k2].
                # Es hängt nur von der Dichte der Spin-Down-Elektronen ab.
                K_down[k1, k2] +=  np.divide(w0, 2*np.pi**2) * D_down[k3, k4]
                
    return J, K_up, K_down


def run_fixed_spin_scf(N_up_start, N_down_start, w0, M, t, max_iter, threshold, mix):
    """
    Führt eine Self-Consistent Field (SCF) Rechnung für eine feste Anzahl von 
    Spin-Up- und Spin-Down-Elektronen durch.
    
    Args:
        N_up_start (int): Anzahl der Spin-Up-Elektronen.
        N_down_start (int): Anzahl der Spin-Down-Elektronen.
        w0 (float): Wechselwirkungsstärke.
        M (int): Anzahl der k-Punkte (Gitterplätze).
        t (float): Hopping-Parameter ( kinetische Energie).
        max_iter (int): Maximale Anzahl von SCF-Iterationen.
        threshold (float): Konvergenzschwelle für die Dichtematrix.
        mix (float): Mischungsparameter (0 <= mix < 1) zur Stabilisierung der Konvergenz.
                     mix=0 bedeutet keine Mischung.

    Returns:
        dict: Ein Dictionary mit den Ergebnissen der Rechnung (Konvergenzstatus,
              Gesamtenergie, Eigenenergien, etc.).
    """
    # 1. System-Setup (nicht-wechselwirkender Teil)
    k_vektor = (2 * np.pi / M) * np.arange(1, M + 1) - np.pi # Definiert die k-Punkte in der 1. BZ.
    Bloch_Energien = -t * np.cos(k_vektor) # Kinetische Energie für ein 1D-Tight-Binding-Modell.
    FE = np.diag(Bloch_Energien) # Der kinetische Teil des Hamiltonoperators (Fock-Matrix) ist diagonal im k-Raum.
    
    # 2. Initialer "Guess" für die Dichtematrix
    # Wir starten mit der Lösung des nicht-wechselwirkenden Systems (w0=0).
    energies_0, C_0 = scl.eigh(FE) # Löse das Eigenwertproblem für FE. C_0 sind die Eigenvektoren.
    idx_0 = np.argsort(energies_0) # Sortiere die Zustände nach ihrer Energie.
    
    # "Aufbau-Prinzip": Fülle die energetisch niedrigsten Zustände.
    # Die Spalten von C_0 sind die Eigenzustände (Orbitale). Wir wählen die N_up/N_down niedrigsten.
    C_up = C_0[:, idx_0][:, :N_up_start]
    C_down = C_0[:, idx_0][:, :N_down_start]
    
    # Konstruiere die initialen Dichtematrizen aus den besetzten Orbitalen.
    D_up = C_up @ C_up.conj().T
    D_down = C_down @ C_down.conj().T

    # 3. SCF-Iterationsschleife
    for i in range(max_iter):
        # Speichere die Dichtematrizen des vorherigen Schritts, um Konvergenz zu prüfen.
        D_old_up = np.copy(D_up)
        D_old_down = np.copy(D_down)
        
        # Berechne die Interaktionsterme J und K basierend auf der aktuellen Dichte.
        J, K_up, K_down = compute_F(D_up, D_down, w0)
        
        # Baue die neuen Fock-Matrizen für Spin-Up und Spin-Down.
        # F = T + V_eff = T + (J - K)
        # T: Kinetische Energie (FE)
        # J: Coulomb-Abstoßung (mittleres Feld aller Elektronen)
        # K: Austausch-Wechselwirkung (Pauli-Prinzip)
        F_up = FE + J - K_up
        F_down = FE + J - K_down    

        # Löse das Eigenwertproblem für die neuen Fock-Matrizen.
        # Die Eigenwerte sind die neuen Quasi-Teilchen-Energien.
        # Die Eigenvektoren (C_all_up/down) sind die neuen (Hartree-Fock-)Orbitale.
        energies_up, C_all_up = scl.eigh(F_up)
        energies_down, C_all_down = scl.eigh(F_down)

        # Sortiere die neuen Orbitale nach Energie.
        idx_up = np.argsort(energies_up)
        sorted_energies_up = energies_up[idx_up]
        
        idx_down = np.argsort(energies_down)
        sorted_energies_down = energies_down[idx_down]
        
        # "Aufbau-Prinzip": Besetze die N_up/N_down niedrigsten neuen Orbitale.
        C_up = C_all_up[:, idx_up][:, :N_up_start]
        C_down = C_all_down[:, idx_down][:, :N_down_start]

        # Konstruiere die neuen Dichtematrizen aus den neuen besetzten Orbitalen.
        # Eine lineare Mischung mit der alten Dichte kann Oszillationen dämpfen und
        # die Konvergenz verbessern (simple Form von "DIIS").
        D_up_new = C_up @ C_up.conj().T
        D_down_new = C_down @ C_down.conj().T
        D_up = mix * D_old_up + (1 - mix) * D_up_new
        D_down = mix * D_old_down + (1 - mix) * D_down_new
        
        # 4. Konvergenzprüfung
        # Prüfe, wie stark sich die Dichtematrix im Vergleich zum letzten Schritt geändert hat.
        diff = np.linalg.norm(D_up - D_old_up) + np.linalg.norm(D_down - D_old_down)

        if diff < threshold: 
            # Wenn die Änderung klein genug ist, ist die Lösung selbstkonsistent.
            
            # Berechnung der Hartree-Fock-Gesamtenergie.
            # E_total = Summe(besetzte Orbitalenergien) - 0.5 * <Wechselwirkungs-Energie>
            # Der zweite Term korrigiert die Doppelzählung der Elektron-Elektron-Wechselwirkung,
            # die in der Summe der Orbitalenergien enthalten ist.
            total_energy = np.sum(sorted_energies_up[:N_up_start]) + np.sum(sorted_energies_down[:N_down_start]) \
                         - 0.5 * (np.trace((J - K_up) @ D_up) + np.trace((J - K_down) @ D_down))
           
            print(f"für (N_up={N_up_start}, N_down={N_down_start}) mit w0={w0:.4f}: Konvergenz nach {i+1} Iterationen. E_total = {total_energy:.6f}")
            
            # Rückgabe eines Dictionaries mit allen relevanten Ergebnissen.
            # Dies ist eine gute Praxis, um Ergebnisse übersichtlich zu speichern.
            return {
                "converged": True,
                "total_energy": total_energy,
                "energies_up": sorted_energies_up,
                "energies_down": sorted_energies_down,
                "N_up": N_up_start,
                "N_down": N_down_start,
                "w0": w0,
                "M": M
            }
    
    # Falls die Schleife ohne Konvergenz durchläuft:
    print(f"für (N_up={N_up_start}, N_down={N_down_start}) mit w0={w0:.3f}: Maximale Anzahl an Iterationen erreicht, keine Konvergenz.")
    return {"converged": False, "total_energy": np.inf}


def calc_result(Ne, M, t, max_iter, weights, mix):
    """
    Orchestriert die Berechnungen, um den Grundzustand für verschiedene Wechselwirkungsstärken 
    zu finden. Findet für jedes w0 die Spinkonfiguration (N_up, N_down) mit der minimalen Energie.
    
    Args:
        Ne (int): Gesamtzahl der Elektronen.
        M (int): Anzahl der k-Punkte.
        ... (andere Parameter für run_fixed_spin_scf)
    """
    # Bestimme alle physikalisch möglichen Spinkonfigurationen (N_up, N_down).
    configs_to_test = []
    # Das Pauli-Prinzip begrenzt die Anzahl der Elektronen pro Spin: N_up <= M und N_down <= M.
    lower_bound_n_up = max(0, Ne - M)
    upper_bound_n_up = min(M, Ne)

    # Aus Symmetriegründen ist E(N_up, N_down) = E(N_down, N_up).
    # Wir müssen also nur Konfigurationen testen, bei denen z.B. N_up <= N_down ist.
    # Die Schleife läuft hier bis zur Hälfte, um diese Redundanz auszunutzen.
    for n_up in range(lower_bound_n_up, (Ne // 2) + 1):
        n_down = Ne - n_up
        if n_down <= M: # Prüfe, ob die Konfiguration gültig ist
             configs_to_test.append((n_up, n_down))
    
    # Sicherstellen, dass die Konfigurationen in der richtigen Reihenfolge sind.
    configs_to_test.sort(key=lambda config: abs(config[0] - config[1]))
    print(f"Zu testende Konfigurationen (N_up, N_down) für Ne={Ne}, M={M}: {configs_to_test}")

    # Speichert die besten Ergebnisse für jedes w0.
    best_results = []

    # Iteriere über alle zu testenden Wechselwirkungsstärken.
    for w in weights: 
        min_energy_for_this_w = float('inf')
        best_config_for_this_w = None
        
        # Teste für das aktuelle w alle möglichen Spinkonfigurationen.
        for config in configs_to_test:
            # Führe SCF für diese spezifische Konfiguration durch.
            scf_results = run_fixed_spin_scf(config[0], config[1], w, M, t, max_iter, threshold, mix)
            
            # Wenn die Rechnung konvergiert ist und die Energie niedriger als die bisher
            # gefundene minimale Energie ist, speichere sie als neuen Grundzustand.
            if scf_results["converged"] and scf_results["total_energy"] < min_energy_for_this_w:
                min_energy_for_this_w = scf_results["total_energy"]
                best_config_for_this_w = config

        # Speichere das beste Ergebnis für das aktuelle w0.
        if best_config_for_this_w is not None:
            # Da wir nur N_up <= N_down getestet haben, müssen wir die gespiegelte Konfiguration
            # mit der gleichen Energie berücksichtigen. Da die Energie dieselbe ist, können wir direkt
            # die gefundene Konfiguration verwenden.
            best_results.append((w, min_energy_for_this_w, best_config_for_this_w))
        else:
            print(f"WARNUNG: Für w0 = {w:.3f} ist keine Konfiguration konvergiert.")

    # Nach Abschluss der Schleifen, plotte die Ergebnisse.
    if best_results:
        w_values = [res[0] for res in best_results]
        # Berechne die relative Spinpolarisation (Magnetisierung) des Grundzustandes.
        # delta = (N_up - N_down) / N_e
        magnetization = [(res[2][1] - res[2][0]) / Ne for res in best_results]

        # Plotte die Magnetisierung als Funktion der Wechselwirkungsstärke.
        # Ein plötzlicher Anstieg von 0 auf einen endlichen Wert deutet auf einen
        # Phasenübergang hin (z.B. Stoner-Instabilität).
        plt.plot(w_values, magnetization, 'o-', label=fr'$N_\mathrm{{e}}=$ {Ne}, $M$= {M}', ms=5)
        plt.xlabel(r'Wechselwirkungsstärke $\omega_0$', fontsize=35)
        plt.ylabel(r'Relative Spinpolarisation $\delta$', fontsize=35)


def plot_energy_vs_spin_config(w0_target, Ne, M, t, max_iter, threshold, mix):
    """
    Eine diagnostische Funktion: Berechnet und plottet die Gesamtenergie für *alle*
    möglichen Spinkonfigurationen bei einer *festen* Wechselwirkungsstärke w0.
    Dies visualisiert, warum eine bestimmte Konfiguration der energetische Grundzustand ist.
    """
    print(f"\nBerechne Energielandschaft für w0 = {w0_target}, Ne = {Ne}, M = {M}")
    
    # Erstelle eine Liste aller physikalisch möglichen (N_up, N_down) Konfigurationen.
    configs_to_test = []
    lower_bound_n_up = max(0, Ne - M)
    upper_bound_n_up = min(M, Ne)
    for n_up in range(lower_bound_n_up, upper_bound_n_up + 1):
        n_down = Ne - n_up
        configs_to_test.append((n_up, n_down))
        
    if not configs_to_test:
        print("Keine erlaubten Konfigurationen.")
        return

    # Listen zum Speichern der Ergebnisse für den Plot.
    spins = []
    energies = []
    
    # Iteriere über alle Konfigurationen und berechne die jeweilige Energie.
    for config in configs_to_test:
        n_up, n_down = config
        scf_result = run_fixed_spin_scf(n_up, n_down, w0_target, M, t, max_iter, threshold, mix)
        
        if scf_result["converged"]:
            # Relative Spinpolarisation delta = (N_up - N_down) / N_e
            spins.append((n_up - n_down) / Ne)
            energies.append(scf_result["total_energy"])
            
    if not energies:
        print("Keine der Konfigurationen ist konvergiert.")
        return

    # Sortiere die Ergebnisse für eine saubere Darstellung.
    energies = np.array(energies)
    spins = np.array(spins)
    sort_indices = np.argsort(spins)
    spins_sorted = spins[sort_indices]
    energies_sorted = energies[sort_indices]

    # Erstelle den Plot.
    plt.plot(spins_sorted, energies_sorted, 'o--', label=fr'Energien für $\omega_0={w0_target}$')
    # Hebe die Grundzustandsenergie hervor.
    min_energy = np.min(energies)
    plt.hlines(min_energy, np.min(spins), np.max(spins), color="red", lw=1, ls="--", label=r"Grundzustandsenergie")
    plt.xlabel(r'Relative Spinpolarisation $\delta$', fontsize=21)
    plt.ylabel(r'Gesamtenergie in relativen Einheiten', fontsize=21)

# HINWEIS: Diese Funktion wird im aktuellen Skript nicht verwendet.
# Sie könnte nützlich sein, um z.B. einen zufälligen Startpunkt für die
# SCF-Iteration zu erzeugen, anstatt mit der nicht-wechselwirkenden Lösung zu beginnen.
def random_orthogonal_matrix(n):
    """
    Erzeugt eine zufällige n x n orthogonale Matrix mittels QR-Zerlegung.
    """
    H = np.random.randn(n, n) # Zufällige Matrix
    Q, R = np.linalg.qr(H)    # QR-Zerlegung
    return Q


#--------------------------- Globale Parameter  ----------------------------
Ne_sets = [6]       # Liste der zu untersuchenden Elektronenzahlen
M = 10              # Anzahl der k-Punkte / Gitterplätze
t = 1.0             # Hopping-Parameter (setzt die Energieskala)
max_iter = 100      # Maximale SCF-Iterationen
threshold = 1e-6    # Konvergenzschwelle
# Array der Wechselwirkungsstärken, die gescannt werden sollen.
weights = np.linspace(0.018, 0.021, 200) 
# Mischungsparameter für die Dichtematrix. 0 bedeutet keine Mischung.
# Ein Wert > 0 (z.B. 0.5) kann bei Konvergenzproblemen helfen.
mix = 0.0

plt.figure(figsize=(12, 8)) # Erstellt die Plot-Leinwand

#--------------------------- Main - Ausführung des Skripts -------------------------------
# Hier wird gesteuert, welche Funktion ausgeführt wird.
# Je nach Funktionswunsch muss die andere Funktion auskommentiert werden.

# Um die Magnetisierung als Funktion von w0 zu plotten (Phasendiagramm):
for Ne in Ne_sets:
    calc_result(Ne, M, t, max_iter, weights, mix)

# Um die Energielandschaft für eine feste Wechselwirkungsstärke zu untersuchen:
plot_energy_vs_spin_config(w0_target=5, Ne=Ne_sets[0], M=M, t=t, max_iter=max_iter, threshold=threshold, mix=mix)

#--------------------------- Plot-Finalisierung  ----------------------
plt.grid(True)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=20)

# Speichert die erzeugte Grafik als PDF-Datei.
# bbox_inches='tight' schneidet den leeren Rand um die Grafik ab.
#plt.savefig("Energy_vs_SpinPolarization.pdf", format="pdf", bbox_inches="tight", pad_inches=0.1)
plt.show() # Zeigt die Grafik an.