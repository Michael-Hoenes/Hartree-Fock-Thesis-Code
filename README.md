Code zur Bachelorarbeit – Hartree-Fock-SCF für ein eindimensionales System

Dieses Repository enthält den Python-Code zur Bachelorarbeit, in der eine selbstkonsistente Feldberechnung (SCF) im Rahmen der Hartree-Fock-Theorie für ein eindimensionales System mit wechselwirkenden Fermionen durchgeführt wird.

## 🔍 Inhalt

- `run_fixed_spin_scf(...)`: Führt eine SCF-Berechnung mit fester Spinkonfiguration durch.
- `calc_result(...)`: Bestimmt die energetisch günstigste Spinkonfiguration für gegebene Parameter.
- `plot_energy_vs_spin_config(...)`: Plottet die Gesamtenergie gegen die relative Spinpolarisation.
- `compute_F(...)`: Berechnet Hartree- und Austauschmatrix.
- `random_orthogonal_matrix(...)`: Generiert eine zufällige orthogonale Matrix (für evtl. Tests).
- Plot- und Darstellungscode mit LaTeX-Schrift für wissenschaftliche Darstellung.

## ⚙️ Abhängigkeiten

Folgende Python-Pakete werden verwendet:
- `numpy`
- `matplotlib`
- `scipy`
- `numba`

Installierbar z. B. via:

```bash
pip install numpy matplotlib scipy numba
