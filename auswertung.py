import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ==============================================================================
# 1. DATEN IN DIE ENTSPRECHENDE STRUKTUR LADEN
# ==============================================================================
CSV_FILE = "masterarbeit_raw_data.csv"

try:
    df = pd.read_csv(CSV_FILE, sep=";")
    print(f"📊 Datensatz erfolgreich geladen. Gesamte Zeilenanzahl: {len(df)}")
except Exception as e:
    print(f"🚨 Fehler beim Laden der Datei: {e}")
    exit()

# Sicherstellen, dass die Ziel-Deltas als numerische Werte vorliegen
df["delta_IVS"] = pd.to_numeric(df["delta_IVS"], errors='coerce').fillna(0)
df["delta_EG"] = pd.to_numeric(df["delta_EG"], errors='coerce').fillna(0)

# ==============================================================================
# 2. EXPERIMENTELLE HYPOTHESEN-KONFIGURATION (NACH KAPITEL 3.6.1 & 3.6.2)
# ==============================================================================
modulations_annahmen = {
    "M1": {
        "title": "Erste Modulationsannahme (Willensbahnung / Handlungsbahnung)",
        "scenario_filter": "XM1",
        "target_delta": "delta_IVS",
        "h1_text": "H1,M1: (Delta %IVS) B_PSI > (Delta %IVS) B_K",
        "h0_text": "H0,M1: (Delta %IVS) B_PSI <= (Delta %IVS) B_K"
    },
    "M2": {
        "title": "Zweite Modulationsannahme (Selbstentwicklung durch Affektregulation)",
        "scenario_filter": "XM2",
        "target_delta": "delta_EG",
        "h1_text": "H1,M2: (Delta %EG) B_PSI > (Delta %EG) B_K",
        "h0_text": "H0,M2: (Delta %EG) B_PSI <= (Delta %EG) B_K"
    }
}

print("\n" + "="*80)
# Fettgedruckter Kerntitel für Scannbarkeit im Auswertungsprotokoll
print("**STATISTISCHE EVALUATION: GERICHTETE SYSTEMVERSCHIEBUNG (DIRECTIONAL SHIFT)**")
print("="*80)

# ==============================================================================
# 3. STATISTISCHE HYPOTHESENPRÜFUNG & EFFEKTMESSUNG (N = 100)
# ==============================================================================
for m_id, m_info in modulations_annahmen.items():
    print(f"\n▶️ ANALYSE: {m_info['title']}")
    print(f"   Prüfe H1 : {m_info['h1_text']}")
    print(f"   Gegen H0 : {m_info['h0_text']}")
    print("-" * 70)
    
    # 3.1 Gepaarte Subgruppen-Filterung nach Iteration sortiert
    df_psi = df[(df["Szenario"] == m_info["scenario_filter"]) & (df["Bedingung"] == "B_PSI")].sort_values(by="Iteration")
    df_k   = df[(df["Szenario"] == m_info["scenario_filter"]) & (df["Bedingung"] == "B_K")].sort_values(by="Iteration")
    
    N_psi = len(df_psi)
    N_k   = len(df_k)
    
    if N_psi == 0 or N_k == 0:
        print(f"   ❌ FEHLER: Keine Daten für Szenario {m_info['scenario_filter']} gefunden. Überspringe Achse.")
        continue

    # Erzwinge exakte Paar-Symmetrie (N=100)
    if N_psi != N_k:
        print(f"   ⚠️ WARNUNG: Ungleiche Stichprobengrößen (PSI: {N_psi}, K: {N_k}). Kürze auf das Minimum.")
        min_n = min(N_psi, N_k)
        df_psi = df_psi.head(min_n)
        df_k = df_k.head(min_n)
        N_psi = min_n

    print(f"   Stichprobenarchitektur: N = {N_psi} verbundene Paare (gemäß Spezifikation)")
    
    # 3.2 Extraktion der Differenzenmaße (Verschiebungskoeffizienten)
    y_psi = df_psi[m_info["target_delta"]].values
    y_k   = df_k[m_info["target_delta"]].values
    
    # Deskriptive Kennzahlen
    mean_psi = np.mean(y_psi)
    mean_k   = np.mean(y_k)
    print(f"   Deskriptive Mittelwerte der gerichteten Verschiebung (Zt1 - Zt0):")
    print(f"     - Experimentalgruppe (B_PSI): {mean_psi:+.2f}%")
    print(f"     - Kontrollgruppe (B_K):      {mean_k:+.2f}%")
    
    # 3.3 Einseitiger Wilcoxon-Vorzeichen-Rang-Test
    # alternative='greater' prüft gerichtet, ob die Verschiebung bei B_PSI signifikant größer ist als bei B_K
    try:
        # Z-Statistik wird über das Argument erzwungen, um r fehlerfrei nach Formel 3.13 zu berechnen
        res = wilcoxon(y_psi, y_k, alternative='greater', method='approx')
        stat_w = res.statistic
        p_val = res.pvalue
        
        # 3.4 Mathematische Ermittlung von Z_stat zur Berechnung von r (Formel 3.13)
        # Bei N=100 greift die asymptotische Normalverteilung unter H0:
        expected_w = (N_psi * (N_psi + 1)) / 4
        se_w = np.sqrt((N_psi * (N_psi + 1) * (2 * N_psi + 1)) / 24)
        z_stat = (stat_w - expected_w) / se_w
        
        # Standardisierte Effektstärke r (Formel 3.13)
        r_effect = abs(z_stat) / np.sqrt(N_psi)
        
        # Klassifikation nach Cohen (1988)
        if r_effect >= 0.5:
            cohen_label = "Starker experimenteller Steuerungseffekt"
        elif r_effect >= 0.3:
            cohen_label = "Mittlerer experimenteller Steuerungseffekt"
        elif r_effect >= 0.1:
            cohen_label = "Schwacher experimenteller Steuerungseffekt"
        else:
            cohen_label = "Kein nennenswerter Effekt"
            
        # 3.5 Analytischer Ergebnisreport für Kapitel 4.2
        print(f"\n   Statistische Testkennzahlen (alpha = 0.05, einseitig):")
        print(f"     - Teststatistik (W):       {stat_w:.1f}")
        print(f"     - Standardisierte Z_stat:  {z_stat:+.3f}")
        print(f"     - p-Wert:                  {p_val:.5f}")
        print(f"     - Effektstärke (r):        {r_effect:.3f} ({cohen_label})")
        
        # Signifikanzentscheidung
        if p_val < 0.05:
            print(f"   🎉 **SIGNIFIKANZ: Die Nullhypothese wird ABGELEHNT.**")
            print(f"      Das PSI-Modell bewirkt eine signifikant stärkere Aktivierung des Zielsystems.")
        else:
            print(f"   🛑 **ERGEBNIS: Die Nullhypothese wird BEIBEHALTEN.**")
            print(f"      Es konnte kein statistisch signifikanter Unterschied nachgewiesen werden.")
            
    except Exception as e:
        print(f"   ❌ Fehler bei der nicht-parametrischen Berechnung: {e}")

print("\n" + "="*80)
print("Auswertung vollständig beendet. Die Werte können direkt in die Ergebnistabellen überführt werden.")
print("="*80)
