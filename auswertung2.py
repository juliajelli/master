import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# seaborn-Stil für wissenschaftliche Publikationen setzen
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

# ==============================================================================
# 1. DATEN IN DIE ENTSPRECHENDE STRUKTUR LADEN
# ==============================================================================
CSV_FILE = "masterarbeit_raw_data_typed.csv"

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
        "title": "Erste Modulationsannahme (Willensbahnung: IG -> IVS)",
        "scenario_filter": "XM1",
        "target_delta": "delta_IVS",
        "axis_label": "Gerichtete Verschiebung $\Delta$ %IVS",
        "file_boxplot": "plot_m1_willensbahnung_boxplot.png",
        "file_vector": "plot_m1_willensbahnung_vektor.png",
        "source_sys": "IG",
        "target_sys": "IVS",
        "h1_text": "H1,M1: (Delta %IVS) B_PSI > (Delta %IVS) B_K",
        "h0_text": "H0,M1: (Delta %IVS) B_PSI <= (Delta %IVS) B_K"
    },
    "M2": {
        "title": "Zweite Modulationsannahme (Selbstentwicklung: OES -> EG)",
        "scenario_filter": "XM2",
        "target_delta": "delta_EG",
        "axis_label": "Gerichtete Verschiebung $\Delta$ %EG",
        "file_boxplot": "plot_m2_selbstentwicklung_boxplot.png",
        "file_vector": "plot_m2_selbstentwicklung_vektor.png",
        "source_sys": "OES",
        "target_sys": "EG",
        "h1_text": "H1,M2: (Delta %EG) B_PSI > (Delta %EG) B_K",
        "h0_text": "H0,M2: (Delta %EG) B_PSI <= (Delta %EG) B_K"
    }
}

print("\n" + "="*80)
print("**STATISTISCHE EVALUATION & GRAFIKGENERIERUNG: GERICHTETE SYSTEMVERSCHIEBUNG**")
print("="*80)

# ==============================================================================
# 3. STATISTISCHE HYPOTHESENPRÜFUNG, EFFEKTMESSUNG & PLOTTING
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
    
    # 3.2 Extraktion der Differenzenmaße
    y_psi = df_psi[m_info["target_delta"]].values
    y_k   = df_k[m_info["target_delta"]].values
    
    # Deskriptive Kennzahlen
    mean_psi = np.mean(y_psi)
    mean_k   = np.mean(y_k)
    print(f"   Deskriptive Mittelwerte der gerichteten Verschiebung (Zt1 - Zt0):")
    print(f"     - Experimentalgruppe (B_PSI): {mean_psi:+.2f}%")
    print(f"     - Kontrollgruppe (B_K):      {mean_k:+.2f}%")
    
    # 3.3 Einseitiger Wilcoxon-Vorzeichen-Rang-Test
    try:
        res = wilcoxon(y_psi, y_k, alternative='greater', method='approx')
        stat_w = res.statistic
        p_val = res.pvalue
        
        # 3.4 Mathematische Ermittlung von Z_stat zur Berechnung von r (Formel 3.13)
        expected_w = (N_psi * (N_psi + 1)) / 4
        se_w = np.sqrt((N_psi * (N_psi + 1) * (2 * N_psi + 1)) / 24)
        z_stat = (stat_w - expected_w) / se_w
        
        # Standardisierte Effektstärke r
        r_effect = abs(z_stat) / np.sqrt(N_psi)
        
        if r_effect >= 0.5:
            cohen_label = "Starker experimenteller Steuerungseffekt"
        elif r_effect >= 0.3:
            cohen_label = "Mittlerer experimenteller Steuerungseffekt"
        elif r_effect >= 0.1:
            cohen_label = "Schwacher experimenteller Steuerungseffekt"
        else:
            cohen_label = "Kein nennenswerter Effekt"
            
        print(f"\n   Statistische Testkennzahlen (alpha = 0.05, einseitig):")
        print(f"     - Teststatistik (W):       {stat_w:.1f}")
        print(f"     - Standardisierte Z_stat:  {z_stat:+.3f}")
        print(f"     - p-Wert:                  {p_val:.5f}")
        print(f"     - Effektstärke (r):        {r_effect:.3f} ({cohen_label})")
        
        if p_val < 0.05:
            print(f"   🎉 **SIGNIFIKANZ: Die Nullhypothese wird ABGELEHNT.**")
        else:
            print(f"   🛑 **ERGEBNIS: Die Nullhypothese wird BEIBEHALTEN.**")
            
        # Kombinierten Datensatz für das Plotting bauen
        plot_df = pd.concat([
            pd.DataFrame({m_info["target_delta"]: y_k, "Bedingung": "Kontrollgruppe ($B_K$)"}),
            pd.DataFrame({m_info["target_delta"]: y_psi, "Bedingung": "PSI-Chatbot ($B_{PSI}$)"})
        ])
        
        # ==============================================================================
        # GRAFIK 1: BOXPLOT & STRIPPLOT
        # ==============================================================================
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.boxplot(
            ax=ax1, data=plot_df, x="Bedingung", y=m_info["target_delta"], 
            palette=["#bdc3c7", "#3498db"], width=0.5, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
        )
        sns.stripplot(ax=ax1, data=plot_df, x="Bedingung", y=m_info["target_delta"], color="black", alpha=0.3, jitter=0.15)
        
        ax1.set_title(f"{m_info['title']}\nVerteilungs- und Dispersionsanalyse ($N={N_psi}$)", fontweight='bold')
        ax1.set_ylabel(m_info["axis_label"])
        ax1.set_xlabel("")
        
        plt.tight_layout()
        plt.savefig(m_info["file_boxplot"], dpi=300)
        plt.close(fig1)
        print(f"   💾 Boxplot gespeichert unter: {m_info['file_boxplot']}")

        # ==============================================================================
        # GRAFIK 2: VEKTORPFEILE (ACHSENVERSCHIEBUNG)
        # ==============================================================================
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        
        # Rekonstruktion der absoluten Werte für das System-Trading (falls Spalten vorhanden)
        try:
            mean_src_k = df_k[f"t0_{m_info['source_sys']}"].mean()
            mean_tgt_k = df_k[f"t1_{m_info['target_sys']}"].mean()
            mean_src_psi = df_psi[f"t0_{m_info['source_sys']}"].mean()
            mean_tgt_psi = df_psi[f"t1_{m_info['target_sys']}"].mean()
        except:
            # Fallback falls absolute T0/T1 Werte nicht separat in der CSV liegen: rein deskriptive Visualisierung
            mean_src_k, mean_tgt_k = 50.0, 50.0 + mean_k
            mean_src_psi, mean_tgt_psi = 50.0, 50.0 + mean_psi

        # Plot der Vektorpfeile für die Achsenverschiebung
        ax2.plot([0, 1], [mean_src_k, mean_tgt_k], marker='o', color="#7f8c8d", linestyle="--", linewidth=2, label="Kontrollgruppe ($B_K$)")
        ax2.annotate('', xy=(1, mean_tgt_k), xytext=(0, mean_src_k), arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))
        
        ax2.plot([0, 1], [mean_src_psi, mean_tgt_psi], marker='o', color="#2980b9", linestyle="-", linewidth=3, label="PSI-Chatbot ($B_{PSI}$)")
        ax2.annotate('', xy=(1, mean_tgt_psi), xytext=(0, mean_src_psi), arrowprops=dict(arrowstyle="->", color="#2980b9", lw=3))
        
        ax2.set_title(f"{m_info['title']}\nVerschiebung der Systemachsen", fontweight='bold')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels([f"Ausgangssystem\n({m_info['source_sys']})", f"Zielsystem\n({m_info['target_sys']})"])
        ax2.set_ylabel("Systemgewichtung (%)")
        ax2.legend(loc="upper left")
        
        plt.tight_layout()
        plt.savefig(m_info["file_vector"], dpi=300)
        plt.close(fig2)
        print(f"   💾 Vektorgrafik gespeichert unter: {m_info['file_vector']}")

    except Exception as statistical_error:
        print(f"   ❌ FEHLER bei der statistischen Verarbeitung/Visualisierung: {statistical_error}")

