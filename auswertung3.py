
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon, rankdata


# ==============================================================================
# 0. GLOBALE EINSTELLUNGEN
# ==============================================================================

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 100
})

CSV_FILE = "masterarbeit_raw_data_typed.csv"
ALPHA = 0.05


# ==============================================================================
# HILFSFUNKTION: NUMERISCHE DATENKONVERTIERUNG
# ==============================================================================

def to_numeric(series):
    """
    Konvertiert CSV-Werte sicher in numerische Werte.
    Unterstützt Dezimalkommas und Prozentzeichen.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(cleaned, errors="coerce")


# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================

try:
    df = pd.read_csv(CSV_FILE, sep=";")

    print(
        f"Datensatz erfolgreich geladen. "
        f"Gesamte Zeilenanzahl: {len(df)}"
    )

except Exception as e:
    print(f"Fehler beim Laden der Datei: {e}")
    raise SystemExit


# Benötigte Pflichtspalten prüfen
required_columns = [
    "Szenario",
    "Bedingung",
    "Iteration",
    "delta_IVS",
    "delta_EG"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Folgende Pflichtspalten fehlen in der CSV: {missing_columns}"
    )


# Ziel-Deltas numerisch konvertieren
df["delta_IVS"] = to_numeric(df["delta_IVS"])
df["delta_EG"] = to_numeric(df["delta_EG"])

# Iterationen für die spätere Paarbildung vereinheitlichen
df["Iteration"] = df["Iteration"].astype("string").str.strip()

print("Zielvariablen numerisch konvertiert.")


# ==============================================================================
# 2. EXPERIMENTELLE HYPOTHESEN-KONFIGURATION
# ==============================================================================

modulations_annahmen = {

    "M1": {
        "title": "Erste Modulationsannahme (Willensbahnung: IG -> IVS)",
        "scenario_filter": "XM1",
        "target_delta": "delta_IVS",
        "axis_label": r"Gerichtete Verschiebung $\Delta$ %IVS",
        "file_boxplot": "plot_m1_willensbahnung_boxplot.png",
        "h1_text": "H1,M1: (Delta %IVS) B_PSI > (Delta %IVS) B_K",
        "h0_text": "H0,M1: (Delta %IVS) B_PSI <= (Delta %IVS) B_K"
    },

    "M2": {
        "title": "Zweite Modulationsannahme (Selbstentwicklung: OES -> EG)",
        "scenario_filter": "XM2",
        "target_delta": "delta_EG",
        "axis_label": r"Gerichtete Verschiebung $\Delta$ %EG",
        "file_boxplot": "plot_m2_selbstentwicklung_boxplot.png",
        "h1_text": "H1,M2: (Delta %EG) B_PSI > (Delta %EG) B_K",
        "h0_text": "H0,M2: (Delta %EG) B_PSI <= (Delta %EG) B_K"
    }
}


print("\n" + "=" * 80)
print(
    "STATISTISCHE EVALUATION & GRAFIKGENERIERUNG: "
    "GERICHTETE SYSTEMVERSCHIEBUNG"
)
print("=" * 80)


# Speicher für die kombinierte Grafik am Ende
global_plot_data = []


# ==============================================================================
# 3. STATISTISCHE HYPOTHESENPRÜFUNG, EFFEKTMESSUNG & BOXPLOTT
# ==============================================================================

for m_id, m_info in modulations_annahmen.items():

    print(f"\nANALYSE: {m_info['title']}")
    print(f"Prüfe H1: {m_info['h1_text']}")
    print(f"Gegen H0: {m_info['h0_text']}")
    print("-" * 70)

    try:

        # ======================================================================
        # 3.1 SUBGRUPPEN-FILTERUNG
        # ======================================================================

        target = m_info["target_delta"]

        df_psi = df[
            (df["Szenario"] == m_info["scenario_filter"]) &
            (df["Bedingung"] == "B_PSI")
        ].copy()

        df_k = df[
            (df["Szenario"] == m_info["scenario_filter"]) &
            (df["Bedingung"] == "B_K")
        ].copy()

        if df_psi.empty or df_k.empty:
            print(
                f"Keine Daten für Szenario "
                f"{m_info['scenario_filter']} gefunden."
            )
            continue

        # ======================================================================
        # 3.2 PAARBILDUNG ANHAND DER ITERATION
        # ======================================================================

        psi_pairs = df_psi[["Iteration", target]].copy()
        k_pairs = df_k[["Iteration", target]].copy()

        psi_pairs[target] = to_numeric(psi_pairs[target])
        k_pairs[target] = to_numeric(k_pairs[target])

        # Eindeutigkeit der Iterationen prüfen
        if psi_pairs["Iteration"].duplicated().any():
            raise ValueError(
                "Doppelte Iterationen in der Experimentalgruppe."
            )

        if k_pairs["Iteration"].duplicated().any():
            raise ValueError(
                "Doppelte Iterationen in der Kontrollgruppe."
            )

        # Tatsächlich verbundene Beobachtungen zusammenführen
        paired_df = pd.merge(
            psi_pairs,
            k_pairs,
            on="Iteration",
            how="inner",
            suffixes=("_psi", "_k"),
            validate="one_to_one"
        )

        # Fehlende Werte aus vollständigen Paaren ausschließen
        paired_df = paired_df.dropna(
            subset=[f"{target}_psi", f"{target}_k"]
        )

        paired_df = paired_df.sort_values(
            by="Iteration"
        ).reset_index(drop=True)

        if paired_df.empty:
            print("Keine vollständigen Datenpaare vorhanden.")
            continue

        y_psi = paired_df[f"{target}_psi"].to_numpy(dtype=float)
        y_k = paired_df[f"{target}_k"].to_numpy(dtype=float)

        N_pairs = len(paired_df)

        print(
            f"Stichprobenarchitektur: "
            f"N = {N_pairs} tatsächlich verbundene Paare"
        )

        # ======================================================================
        # 3.3 DESKRIPTIVE STATISTIK
        # ======================================================================

        mean_k = np.mean(y_k)
        median_k = np.median(y_k)

        mean_psi = np.mean(y_psi)
        median_psi = np.median(y_psi)

        print("\nDESKRIPTIVE STATISTIK:")

        print(
            f"Kontrollgruppe (B_K): "
            f"Mittelwert = {mean_k:+.2f}%, "
            f"Median = {median_k:+.2f}%"
        )

        print(
            f"Experimentalgruppe (B_PSI): "
            f"Mittelwert = {mean_psi:+.2f}%, "
            f"Median = {median_psi:+.2f}%"
        )

        # Differenz der gepaarten Beobachtungen
        differences = y_psi - y_k

        # Null-Differenzen ausschließen
        nonzero_differences = differences[differences != 0]

        N_effective = len(nonzero_differences)

        # ======================================================================
        # 3.4 EINSEITIGER WILCOXON-VORZEICHEN-RANG-TEST
        # ======================================================================

        if N_effective == 0:

            stat_w = 0.0
            z_stat = 0.0
            p_val = 1.0
            r_effect = 0.0

            print(
                "\nAlle gepaarten Differenzen sind null. "
                "Es liegt kein messbarer Unterschied vor."
            )

        else:

            res = wilcoxon(
                y_psi,
                y_k,
                alternative="greater",
                zero_method="wilcox",
                correction=False,
                method="approx"
            )

            stat_w = float(res.statistic)
            p_val = float(res.pvalue)

            # ==================================================================
            # 3.5 Z-BERECHNUNG MIT BINDUNGSKORREKTUR
            # ==================================================================

            abs_differences = np.abs(nonzero_differences)

            # Durchschnittsränge bei Bindungen
            ranks = rankdata(
                abs_differences,
                method="average"
            )

            # Summe der positiven Ränge
            W_positive = np.sum(
                ranks[nonzero_differences > 0]
            )

            n = N_effective

            # Erwartungswert
            expected_w = n * (n + 1) / 4

            # Bindungsgruppen ermitteln
            _, tie_counts = np.unique(
                abs_differences,
                return_counts=True
            )

            # Varianz mit Bindungskorrektur
            tie_correction = np.sum(
                tie_counts ** 3 - tie_counts
            ) / 48

            variance_w = (
                n * (n + 1) * (2 * n + 1) / 24
                - tie_correction
            )

            if variance_w > 0:

                se_w = np.sqrt(variance_w)

                z_stat = (
                    W_positive - expected_w
                ) / se_w

                # Gerichtete Effektstärke
                r_effect = z_stat / np.sqrt(n)

            else:

                z_stat = 0.0
                r_effect = 0.0

        # ======================================================================
        # 3.6 INTERPRETATION DER EFFEKTSTÄRKE
        # ======================================================================

        abs_r = abs(r_effect)

        if abs_r >= 0.5:
            effect_label = "Starker Effekt"

        elif abs_r >= 0.3:
            effect_label = "Mittlerer Effekt"

        elif abs_r >= 0.1:
            effect_label = "Schwacher Effekt"

        else:
            effect_label = "Vernachlässigbarer Effekt"

        if r_effect > 0:
            effect_direction = "in hypothesenkonformer Richtung"

        elif r_effect < 0:
            effect_direction = "entgegen der Hypothesenrichtung"

        else:
            effect_direction = "ohne erkennbare Richtung"

        # ======================================================================
        # 3.7 AUSGABE DER TESTKENNZAHLEN
        # ======================================================================

        print(
            f"\nStatistische Testkennzahlen "
            f"(alpha = {ALPHA}, einseitig):"
        )

        print(f"Anzahl vollständiger Paare: {N_pairs}")
        print(f"Anzahl nicht-null Differenzen: {N_effective}")
        print(f"Teststatistik (W): {stat_w:.1f}")
        print(f"Standardisierte Z_stat: {z_stat:+.3f}")
        print(f"p-Wert: {p_val:.6g}")

        print(
            f"Effektstärke (r): {r_effect:+.3f} "
            f"({effect_label}, {effect_direction})"
        )

        if p_val < ALPHA:
            print(
                "\nSIGNIFIKANZ: "
                "Die Nullhypothese wird ABGELEHNT."
            )

        else:
            print(
                "\nERGEBNIS: "
                "Die Nullhypothese kann NICHT verworfen werden."
            )

        # ======================================================================
        # 4. GRAFIK: BOXPLOT & STRIPPLOT
        # ======================================================================

        plot_df = pd.concat([
            pd.DataFrame({
                target: y_k,
                "Bedingung": "Kontrollgruppe ($B_K$)"
            }),
            pd.DataFrame({
                target: y_psi,
                "Bedingung": "PSI-Chatbot ($B_{PSI}$)"
            })
        ], ignore_index=True)

        # Daten für globale Grafik zwischenspeichern
        df_for_global = plot_df.copy()

        df_for_global.columns = ["Wert", "Bedingung"]

        df_for_global["Metrik"] = (
            "EM1: $\Delta$ %IVS"
            if m_id == "M1"
            else "EM2: $\Delta$ %EG"
        )

        global_plot_data.append(df_for_global)

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.boxplot(
            ax=ax,
            data=plot_df,
            x="Bedingung",
            y=target,
            hue="Bedingung",
            palette=["#bdc3c7", "#3498db"],
            width=0.5,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 8
            },
            legend=False
        )

        sns.stripplot(
            ax=ax,
            data=plot_df,
            x="Bedingung",
            y=target,
            color="black",
            alpha=0.3,
            jitter=0.15
        )

        ax.set_title(
            f"{m_info['title']}\n"
            f"Verteilungs- und Dispersionsanalyse "
            f"($N={N_pairs}$)",
            fontweight="bold"
        )

        ax.set_ylabel(m_info["axis_label"])
        ax.set_xlabel("")

        plt.tight_layout()

        plt.savefig(
            m_info["file_boxplot"],
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"Boxplot gespeichert unter: "
            f"{m_info['file_boxplot']}"
        )

    except Exception as e:

        print(
            f"\nFehler bei der statistischen "
            f"Verarbeitung/Visualisierung: {e}"
        )

        plt.close("all")
        continue


# ==============================================================================
# 4.5 NEUE GESAMTGRAFIK: EFFEKTMASSE EM1 & EM2
# ==============================================================================

if len(global_plot_data) == 2:

    try:

        print(
            "\nGeneriere zusammenfassende Grafik "
            "für EM1 und EM2..."
        )

        combined_df = pd.concat(
            global_plot_data,
            ignore_index=True
        )

        # Subplots erstellen (1 Zeile, 2 Spalten)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11, 5),
            sharey=True
        )

        metrics = [
            "EM1: $\Delta$ %IVS",
            "EM2: $\Delta$ %EG"
        ]

        titles = [
            r"$EM_1 = \Delta\ \%IVS$" "\n"
            r"($\%IVS_{Zt1} - \%IVS_{Zt0}$)",

            r"$EM_2 = \Delta\ \%EG$" "\n"
            r"($\%EG_{Zt1} - \%EG_{Zt0}$)"
        ]

        for i, met in enumerate(metrics):

            sub_df = combined_df[
                combined_df["Metrik"] == met
            ]

            ax = axes[i]

            sns.boxplot(
                ax=ax,
                data=sub_df,
                x="Bedingung",
                y="Wert",
                hue="Bedingung",
                palette=["#bdc3c7", "#3498db"],
                width=0.4,
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 7
                },
                legend=False
            )

            sns.stripplot(
                ax=ax,
                data=sub_df,
                x="Bedingung",
                y="Wert",
                color="black",
                alpha=0.25,
                jitter=0.12
            )

            ax.set_title(
                titles[i],
                fontweight="bold",
                pad=10
            )

            ax.set_xlabel("")

            if i == 0:
                ax.set_ylabel(
                    "Gerichtete Systemverschiebung (%)"
                )

            else:
                ax.set_ylabel("")

        plt.suptitle(
            "Vergleich der Effektmaße "
            "$EM_1$ und $EM_2$ über die Bedingungen",
            fontweight="bold",
            fontsize=14,
            y=1.02
        )

        plt.tight_layout()

        output_filename = "plot_em1_em2_vergleich.png"

        plt.savefig(
            output_filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"Kombinierte Grafik erfolgreich gespeichert "
            f"unter: {output_filename}"
        )

    except Exception as e:

        print(
            f"Fehler bei der Generierung "
            f"der kombinierten Grafik: {e}"
        )

        plt.close("all")


# ==============================================================================
# 5. ABSCHLUSS
# ==============================================================================

print("\n" + "=" * 80)
print("AUSWERTUNG ABGESCHLOSSEN")
print("=" * 80)

print("\nErzeugte Grafiken:")
print("  - plot_m1_willensbahnung_boxplot.png")
print("  - plot_m2_selbstentwicklung_boxplot.png")
print("  - plot_em1_em2_vergleich.png (NEU)")

print(
    "\nStatistische Auswertung und "
    "Boxplot-Generierung beendet."
)
