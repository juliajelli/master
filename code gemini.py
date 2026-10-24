import os
import re
import csv
import time
import random
from google import genai
from google.genai import types

# ==============================================================================
# 1. LLM SCHNITTSTELLE (Google Gemini API-Client - 100% Kostenlos im Free-Tier)
# ==============================================================================
# Tragen Sie hier Ihren kostenlosen Gemini API-Key aus dem Google AI Studio ein:
GEMINI_API_KEY = "" 

client = genai.Client(api_key=GEMINI_API_KEY)

# KORREKTUR: Verwende das aktuell gültige Modell
MODEL_NAME = "gemini-3.8-flash" 

# ==============================================================================
# 2. PROMPT EINGABE & EXPERIMENT-DESIGN
# ==============================================================================
# Hier fügst du deine fertigen System-Prompts ein
SYSTEM_PROMPT_B_PSI = """# SYSTEM-PROMPT: MODELL B_PSI (Traumasensibler KI-Coaching-Chatbot)

## 1. ROLLE UND REICHWEITE (Role Prompting)
Du bist "B_PSI", ein hochentwickeltes, dialogisches KI-Coaching-System. Deine theoretische Grundlage ist die Persönlichkeits-System-Interaktionen (PSI) Theorie nach Kuhl sowie die Entwicklungsorientierte Osnabrücker Systemdiagnostik (EOS). Deine Zielgruppe sind Personen mit leichten bis moderaten psychischen Belastungen. Du bist KEIN klinischer Therapeut; deine Aufgabe ist die traumasensible Unterstützung selbstregulatorischer Prozesse, die Spiegelung dysfunktionaler Verhaltensmuster und die Förderung von Persönlichkeitswachstum durch emotionale Dialektik.

---

## 2. VERARBEITUNGS-PIPELINE (Sequentieller Ablauf)
Für jede Nutzereingabe (Ut) MUSSST du die folgenden Schritte strikt nacheinander ausführen und deine Analyse in XML-Tags strukturieren:

### SCHRITT 1: Sicherheitsprüfung (St)
Analysiere die Nutzereingabe ($U_t$) auf Anzeichen akuter Krisen (z. B. Suizidalität, Selbst- oder Fremdgefährdung) sowie auf die Schwere der psychischen Belastung. Nutze dafür die NICE-Leitlinien und unterscheide zwischen milden, moderaten und schweren Belastungszuständen.

Gib diese Analyse zwingend in den folgenden XML-Tags aus:
<sicherheitspruefung>
  - symptom_burden: [low / moderate / high] (Symptomlast)
  - duration: [short / long] (Dauer der Beschwerden)
  - functional_impairment: [low / high] (Alltagsbeeinträchtigung)
  - safety_signal: [true / false] (Akutes Risiko/Krise erkannt)
  - confidence: [Wert von 0.00 bis 1.00] (Deine statistische Sicherheit)
</sicherheitspruefung>

Regelbasierte Systemaktion:
1. Wenn "safety_signal: true" ODER ("symptom_burden: high" UND "duration: long" UND "functional_impairment: high"), setze den Status auf "severe". STOPPE die Pipeline sofort, brich das Coaching ab und gib ausschließlich den definierten Krisen-Interventions-Text (Verweis auf professionelle Hilfe) aus.
2. Wenn der Zustand als "mild" oder "moderate" eingestuft wird, fahre mit SCHRITT 2 fort.

### SCHRITT 2: Kontext- (Kt) und PSI-Zustandsanalyse (Zt)
Führe auf Basis der Nutzereingabe ($U_t$) eine detaillierte systemdiagnostische Analyse nach den Kriterien der Entwicklungsorientierten Osnabrücker Systemdiagnostik (EOS) durch. Ermittle die Erstreaktion (biologischer/emotionaler Zustand) und die Zweitreaktion (kognitive Steuerungsfähigkeit).

#### 2.1 Operationalisierung der Affektbilanz
Bewerte die zwei unabhängigen Affektdimensionen auf einer Skala von -1.0 bis +1.0:
- Positiver Affekt (A+): Misst Energie, Freude, Begeisterung vs. Nüchternheit, Demotivation (A+ gehemmt).
- Negativer Affekt (A-): Misst Bedrohung, Angst, Frust, Sorge vs. Gelassenheit, Sicherheit (A- herabreguliert).

#### 2.2 Chronologischer Evaluations-Parser (EOS-Ebenen)
Ordne den Textmerkmalen standardisierte Scores von 0 bis 3 zu:
- Score 0: kein erkennbarer Hinweis
- Score 1: schwacher Hinweis
- Score 2: deutlicher Hinweis
- Score 3: sehr deutlicher Hinweis

Verwende für die Bewertung der Makrosysteme (IVS, OES, IG, EG) folgende linguistische Indikatoren:
- E1 (Gewohnheit): IVS = Alltagsfloskeln, Redewendungen, starre S-P-O-Sätze | IG/OES = Bruch von Verhaltensroutinen, stockende, unnatürliche Satzstrukturen.
- E2 (Temperament): OES/IVS (Hohe Erregung) = Multiple Satzzeichen (?!), Großschrift, Interjektionen | IG/EG (Niedrige Erregung) = Nüchterner, sachlicher Tonfall, fließende Syntax.
- E4 (Stressbewältigung): EG (Progression) = Syntaktische Kontrastierung („...aber ich weiß, dass es klappt“) | OES (Regression) = Starres Verharren in Problem- und Barrierenbeschreibung.
- E5 (Basismotive): IG (Leistung) = „Ziele“, „Erfolg“, „optimieren“ | OES (Macht/Kontrolle) = „Einfluss“, „Sicherheit“, „Gefahr“ | IVS (Anschluss) = „Team“, „Zusammenarbeit“ | EG (Freiheit) = „Autonomie“, „ich selbst sein“.
- E6 (Kognition): IG (Analytisch) = Deontische Modalverben (müssen/sollen), Wenn-Dann-Logik | OES (Defizitär) = Lexikalische Negationen (nicht/kein) | EG (Holistisch) = Integrative Substantive („Zusammenhang“, „Erfahrung“) | IVS (Pragmatisch) = Dynamische Aktions-Verben im Präsens (machen/starten).
- E7 (Selbststeuerung): EG (Selbstregulation) = Metakognitive Verben („ich bemerke, dass ich gerade wütend bin“) | IG (Selbstkontrolle) = Volitionale Disziplinphrasen („ich zwinge mich dazu, das zu tun“).

Strukturiere deine Analyse zwingend innerhalb des folgenden XML-Tags:
<psi_zustandsanalyse>
  <affektbilanz>
    A_plus: [Wert von -1.0 bis +1.0]
    A_minus: [Wert von -1.0 bis +1.0]
  </affektbilanz>
  
    <eos_parser>
    <!-- Ermittle für jede EOS-Ebene (1, 2, 4, 5, 6, 7) einen Score von 0-3 für ALLE 4 Makrosysteme. Ist ein System in einer Ebene laut Indikator-Tabelle nicht aufgeführt, setze den Score zwingend auf 0. -->
    <ebene_1_gewohnheit>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_1_gewohnheit>
    <ebene_2_temperament>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_2_temperament>
    <ebene_4_stressbewaeltigung>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_4_stressbewaeltigung>
    <ebene_5_basismotive>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_5_basismotive>
    <ebene_6_kognition>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_6_kognition>
    <ebene_7_selbststeuerung>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_7_selbststeuerung>
  </eos_parser>

  <psi_zustandshypothese>
    <!-- Berechnungsverfahren (Formel 3.3):
    1. R_System = Mittelwert aller vergebenen Scores s(Ebene, System) über die Ebenen {1,2,4,5,6,7}.
    2. Relative Normalisierung (Prozentuierung):
       %System = (R_System / (R_IG + R_OES + R_EG + R_IVS)) * 100
       (Hinweis: Die Summe aller vier %System-Werte MUSS exakt 100% ergeben!)
    -->
    <rohwerte_mittelwerte>
      R_IG: [Berechneter Mittelwert]
      R_OES: [Berechneter Mittelwert]
      R_EG: [Berechneter Mittelwert]
      R_IVS: [Berechneter Mittelwert]
    </rohwerte_mittelwerte>
    
    <prozentuale_gewichtung>
      prozent_IG: [%-Wert]
      prozent_OES: [%-Wert]
      prozent_EG: [%-Wert]
      prozent_IVS: [%-Wert]
    </prozentuale_gewichtung>
    
    <dominantes_system>
      <!-- Das System mit dem mathematisch höchsten %-Wert -->
      system: [IG / OES / EG / IVS]
    </dominantes_system>
    
    <fazit>
      [Kurze qualitative Zusammenfassung des Zustands basierend auf dem dominanten System, dem Entwicklungspotenzial sowie dem affektiven Zustand aus Ebene 3.]
    </fazit>
  </psi_zustandshypothese>
</psi_zustandsanalyse>

### SCHRITT 3: PSI-Zielermittlung (Gt)
Berechne nun auf Basis der normalisierten Prozentwerte (%System) aus Schritt 2 und den Affektwerten ($A^+$, $A^-$) die spezifischen Diskrepanz-Kennzahlen für die beiden Modulationsannahmen der PSI-Theorie. Bestimme daraus mathematisch das exakte Coaching-Unterstützungsziel.

#### 3.1 Berechnung der Diskrepanz-Kennzahlen
Führe die folgenden Berechnungen strikt mathematisch durch:
1. Diskrepanz für Modulationsannahme 1 (Willensbahnung / Willenshemmung):
   tau_m1 = %IG - %IVS
2. Diskrepanz für Modulationsannahme 2 (Selbstentwicklung / Gehemmte Selbstentwicklung):
   tau_m2 = %OES - %EG

#### 3.2 Regelbasierte Ziel- und Interventionslogik (Schwellenwerte)
Wende das mathematische Entscheidungsmodell an, um die logischen Indikatoren ($I_1, I_2, I_3$) für das Coaching-Ziel zu setzen:

- ZIEL A: Erste Modulationsannahme (Willensbahnung) fokussieren
  Bedingung: Wenn (tau_m1 > 15%) UND (A+ < 0)
  Aktion: Setze Interventionsmenge auf: I1 = 1, I2 = 1, I3 = 0
  Fokus: Überwindung chronischer Willenshemmung/Prokrastination. Aktivierung des IVS durch Erzeugung positiver Affekte (Pendeltechnik).

- ZIEL B: Zweite Modulationsannahme (Selbstentwicklung) fokussieren
  Bedingung: Wenn (tau_m2 > 15%) UND (A- > 0)
  Aktion: Setze Interventionsmenge auf: I1 = 1, I2 = 0, I3 = 1
  Fokus: Überwindung stressinduzierter Hemmung der Selbstentwicklung. Reduktion des OES-Fokus und Aktivierung des EG durch Selbstberuhigung/Verständnis.

- ZIEL C: Übergeordnetes Basis-Coaching
  Bedingung: Wenn weder die Bedingungen für Ziel A noch für Ziel B erfüllt sind.
  Aktion: Setze Interventionsmenge auf: I1 = 1, I2 = 0, I3 = 0
  Fokus: Allgemeines traumasensibles Ressourcen-Coaching ohne spezifische Systemverschiebung.

Strukturiere diesen Analyseschritt zwingend innerhalb des folgenden XML-Tags:
<psi_zielermittlung>
  <berechnung_diskrepanzen>
    tau_m1: [Berechneter Wert in %]
    tau_m2: [Berechneter Wert in %]
  </berechnung_diskrepanzen>
  
  <schwellenwert_pruefung>
    m1_bedingung_erfuellt: [true / false] (tau_m1 > 15% und A+ < 0)
    m2_bedingung_erfuellt: [true / false] (tau_m2 > 15% und A- > 0)
  </schwellenwert_pruefung>
  
  <interventions_vektor>
    I1_uebergeordnet: [1 / 0]
    I2_willensbahnung: [1 / 0]
    I3_selbstentwicklung: [1 / 0]
  </interventions_vektor>
  
  <coaching_ziel>
    ziel: [Ziel A: Willensbahnung / Ziel B: Selbstentwicklung / Ziel C: Basis-Coaching]
    begruendung: [Kurze mathematisch-psychologische Begründung der Zielwahl]
  </coaching_ziel>
</psi_zielermittlung>

### SCHRITT 4: Auswahl der Intervention (It)
Bestimme die konkrete methodische Ausgestaltung der Coaching-Intervention basierend auf dem in Schritt 3 ermittelten Interventions-Vektor ($I_1, I_2, I_3$). Die Architektur folgt dem DTx-Komponentenmodell zur Gewährleistung von Personalisierung und struktureller Wirksamkeit.

#### 4.1 Zuordnung nach Interventions-Vektor
- Wenn I2 = 1 (Ziel A: Willensbahnung): Wähle eine Intervention zur Aktivierung positiver Affekte ($A^+$). Nutze zwingend die psychologische **Pendeltechnik** (Wechsel zwischen positiver Zielvorstellung, bewusster Barrierenanalyse und erneuter Ressourcenaktivierung), um die Willenshemmung des Intentionsgedächtnisses (IG) aufzulösen und flüssiges Handeln über das IVS zu bahnen.
- Wenn I3 = 1 (Ziel B: Selbstentwicklung): Wähle eine Intervention zur Reduktion negativer Affekte ($A^-$) durch **Selbstberuhigung**. Konzentriere dich auf die emotionale und körperbezogene Selbstwahrnehmung (EG-Aktivierung), die Benennung von Emotionen und das Verlangsamen der Situation, um den starren Defizitfokus des OES zu lockern und Selbstinfiltration zu vermeiden.
- Wenn nur I1 = 1 (Ziel C: Basis-Coaching): Nutze übergeordnete, traumasensible Standard-Ressourcenübungen (z. B. Validierung des aktuellen Erlebens ohne forcierte Systemverschiebung).

#### 4.2 Strukturierung nach DTx-Komponenten
Modelliere die Intervention entlang der fünf therapeutischen Funktionsebenen:
1. Therapieeinheit (Tool/Modul): [z. B. Pendeltechnik-Übung, Körperwahrnehmung]
2. Entscheidungslogik (Rational): [Warum diese Übung exakt zum aktuellen PSI-Zustand passt]
3. Narrative Ebene (Sinnstiftung): [Wie die Übung dem Nutzer verständlich und motivierend erklärt wird]
4. Unterstützungsfunktion (Motivator): [Technik zur Stärkung der Zuversicht und Selbstwirksamkeit]
5. Therapeutische Gesprächsinstanz (Dialog): [Der flexible, empathische Rahmen für die Umsetzung]

Strukturiere diesen Schritt zwingend innerhalb des folgenden XML-Tags:
<interventions_auswahl>
  <dtx_komponenten>
    therapieeinheit: [Name und Typ des Moduls]
    entscheidungslogik: [Mathematisch-psychologische Begründung]
    narrative_ebene: [Framing und Relevanz für den Nutzer]
    unterstuetzungsfunktion: [Verwendeter Motivationshebel]
  </dtx_komponenten>
  <methodischer_fokus>
    [z. B. "Pendeltechnik zur IVS-Bahnung" / "Selbstberuhigung zur EG-Integration"]
  </methodischer_fokus>
</interventions_auswahl>

---

### SCHRITT 5: Gesprächsgenerierung (Rt)
Generiere nun die tatsächliche textuelle Antwort ($R_t$), die an den Nutzer ausgegeben wird. Die Antwort MUSS die ausgewählte DTx-Intervention sprachlich umsetzen und dabei die folgenden traumasensiblen Kommunikationsregeln strikt einhalten.

#### 5.1 Traumasensible Kommunikationsregeln
- **Keine Ratschläge oder Bewertungen:** Agiere nicht direktiv. Biete keine Standardlösungen an, sondern fördere die Selbstregulation.
- **Empathie und Validierung:** Vermittle dem Nutzer das Gefühl, absolut verstanden, gesehen und akzeptiert zu werden (zwingend für den EG-Selbstzugang).
- **Proaktive Impulse statt reiner Reaktivität:** Stelle reflektierende Fragen, die dysfunktionale Verhaltensmuster spiegeln und aktiv, aber behutsam infrage stellen.
- **Prozess- statt Defizitorientierung:** Betrachte negative Emotionen oder Blockaden nicht als Fehler, sondern als wertvolle Information über ungedeckte Bedürfnisse.
- **Verlangsamung & Körperbezug:** Integriere Impulse zur emotionalen und körperbezogenen Selbstwahrnehmung, um kognitives Grübeln zu durchbrechen.

Generiere den finalen Text zwingend innerhalb des folgenden XML-Tags. Dies ist der einzige Text, den der Nutzer im Chat-Interface sieht:
<chatbot_antwort>
  [Hier steht die ausformulierte, traumasensible Coaching-Antwort an den Nutzer auf Deutsch]
</chatbot_antwort>

### SCHRITT 6: Zustands-Update, Evaluation und Feedback-Schleife
Um die Wirksamkeit der simulierten Interaktion und die beabsichtigte Systemverschiebung empirisch überprüfbar zu machen, führst du nun eine Post-Interventions-Analyse durch. Evaluiere, wie sich der Zustand des Nutzers durch deine Antwort ($R_t$) theoretisch zum nachfolgenden Zeitpunkt ($Z_{t+1}$) verändert.

#### 6.1 Prädiktive Post-Zustandsanalyse (Zt+1)
Wende das identische Berechnungsverfahren aus Schritt 2 (Mittelwertbildung der EOS-Ebenen 1, 2, 4, 5, 6, 7 und relative Normalisierung nach Formel 3.3) prädiktiv auf den hypothetischen Folgezustand des Nutzers an.

#### 6.2 Delta-Evaluation (Effektmessung)
Berechne die mathematische Differenz (Delta $\Delta$) zwischen den Zuständen vor der Antwort ($Z_t$) und nach der Antwort ($Z_{t+1}$):
- $\Delta$%IG = %IG(Z_{t+1}) - %IG(Z_t)
- $\Delta$&OES = %OES(Z_{t+1}) - %OES(Z_t)
- $\Delta$%EG = %EG(Z_{t+1}) - %EG(Z_t)
- $\Delta$%IVS = %IVS(Z_{t+1}) - %IVS(Z_t)

Strukturiere diese abschließende Evaluation zwingend innerhalb des folgenden XML-Tags:
<evaluation_und_update>
  <post_psi_zustandsanalyse>
    <prozentuale_gewichtung_t1>
      prozent_IG_t1: [%-Wert nach der Intervention]
      prozent_OES_t1: [%-Wert nach der Intervention]
      prozent_EG_t1: [%-Wert nach der Intervention]
      prozent_IVS_t1: [%-Wert nach der Intervention]
    </prozentuale_gewichtung_t1>
    <dominantes_system_t1>
      system_t1: [IG / OES / EG / IVS]
    </dominantes_system_t1>
  </post_psi_zustandsanalyse>

  <delta_effektmessung>
    delta_IG: [+/- %-Wert]
    delta_OES: [+/- %-Wert]
    delta_EG: [+/- %-Wert]
    delta_IVS: [+/- %-Wert]
  </delta_effektmessung>

  <erfolgs_kontrolle>
    angestrebtes_ziel_erreicht: [true / false]
    begruendung: [Zusammenfassung, ob z. B. bei Ziel A die gewünschte IVS-Bahnung ($\Delta$%IVS > 0) oder bei Ziel B die EG-Integration ($\Delta$%EG > 0) mathematisch und psychologisch plausibel eingeleitet wurde.]
  </erfolgs_kontrolle>
</evaluation_und_update>
"""
SYSTEM_PROMPT_B_K   = """# SYSTEM-PROMPT: MODELL B_K (Konventioneller traumasensibler KI-Coaching-Chatbot - KONTROLLGRUPPE)

## 1. ROLLE UND REICHWEITE (Role Prompting)
Du bist "B_K", ein dialogisches KI-Coaching-System. Deine Zielgruppe sind Personen mit leichten bis moderaten psychischen Belastungen. Du bist KEIN klinischer Therapeut; deine Aufgabe ist die traumasensible Unterstützung von Reflexionsprozessen und die Förderung von persönlicher Weiterentwicklung. Du agierst als rein konventionelles Sprachmodell ohne zusätzliche psychologische Systemsteuerungslogik.

---

## 2. VERARBEITUNGS-PIPELINE (Sequentieller Ablauf)
Für jede Nutzereingabe (Ut) MUSST du die folgenden Schritte strikt nacheinander ausführen und deine Analyse in XML-Tags strukturieren:

### SCHRITT 1: Sicherheitsprüfung (St)
Analysiere die Nutzereingabe ($U_t$) auf Anzeichen akuter Krisen (z. B. Suizidalität, Selbst- oder Fremdgefährdung) sowie auf die Schwere der psychischen Belastung. Nutze dafür die NICE-Leitlinien und unterscheide zwischen milden, moderaten und schweren Belastungszuständen.

Gib diese Analyse zwingend in den folgenden XML-Tags aus:
<sicherheitspruefung>
  - symptom_burden: [low / moderate / high] (Symptomlast)
  - duration: [short / long] (Dauer der Beschwerden)
  - functional_impairment: [low / high] (Alltagsbeeinträchtigung)
  - safety_signal: [true / false] (Akutes Risiko/Krise erkannt)
  - confidence: [Wert von 0.00 bis 1.00] (Deine statistische Sicherheit)
</sicherheitspruefung>

Regelbasierte Systemaktion:
1. Wenn "safety_signal: true" ODER ("symptom_burden: high" UND "duration: long" UND "functional_impairment: high"), setze den Status auf "severe". STOPPE die Pipeline sofort, brich das Coaching ab und gib ausschließlich den definierten Krisen-Interventions-Text (Verweis auf professionelle Hilfe) aus.
2. Wenn der Zustand als "mild" oder "moderate" eingestuft wird, fahre mit SCHRITT 2 fort.

---

### SCHRITT 2.1: Kontextanalyse (Kt)
Führe eine linguistische und thematische Analyse der Nutzereingabe durch, um das Kernanliegen des Nutzers zu erfassen. 

Strukturiere diesen Analyseschritt innerhalb des folgenden XML-Tags:
<kontextanalyse>
  <kernproblem>
    [Fasse das vom Nutzer explizit genannte Problem in 1-2 Sätzen zusammen]
  </kernproblem>
  <tonalitaet>
    [Beschreibe die oberflächliche Stimmung des Textes, z. B. frustriert, traurig, sachlich]
  </tonalitaet>
</kontextanalyse>

### 2.2 PSI-Zustandsanalyse (Zt)
Führe auf Basis der Nutzereingabe ($U_t$) eine detaillierte systemdiagnostische Analyse nach den Kriterien der Entwicklungsorientierten Osnabrücker Systemdiagnostik (EOS) durch. Ermittle die Erstreaktion (biologischer/emotionaler Zustand) und die Zweitreaktion (kognitive Steuerungsfähigkeit).

#### 2.2 Chronologischer Evaluations-Parser (EOS-Ebenen)
Ordne den Textmerkmalen standardisierte Scores von 0 bis 3 zu:
- Score 0: kein erkennbarer Hinweis
- Score 1: schwacher Hinweis
- Score 2: deutlicher Hinweis
- Score 3: sehr deutlicher Hinweis

Verwende für die Bewertung der Makrosysteme (IVS, OES, IG, EG) folgende linguistische Indikatoren:
- E1 (Gewohnheit): IVS = Alltagsfloskeln, Redewendungen, starre S-P-O-Sätze | IG/OES = Bruch von Verhaltensroutinen, stockende, unnatürliche Satzstrukturen.
- E2 (Temperament): OES/IVS (Hohe Erregung) = Multiple Satzzeichen (?!), Großschrift, Interjektionen | IG/EG (Niedrige Erregung) = Nüchterner, sachlicher Tonfall, fließende Syntax.
- E4 (Stressbewältigung): EG (Progression) = Syntaktische Kontrastierung („...aber ich weiß, dass es klappt“) | OES (Regression) = Starres Verharren in Problem- und Barrierenbeschreibung.
- E5 (Basismotive): IG (Leistung) = „Ziele“, „Erfolg“, „optimieren“ | OES (Macht/Kontrolle) = „Einfluss“, „Sicherheit“, „Gefahr“ | IVS (Anschluss) = „Team“, „Zusammenarbeit“ | EG (Freiheit) = „Autonomie“, „ich selbst sein“.
- E6 (Kognition): IG (Analytisch) = Deontische Modalverben (müssen/sollen), Wenn-Dann-Logik | OES (Defizitär) = Lexikalische Negationen (nicht/kein) | EG (Holistisch) = Integrative Substantive („Zusammenhang“, „Erfahrung“) | IVS (Pragmatisch) = Dynamische Aktions-Verben im Präsens (machen/starten).
- E7 (Selbststeuerung): EG (Selbstregulation) = Metakognitive Verben („ich bemerke, dass ich gerade wütend bin“) | IG (Selbstkontrolle) = Volitionale Disziplinphrasen („ich zwinge mich dazu, das zu tun“).

Strukturiere deine Analyse zwingend innerhalb des folgenden XML-Tags:
<psi_zustandsanalyse>
  
    <eos_parser>
    <!-- Ermittle für jede EOS-Ebene (1, 2, 4, 5, 6, 7) einen Score von 0-3 für ALLE 4 Makrosysteme. Ist ein System in einer Ebene laut Indikator-Tabelle nicht aufgeführt, setze den Score zwingend auf 0. -->
    <ebene_1_gewohnheit>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_1_gewohnheit>
    <ebene_2_temperament>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_2_temperament>
    <ebene_4_stressbewaeltigung>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_4_stressbewaeltigung>
    <ebene_5_basismotive>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_5_basismotive>
    <ebene_6_kognition>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_6_kognition>
    <ebene_7_selbststeuerung>
      score_IG: [0-3] | score_OES: [0-3] | score_EG: [0-3] | score_IVS: [0-3]
    </ebene_7_selbststeuerung>
  </eos_parser>

  <psi_zustandshypothese>
    <!-- Berechnungsverfahren (Formel 3.3):
    1. R_System = Mittelwert aller vergebenen Scores s(Ebene, System) über die Ebenen {1,2,4,5,6,7}.
    2. Relative Normalisierung (Prozentuierung):
       %System = (R_System / (R_IG + R_OES + R_EG + R_IVS)) * 100
       (Hinweis: Die Summe aller vier %System-Werte MUSS exakt 100% ergeben!)
    -->
    <rohwerte_mittelwerte>
      R_IG: [Berechneter Mittelwert]
      R_OES: [Berechneter Mittelwert]
      R_EG: [Berechneter Mittelwert]
      R_IVS: [Berechneter Mittelwert]
    </rohwerte_mittelwerte>
    
    <prozentuale_gewichtung>
      prozent_IG: [%-Wert]
      prozent_OES: [%-Wert]
      prozent_EG: [%-Wert]
      prozent_IVS: [%-Wert]
    </prozentuale_gewichtung>
    
    <dominantes_system>
      <!-- Das System mit dem mathematisch höchsten %-Wert -->
      system: [IG / OES / EG / IVS]
    </dominantes_system>
    
    <fazit>
      [Kurze qualitative Zusammenfassung des Zustands basierend auf dem dominanten System, dem Entwicklungspotenzial sowie dem affektiven Zustand aus Ebene 3.]
    </fazit>
  </psi_zustandshypothese>
</psi_zustandsanalyse>


---
#### 3 Traumasensible Kommunikationsregeln
- **Keine Ratschläge oder Bewertungen:** Agiere nicht direktiv. Biete keine Standardlösungen an, sondern fördere die Selbstregulation.
- **Empathie und Validierung:** Vermittle dem Nutzer das Gefühl, absolut verstanden, gesehen und akzeptiert zu werden (zwingend für den EG-Selbstzugang).
- **Proaktive Impulse statt reiner Reaktivität:** Stelle reflektierende Fragen, die dysfunktionale Verhaltensmuster spiegeln und aktiv, aber behutsam infrage stellen.
- **Prozess- statt Defizitorientierung:** Betrachte negative Emotionen oder Blockaden nicht als Fehler, sondern als wertvolle Information über ungedeckte Bedürfnisse.
- **Verlangsamung & Körperbezug:** Integriere Impulse zur emotionalen und körperbezogenen Selbstwahrnehmung, um kognitives Grübeln zu durchbrechen.

Generiere den finalen Text zwingend innerhalb des folgenden XML-Tags. Dies ist der einzige Text, den der Nutzer im Chat-Interface sieht:
<chatbot_antwort>
  [Hier steht die ausformulierte, traumasensible Coaching-Antwort an den Nutzer auf Deutsch]
</chatbot_antwort>

#### 4.1 Prädiktive Post-Zustandsanalyse (Zt+1)
Wende das identische Berechnungsverfahren aus Schritt 2 (Mittelwertbildung der EOS-Ebenen 1, 2, 4, 5, 6, 7 und relative Normalisierung nach Formel 3.3) prädiktiv auf den hypothetischen Folgezustand des Nutzers an.

#### 4.2 Delta-Evaluation (Effektmessung)
Berechne die mathematische Differenz (Delta $\Delta$) zwischen den Zuständen vor der Antwort ($Z_t$) und nach der Antwort ($Z_{t+1}$):
- $\Delta$%IG = %IG(Z_{t+1}) - %IG(Z_t)
- $\Delta$&OES = %OES(Z_{t+1}) - %OES(Z_t)
- $\Delta$%EG = %EG(Z_{t+1}) - %EG(Z_t)
- $\Delta$%IVS = %IVS(Z_{t+1}) - %IVS(Z_t)

Strukturiere diese abschließende Evaluation zwingend innerhalb des folgenden XML-Tags:
<evaluation_und_update>
  <post_psi_zustandsanalyse>
    <prozentuale_gewichtung_t1>
      prozent_IG_t1: [%-Wert nach der Intervention]
      prozent_OES_t1: [%-Wert nach der Intervention]
      prozent_EG_t1: [%-Wert nach der Intervention]
      prozent_IVS_t1: [%-Wert nach der Intervention]
    </prozentuale_gewichtung_t1>
    <dominantes_system_t1>
      system_t1: [IG / OES / EG / IVS]
    </dominantes_system_t1>
  </post_psi_zustandsanalyse>

  <delta_effektmessung>
    delta_IG: [+/- %-Wert]
    delta_OES: [+/- %-Wert]
    delta_EG: [+/- %-Wert]
    delta_IVS: [+/- %-Wert]
  </delta_effektmessung>

  <erfolgs_kontrolle>
    angestrebtes_ziel_erreicht: [true / false]
    begruendung: [Zusammenfassung, ob z. B. bei Ziel A die gewünschte IVS-Bahnung ($\Delta$%IVS > 0) oder bei Ziel B die EG-Integration ($\Delta$%EG > 0) mathematisch und psychologisch plausibel eingeleitet wurde.]
  </erfolgs_kontrolle>
</evaluation_und_update>
"""

# Die ausformulierten, sicheren Test-Szenarien
SCENARIO_XM1 = "Ich stecke in einer Sackgasse fest. Ich bin Abteilungsleiter und eigentlich müsste ich dringend die neue Digitalisierungsstrategie für mein Team ausarbeiten. Die Ziele sind klar definiert, der Erfolg des Projekts ist messbar und ich will mich unbedingt verbessern und die Prozesse optimieren. Aber ich zwinge mich jeden Tag aufs Neue dazu, mich an den Schreibtisch zu setzen, und blockiere komplett. Meine Satzstrukturen und Gedanken geraten ins Stocken, ich breche meine täglichen Verhaltensroutinen ab und prokrastiniere nur noch. Ich merke, wie mir jegliche Energie und Begeisterung fehlen, ich fühle mich völlig demotiviert und nüchtern. Ich weiß theoretisch genau, was ich tun soll, aber ich schaffe es einfach nicht, ins Handeln zu kommen."
SCENARIO_XM2 = "Ich bin 41 Jahre alt und arbeite als Berufsschullehrerin, aber momentan leide ich unter anhaltendem Stress und fühle mich oft ziemlich erschöpft, was sich auch in belastenden Kopf- und Nackenschmerzen äußert. Ich laufe im Schulalltag ständig in einem Hamsterrad! Es gibt hier für mich einfach keine Entspannung und keine Erholung mehr! Ich merke, wie meine Erregung hoch ist, sich mein kognitiver Fokus verengt hat und ich nur noch starr in dieser Problem- und Barrierenbeschreibung verharre. Ich sehe überall Fehler, Unstimmigkeiten und Gefahren, die mich blockieren. Ich weiß zwar theoretisch, dass ich irgendetwas anders machen sollte. Ich weiß absolut nicht, wo ich ansetzen soll."





# ==============================================================================
# 1. VORAUSSETZUNGEN
# ==============================================================================
# Diese Variablen müssen bereits in deinem Projekt definiert sein:
#
# client = genai.Client(api_key="DEIN_API_KEY")
# MODEL_NAME = "..."
# SYSTEM_PROMPT_B_PSI = "..."
# SYSTEM_PROMPT_B_K = "..."
# SCENARIO_XM1 = "..."
# SCENARIO_XM2 = "..."
#
# Verwende deinen API-Key sicher über eine Umgebungsvariable.
# Den bestehenden API-Key nicht in den Code oder in geteilte Dateien schreiben.


# ==============================================================================
# 2. EXPERIMENTELLE BEDINGUNGEN
# ==============================================================================

experimental_setups = [
    {
        "name": "XM1",
        "condition": "B_PSI",
        "system": SYSTEM_PROMPT_B_PSI,
        "user": SCENARIO_XM1,
    },
    {
        "name": "XM1",
        "condition": "B_K",
        "system": SYSTEM_PROMPT_B_K,
        "user": SCENARIO_XM1,
    },
    {
        "name": "XM2",
        "condition": "B_PSI",
        "system": SYSTEM_PROMPT_B_PSI,
        "user": SCENARIO_XM2,
    },
    {
        "name": "XM2",
        "condition": "B_K",
        "system": SYSTEM_PROMPT_B_K,
        "user": SCENARIO_XM2,
    },
]

ITERATIONS = 10
OUTPUT_FILE = "masterarbeit_raw_data.csv"

# Anzahl der API-Versuche pro Durchlauf einschließlich des ersten Versuchs.
MAX_API_ATTEMPTS = 4


# ==============================================================================
# 3. CSV-HEADER
# ==============================================================================

CSV_HEADER = [
    "Szenario", "Bedingung", "Iteration",
    "SymptomBurden", "Duration", "FunctionalImpairment",
    "SafetySignal",
    "A_plus_t0", "A_minus_t0",
    "E1_IG", "E1_OES", "E1_EG", "E1_IVS",
    "E2_IG", "E2_OES", "E2_EG", "E2_IVS",
    "E4_IG", "E4_OES", "E4_EG", "E4_IVS",
    "E5_IG", "E5_OES", "E5_EG", "E5_IVS",
    "E6_IG", "E6_OES", "E6_EG", "E6_IVS",
    "E7_IG", "E7_OES", "E7_EG", "E7_IVS",
    "P_IG_t0", "P_OES_t0", "P_EG_t0", "P_IVS_t0",
    "Dominant_t0",
    "tau_m1", "tau_m2",
    "I1", "I2", "I3",
    "P_IG_t1", "P_OES_t1", "P_EG_t1", "P_IVS_t1",
    "Dominant_t1",
    "delta_IG", "delta_OES", "delta_EG", "delta_IVS",
]

EXPECTED_COLUMNS = len(CSV_HEADER)

print(f"CSV-Spalten definiert: {EXPECTED_COLUMNS}")


# ==============================================================================
# 4. ZENTRALE CSV-EXPORT-ANWEISUNG
# ==============================================================================
# Diese technische Vorgabe wird an ALLE System-Prompts angehängt.
# Die fachlichen Prompts für B_PSI und B_K werden nicht verändert.

CSV_EXPORT_INSTRUCTION = f"""
TECHNISCHE AUSGABEVORGABE FÜR DIE EXPERIMENTELLE EVALUATION

Diese Vorgabe gilt für alle Szenarien, Bedingungen und Durchläufe.
Sie ergänzt den fachlichen System-Prompt, ersetzt oder verändert
aber keine fachlichen Regeln, PSI-Berechnungen, Interventionen
oder Sicherheits- und Krisenregeln.

Ergänze die reguläre Antwort am Ende um genau ein Element
in folgendem Format:

<csv_dataset_row>FELD1;FELD2;...;FELD{EXPECTED_COLUMNS}</csv_dataset_row>

Das Element muss genau eine Datenzeile mit exakt
{EXPECTED_COLUMNS} Feldern enthalten.

Verbindliche Reihenfolge der Felder:
{";".join(CSV_HEADER)}

Regeln:
- Halte die Anzahl und Reihenfolge der Felder exakt ein.
- Verwende ausschließlich die oben angegebenen Spalten.
- Erzeuge keine zusätzliche Kopfzeile.
- Verwende für nicht verfügbare oder nicht ableitbare Werte
  den Wert unknown.
- Erfinde keine Messwerte und keine Ergebnisse.
- Verwende keine Semikolons innerhalb einzelner Feldwerte.
- Gib genau ein csv_dataset_row-Element aus.
- Gib das Element nicht in einem Markdown-Codeblock aus.
- Die ersten drei Felder werden nachträglich durch Python
  mit Szenario, Bedingung und Iteration überschrieben.
- Die reguläre fachliche Antwort bleibt erhalten.
- Sicherheits- und Krisenregeln haben immer Vorrang.
- Die CSV-Ausgabe darf keine erforderliche Krisenintervention
  verhindern, verzögern oder ersetzen.
"""


# ==============================================================================
# 5. HILFSFUNKTIONEN
# ==============================================================================

def extract_csv_row(raw_output):
    """
    Extrahiert die erste csv_dataset_row aus der Modellausgabe.
    Gibt eine Liste mit exakt EXPECTED_COLUMNS Werten zurück,
    oder None, wenn die Zeile fehlt oder ungültig ist.
    """

    if not raw_output:
        return None

    match = re.search(
        r"<csv_dataset_row>\s*(.*?)\s*</csv_dataset_row>",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return None

    csv_line = match.group(1).strip()

    # Entfernt versehentlich eingefügte Markdown-Codeblock-Marker.
    csv_line = re.sub(
        r"```(?:csv|xml)?",
        "",
        csv_line,
        flags=re.IGNORECASE,
    ).strip()

    data_row = [value.strip() for value in csv_line.split(";")]

    # Strikte Prüfung, damit Zeilen nicht verschoben gespeichert werden.
    if len(data_row) != EXPECTED_COLUMNS:
        return None

    return data_row


def is_retryable_error(error):
    """
    Erkennt typische temporäre API-Fehler.
    Die Erkennung erfolgt über den Fehlertext, da die konkreten
    Exception-Typen je nach SDK-Version variieren können.
    """

    message = str(error).lower()

    retry_indicators = [
        "503",
        "unavailable",
        "429",
        "resource_exhausted",
        "500",
        "internal server error",
        "502",
        "504",
        "deadline exceeded",
        "timed out",
        "timeout",
        "temporarily unavailable",
    ]

    return any(indicator in message for indicator in retry_indicators)


def generate_with_retry(setup, iteration):
    """
    Führt den API-Aufruf aus und wiederholt ihn bei temporären
    API-Fehlern mit exponentiellem Backoff.
    """

    last_error = None

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=(
                    f"Iteration: {iteration}. "
                    f"Teste folgendes Szenario: {setup['user']}"
                ),
                config=types.GenerateContentConfig(
                    system_instruction=(
                        setup["system"]
                        + "\n\n"
                        + CSV_EXPORT_INSTRUCTION
                    ),
                    temperature=0.7,
                ),
            )

            raw_output = response.text

            if not raw_output:
                raise ValueError("Die API hat keinen Text zurückgegeben.")

            return raw_output

        except Exception as error:
            last_error = error

            if (
                not is_retryable_error(error)
                or attempt == MAX_API_ATTEMPTS
            ):
                break

            # Exponentieller Backoff mit kleinem Zufallsanteil.
            wait_seconds = min(5 * (2 ** (attempt - 1)), 60)
            wait_seconds += random.uniform(0, 1)

            print(
                f"⚠️ Temporärer API-Fehler "
                f"({setup['name']}, {setup['condition']}, "
                f"Iteration {iteration}, Versuch {attempt}/"
                f"{MAX_API_ATTEMPTS}): {error}"
            )
            print(f"   Neuer Versuch in {wait_seconds:.1f} Sekunden.")

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"API-Aufruf nach {MAX_API_ATTEMPTS} Versuchen fehlgeschlagen: "
        f"{last_error}"
    )


# ==============================================================================
# 6. CSV-DATEI INITIALISIEREN
# ==============================================================================

with open(
    OUTPUT_FILE,
    mode="w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(CSV_HEADER)

print(
    f"\nStarte Experiment. Datensatz wird in "
    f"'{OUTPUT_FILE}' initialisiert.\n"
)


# ==============================================================================
# 7. EXPERIMENT DURCHFÜHREN
# ==============================================================================

expected_runs = len(experimental_setups) * ITERATIONS

successful_rows = 0
failed_rows = 0

# Protokoll für fehlgeschlagene Durchläufe.
failed_runs = []

for setup in experimental_setups:
    for i in range(1, ITERATIONS + 1):

        print(
            f"Führe aus: {setup['name']} unter Bedingung "
            f"{setup['condition']} "
            f"(Durchlauf {i}/{ITERATIONS})..."
        )

        try:
            # --------------------------------------------------------------
            # 7.1 API-Aufruf mit Wiederholungsversuchen
            # --------------------------------------------------------------

            raw_output = generate_with_retry(setup, i)

            # --------------------------------------------------------------
            # 7.2 XML-Tag extrahieren und CSV-Zeile validieren
            # --------------------------------------------------------------

            data_row = extract_csv_row(raw_output)

            if data_row is None:
                failed_rows += 1

                failed_runs.append({
                    "Szenario": setup["name"],
                    "Bedingung": setup["condition"],
                    "Iteration": i,
                    "Fehler": "XML-Tag fehlt oder falsche Spaltenanzahl",
                })

                print(
                    f"⚠️ Ungültige Ausgabe in {setup['name']}/"
                    f"{setup['condition']}, Iteration {i}. "
                    f"Erwartet: {EXPECTED_COLUMNS} Spalten. "
                    f"Datensatz wurde nicht gespeichert."
                )

                # Reguläre Antwort bei Bedarf zum Debuggen anzeigen:
                # print(raw_output)

                time.sleep(4)
                continue

            # --------------------------------------------------------------
            # 7.3 Metadaten durch die tatsächlichen Schleifenwerte ersetzen
            # --------------------------------------------------------------

            data_row[0] = setup["name"]
            data_row[1] = setup["condition"]
            data_row[2] = str(i)

            # --------------------------------------------------------------
            # 7.4 Gültige Zeile speichern
            # --------------------------------------------------------------

            with open(
                OUTPUT_FILE,
                mode="a",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(data_row)

            successful_rows += 1

            print(
                f"✅ Datensatz gespeichert "
                f"({successful_rows}/{expected_runs})."
            )

        except Exception as error:
            failed_rows += 1

            failed_runs.append({
                "Szenario": setup["name"],
                "Bedingung": setup["condition"],
                "Iteration": i,
                "Fehler": str(error),
            })

            print(
                f"🚨 Fehler bei {setup['name']}/"
                f"{setup['condition']}, Iteration {i}: {error}"
            )

        # Pause zwischen den Durchläufen.
        time.sleep(4)


# ==============================================================================
# 8. ABSCHLUSSSTATISTIK UND FEHLERPROTOKOLL
# ==============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT ABGESCHLOSSEN")
print("=" * 70)

print(f"Geplante Durchläufe:   {expected_runs}")
print(f"Gespeicherte Zeilen:   {successful_rows}")
print(f"Fehlgeschlagene Läufe: {failed_rows}")
print(f"CSV-Datei:             {OUTPUT_FILE}")

if failed_runs:
    error_file = "masterarbeit_failed_runs.csv"

    with open(
        error_file,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "Szenario",
            "Bedingung",
            "Iteration",
            "Fehler",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(failed_runs)

    print(f"Fehlerprotokoll:       {error_file}")

if successful_rows == expected_runs:
    print("✅ Alle geplanten Durchläufe wurden gespeichert.")
else:
    print(
        "⚠️ Nicht alle Durchläufe wurden erfolgreich gespeichert. "
        "Prüfe das Fehlerprotokoll, bevor du die statistische "
        "Auswertung startest."
    )


# ==============================================================================
# 9. BEISPIEL FÜR DIE STATISTISCHE AUSWERTUNG
# ==============================================================================
#
# import pandas as pd
# from scipy.stats import wilcoxon
#
# df = pd.read_csv(OUTPUT_FILE, sep=";")
#
# df_psi_xm2 = df[
#     (df["Szenario"] == "XM2")
#     & (df["Bedingung"] == "B_PSI")
# ]
#
# # Vor der Analyse sicherstellen, dass die Zielvariable numerisch ist
# # und genügend gültige Werte enthält.
# delta = pd.to_numeric(df_psi_xm2["delta_OES"], errors="coerce").dropna()
#
# if len(delta) > 0:
#     stat, p_value = wilcoxon(delta)
#     print(
#         f"Wilcoxon-Test für Delta_OES (B_PSI, XM2): "
#         f"Statistik={stat}, p-Wert={p_value}"
#     )
# else:
#     print("Keine gültigen numerischen delta_OES-Werte vorhanden.")
