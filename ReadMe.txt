


LaTex - FORMATVORLAGE 
für Abschluss- und Seminararbeiten
am Lehrstuhl für Wirtschaftsinformatik
________________________________________________
________________________________________________



Ordner bzw. Dateien dieser Vorlage
________________________________________________

/Bilder => alle Bilder/Grafiken der Arbeit hier speichern

/Kapitel => je Kap. eine Kapitel-Tex-Datei, die Kap. dann hier speichern

/Literatur => enthält im Normafall nur "quellen.bib" (Stichwort: BibTex)

/Titel => enthält die Deckblatt-Tex-Datei (Titel,Autor usw.)

vorlage.tex => daraus wird das PDF-Dokument erstellt (Wurzeldokument)




Wurzeldokument - main.tex
________________________________________________

Die Unterscheidung, ob die Arbeit als Online-Dokument oder Druckversion generiert werden soll, kann in Zeile 1 in main.tex festgelegt werden. 'twoside=on' für Druckversion oder 'twoside=off' für Onlineversion.



Unterteilt in Packages und Document

1.Packages
In der Regel sind hier keine Änderungen nötig!
Die aufgeführten Packete müssen vorhanden/installiert sein,
damit das PDF fehlerfrei generiert werden kann. 
Um LaTex Packete zu installieren, die kostenlose Software "Miktex" nutzen.

2.Document
Hier werden die Elemente angegeben, aus denen das Dokument am Ende besteht.
Dazu gehören beispielsweise Titelblatt, die Kapitel, Abstract, 
Literaturverzeichnis, Inhaltsverzeichnis usw..
Zeilen, die mit einem %-Zeichen beginnen sind Kommentare und werden
vom LaTex-Interpreter nicht berücksichtigt.



LaTex Software
________________________________________________

Kollaborativer Online Editor - Overleaf

TeXworks - Textverarbeitung in LaTex
(erstellen der PDF Dokumente)

(optionale Software)
Miktex - um zusätzlich Pakete einzubinden
JabRef - Verwaltung des BibTex Literaturverzeichnisses


Zitieren in Latex
________________________________________________

Das natbib-Paket verfügt über zwei grundlegende Zitierbefehle, \citet und \citep für Text- bzw. Klammerzitate. Es gibt auch die mit Sternchen versehenen Versionen \citet* und \citep*, die die vollständige Autorenliste ausgeben und nicht nur die abgekürzte. Alle diese Varianten können ein oder zwei optionale Argumente enthalten, um Text vor und nach dem Zitat hinzuzufügen.

\citet{jon90}                   ⇒ Jones et al. (1990)
\citet[Kap.~2]{jon90}           ⇒ Jones et al. (1990, Kap. 2)
\citep{jon90}                   ⇒ (Jones et al., 1990)
\citep[Kap.~2]{jon90}           ⇒ (Jones et al., 1990, Kap. 2)
\citep[Siehe][]{jon90}          ⇒ (Siehe Jones et al., 1990)
\citep[Siehe][Kap.~2]{jon90}    ⇒ (Siehe Jones et al., 1990, Kap. 2)
\citet*{jon90}                  ⇒ Jones, Baker, and Williams (1990)
\citep*{jon90}                  ⇒ (Jones, Baker, and Williams, 1990)

Mehrere Zitate können durch Angabe von mehr als einem Zitierschlüssel im Argument des Befehls \cite erstellt werden:

\citet{jon90,jam91}     ⇒ Jones et al. (1990); James et al. (1991)
\citep{jon90,jam91}     ⇒ (Jones et al., 1990; James et al. 1991)
\citep{jon90,jon91}     ⇒ (Jones et al., 1990, 1991)
\citep{jon90a,jon90b}   ⇒ (Jones et al., 1990a,b)
