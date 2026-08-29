# Notatka projektowa: projekt analiz DTF interbrain + HRV (H2, H4)
**Projekt:** SYNCC-IN hyperscanning | **Kontekst:** diady rodzic-dziecko (TD vs ASD, dzieci 3-6 lat), pasywne oglądanie 3 filmów  
**Status:** do dyskusji ze współpracownikami  
**Data:** 2026-08

---

## 1. Cel notatki

Zapis ustaleń z dyskusji nad projektem analizy skierowanej łączności mózg-mózg (interbrain DTF) w połączeniu z analizą HRV, dla hipotez H2 (temporo-parietal, uwaga społeczna) i H4 (autonomiczna ko-regulacja). Notatka porządkuje decyzje metodologiczne, otwarte pytania i zabezpieczenia. Wszystkie punkty są propozycjami do przedyskutowania, nie decyzjami ostatecznymi.

---

## 2. Wybrane hipotezy

**H2 - Social-attention / joint-attention coupling (temporo-parietal, fast band).**
Łączność fast-band nad temporo-parietal (P7/P8, przybliżenie TPJ) indeksuje alignment uwagi społecznej. Predykcja: silniejsze coupling caregiver->child podczas segmentów o wysokim ładunku społeczno-emocjonalnym; obniżone w ASD.

**H4 - Affective / arousal co-regulation (HRV, slow/autonomiczne).**
Interpersonalna synchronia HF-HRV indeksuje autonomiczną ko-regulację pobudzenia. Predykcja: caregiver->child prowadzi w ko-regulacji; zmienione w ASD.

**Uwaga:** H2 i H4 mają wspólną strukturę MVAR i różnią się tym, które krawędzie są primary. Dlatego projektowane są razem, w jednym modelu 4-zmiennym.

---

## 3. Ocena literaturowa (skrót)

Przegląd literatury (sierpień 2026) wspiera oba filary, z zastrzeżeniami:

- **TPJ / mentalizowanie / uwaga społeczna:** dobrze udokumentowane; obniżona interbrain synchronia w prawym TPJ w ASD w kontekstach naturalistycznych (Hirsch 2022, Quinones-Camacho 2021, Du 2024). Istnieje niemal bezpośredni precedens metodologiczny: synchronia mierzona na elektrodach temporo-parietalnych w theta/alpha/beta u nastolatkow z ASD, z niższą synchronią przy nasilonych trudnościach społecznych.
- **Kierunkowość caregiver->child:** precedens tą samą rodziną metod (Granger/PDC) w pasywnym paradygmacie - Kang et al. 2023 znaleźli dominującą kierunkowość parent->child. UWAGA: może być częściowo artefaktem dojrzewania (SNR, właściwości spektralne dziecka), nie tylko genuine scaffolding.
- **HF-HRV / ko-regulacja / ASD:** synchronia RSA słabsza w diadach z dzieckiem ASD (Wang 2021). WAŻNE ZASTRZEŻENIE: w ASD synchronia RSA bywa NEGATYWNA, a negatywna bywa ADAPTACYJNA (Moser 2026, lepsze umiejętności pragmatyczne). Predykcja grupowa nie może po prostu zakładać TD>ASD - znak synchronii może być bardziej informacyjny niż siła.
- **Pasmo:** literatura nie wskazuje jednoznacznie fast/alpha jako nośnika synchronii TPJ; theta/slow równie prawdopodobne. Fast/alpha jest bliżej UWAGI niż mentalizowania per se.

**Rekomendowane korekty sformułowania H2:** (a) rozdzielić uwagę od mentalizowania - fast/alpha indeksuje raczej uwagę; (b) nie traktować fast band jako jedynego kandydata; (c) kontrolować artefakt dojrzewania w asymetrii kierunkowej; (d) ostrożnie ze znakiem efektu HRV; (e) P7/P8 to grube skalpowe przybliżenie TPJ.

---

## 4. Kluczowa decyzja sygnałowa: obwiednie chwilowe

Praca na **obwiedniach chwilowych** (instantaneous amplitude) rytmów, nie na surowych przefiltrowanych sygnałach.

**Uzasadnienie (mocne, potwierdzone symulacyjnie):** metody fazowe (PLV, DTF na przefiltrowanym sygnale) załamują się, gdy dwie osoby mają różne częstotliwości centralne - faza rozjeżdża się systematycznie. Badanie symulacyjne (biorxiv 2025) pokazało to wprost dla różnic częstotliwości 6-14 Hz. Obwiednia (AEC-podobne) jest odporna, bo porównuje wolne fluktuacje mocy, nie chwilową fazę.

**Konsekwencja dla naszych danych:** dziecko fast ~8-10 Hz, dorosły ~10-12 Hz - phase-based DTF byłby systematycznie obciążony tą różnicą. Obwiednia to obchodzi. To jest empiryczne, nie tylko wygodne uzasadnienie.

**Zastrzeżenie interpretacyjne:** obwiednia + downsampling modeluje WOLNĄ koordynację amplitud (kto-kiedy-ma-silny-rytm w skali ~1-2 s), nie szybkie sprzężenie oscylacyjne. To trzeba jawnie przyjąć. Zgodne z hipotezami o współdzielonej uwadze/zaangażowaniu.

> **AKTUALIZACJA (implementacja Stage 2, patrz `pipeline_plan.md` §1/Stage 2):** powyższe dotyczy
> zmiennej EEG (`*:ROI`). Zmienna HRV **NIE** jest już obwiednią (patrz §5.1 niżej) — jest surowym
> (interpolowanym) IBI, tylko downsamplowanym. Konsekwencja: strona EEG pozostaje wielkością
> drugiego rzędu (obwiednia amplitudy szybkiego rytmu), a strona HRV jest teraz oscylacją
> pierwszego rzędu (samo IBI) — spójne wewnątrz każdej modalności, ale istotne przy interpretacji
> eksploracyjnych krawędzi cross brain-heart (§5.2).

---

## 5. Architektura modelu MVAR

### 5.1 Zmienne (model 4-zmienny, per diada, per film)

Wszystkie zmienne downsamplowane do wspólnej częstości (~2.5 Hz, Nyquist ~1.25 Hz — patrz
aktualizacja niżej):

| Zmienna | Opis |
|---------|------|
| child:P7P8 | obwiednia fast band dziecka (indywidualne pasmo z band_assignments.csv) |
| cg:P7P8 | obwiednia fast band caregivera (indywidualne pasmo) |
| child:HRV | surowe (interpolowane) IBI dziecka, tylko downsamplowane — bez filtracji HF, bez Hilberta |
| cg:HRV | surowe (interpolowane) IBI caregivera, tylko downsamplowane |

**WAŻNE (zaktualizowane w implementacji Stage 2, odwraca pierwotną decyzję poniżej):** HRV jako
**surowe IBI**, NIE obwiednia pasma HF, NIE RMSSD. Pierwotne uzasadnienie ("HRV jako obwiednia HF,
żeby była koncepcyjnie współmierna z obwiednią EEG") okazało się błędne po inspekcji realnych
sygnałów: obwiednie rytmów EEG fluktuują w paśmie, które **pokrywa się z surowym IBI** (RSA,
~0.2-1 Hz), podczas gdy obwiednia HF-IBI jest sygnałem drugiego rzędu, znacznie wolniejszym, który
już nie mieści się w tym paśmie. Podanie surowego IBI utrzymuje obie modalności w porównywalnym
paśmie przy wspólnej, niskiej częstości MVAR. Wiekowo-dostosowane pasmo HF (dzieci ~0.24-1.04 Hz,
dorośli ~0.15-0.40 Hz) jest zachowane wyłącznie jako metadane (`hf_reference` w atrybutach pliku
Stage 2) opisujące, gdzie powinna znajdować się treść RSA surowego IBI — nigdy nie służy do
filtrowania. Wariant z obwiednią HF pozostaje dostępny (`envelopes.hrv_hf_envelope`) jako opcja
porównawcza, ale nie jest już domyślną zmienną pipeline'u. Pełne uzasadnienie: `pipeline_plan.md`
§1 i Stage 2.

### 5.2 Krawędzie i przypisanie do hipotez

4 zmienne -> 6 par -> 12 krawędzi kierunkowych:

| Krawędź | Typ | Hipoteza |
|---------|-----|----------|
| cg:P7P8 -> child:P7P8 | interbrain CNS | **H2 primary** |
| child:P7P8 -> cg:P7P8 | interbrain CNS | H2 test asymetrii |
| cg:HRV -> child:HRV | interbrain ANS | **H4 primary** |
| child:HRV -> cg:HRV | interbrain ANS | H4 test asymetrii |
| cg:P7P8 <-> cg:HRV | intrabrain brain-heart | conditioning |
| child:P7P8 <-> child:HRV | intrabrain brain-heart | conditioning |
| cg:HRV -> child:P7P8 | cross brain-heart | eksploracyjne (autonomiczny scaffolding ANS->CNS) |
| child:HRV -> cg:P7P8 | cross brain-heart | eksploracyjne |

**Wartość modelu 4-zmiennego:** conditioning krzyżowy - ΔDTF cg->child w fast band estymowane z kontrolą na wspólną dynamikę autonomiczną, i odwrotnie. Mocniejszy test niż dwa osobne modele 2-zmienne. Plus dostęp do najbardziej nowatorskiej krawędzi: cg:HRV -> child:P7P8 (fizjologiczny scaffolding przez kanał ANS->CNS, niepokazany czysto w tej literaturze).

### 5.3 Estymacja

- **Per film** (żeby zachować film jako czynnik), z **uśrednianiem funkcji autokorelacji po oknach w obrębie 60-s filmu** (metoda Kaminskiego dla sDTF na poziomie okien - jedno okno = jedna realizacja procesu koordynacji). To daje stabilną estymację per film bez sklejania sygnałów i bez nieciągłości na granicach filmów.
- **Bayesowski MVAR z regularyzującym priorem** (horseshoe/regularized horseshoe albo Minnesota prior) na współczynnikach AR - stabilizuje estymację k=4 i pełni rolę, którą w podejściu klasycznym pełniłaby korekcja multiplicity.
- Sprawdzić rząd modelu p osobno dla EEG i HRV (różna struktura autokorelacji - w implementacji
  Stage 2 HRV to surowe IBI, a nie wygładzona obwiednia, więc założenie "HRV gładsza" nie jest już
  z góry pewne, trzeba zweryfikować empirycznie) przez AIC/BIC, zanim narzuci się wspólne p.

### 5.4 Zmienna zależna

**ΔDTF vs surogaty, ZAWSZE.** 2/4-zmienny DTF bez tego nie kontroluje wspólnego bodźca (oboje oglądają ten sam film). Surogatowe diady (obce pary oglądające ten sam film) usuwają komponentę wspólnobodźcową. Zależna liczona osobno per film.

Alternatywnie/dodatkowo z_vs_surrogate (znormalizowana pozycja w rozkładzie surogatów) - niezależna od bezwzględnej skali, dobra do testu asymetrii odpornego na artefakt dojrzewania.

---

## 6. Model grupowy (brms)

Dla każdej primary krawędzi, wspólny szkielet:

```r
delta_dtf ~ film * group + (1 | dyad_id) + (1 | child_id) + (1 | caregiver_id)
```

- `film`: trójpoziomowy czynnik WALENCYJNY (nie uporządkowana skala arousalu - patrz sekcja 7)
- `group`: TD / ASD
- family = student() (odporność na outliery)
- priory słabo informatywne; wnioskowanie: posterior probability + HDI (spójne z istniejącym podejściem DTF)

**Zaplanowany kontrast filmowy (kluczowy):** Incredibles vs (Peppa + Brave)/2 - "konflikt/wysoki arousal vs reszta", teoretycznie umotywowany wynikiem Esposito. Silniejszy niż pełny 3-poziomowy test.

```r
hypothesis(model, "filmIncredibles > (filmPeppa + filmBrave)/2")
```

### Predykcje testowalne

**H2:**
1. Intercept > 0 na ΔDTF (istnienie interbrain coupling ponad surogaty)
2. Kontrast: ΔDTF wyższe dla Incredibles (modulacja konfliktem)
3. group: ΔDTF niższe w ASD
4. Asymetria AI > 0 (caregiver prowadzi)

**H4:**
1. Intercept != 0 na ΔDTF HRV (ko-regulacja ponad przypadek)
2. group: zmieniona ko-regulacja w ASD - UWAGA NA ZNAK (nie zakładać po prostu TD>ASD)
3. Asymetria caregiver->child > child->caregiver

---

## 7. Bodźce: kontekst Esposito

Nasze bodźce to DOKŁADNIE te same sceny co w serii badań Esposito (Azhari, Durnford i wsp., fNIRS parent-child co-viewing).

### 7.1 Kategoryzacja (zapożyczona i zwalidowana przez Esposito)

- **Peppa Pig** - pozytywny/happy, niski-umiarkowany arousal
- **The Incredibles** - scena konfliktu, WYSOKI arousal
- **Brave** - pośredni/mieszany

Esposito kategoryzował głównie po WALENCJI (happy/angry/neutral), nie jako uporządkowaną skalę arousalu. Kluczowy fakt kotwiczący: w danych Esposito to Incredibles (konflikt) wyprodukował NAJSILNIEJSZY efekt interbrain (w medial left PFC).

### 7.2 Konsekwencja dla estymacji

Trzy filmy to trzy JAKOŚCIOWO RÓŻNE warunki (jeden happy, jeden konflikt, jeden pośredni), nie trzy próbki tego samego procesu. Dlatego:
- NIE uśredniać ACF po filmach (zgubiłoby efekt filmu - najciekawszy)
- Uśredniać ACF po OKNACH w obrębie filmu (zachowuje czynnik, daje stabilność)
- Film jako czynnik walencyjny + zaplanowany kontrast Incredibles vs reszta

### 7.3 KRYTYCZNE: relacja przestrzenna ROI

**Nasze P7/P8 i ROI Esposito są w dużej mierze ROZŁĄCZNE.** Esposito mierzył WYŁĄCZNIE PFC (fNIRS, klastry frontal/medial). Jego efekt był w medial left PFC (przód głowy). P7/P8 to temporo-parietal (tył-bok, TPJ). Dzieli je cały płat.

**Czego NIE można twierdzić:** że P7/P8 coupling replikuje efekt Esposito w tym samym regionie (oni nie mieli elektrod nad TPJ).

**Co można zapożyczyć:** (a) walidację i kategoryzację bodźców (własność filmów, niezależna od miejsca pomiaru); (b) obserwację, że scena konfliktu maksymalizuje interbrain coupling - ale jako HIPOTEZĘ do testu w TPJ, nie fakt.

**Implikacje:**
1. Frontal-midline (Fz) to nasz najbliższy odpowiednik Esposito - ale Fz ma słabą prevalence pików po ICA (utrudniona konceptualna replikacja; sam w sobie ciekawy wynik fNIRS-hemodynamika vs EEG-oscylacje).
2. H2 (temporo-parietal) jest analizą NOWĄ, nie replikacyjną - i to jej siła. Uzasadnienie z niezależnej literatury TPJ/ASD, nie z Esposito. Stoi na własnych nogach.
3. Możliwy ładny kontrast frontal (blisko Esposito) vs temporo-parietal (TPJ) - czy modulacja konfliktem jest specyficzna dla PFC czy uogólnia się na sieć TPJ. Esposito nie mógł tego zbadać (miał tylko PFC).

---

## 8. Rola HF-HRV dla H2 i H4

HF-HRV (~0.15-0.40 Hz dorośli, ~0.24-1.04 Hz dzieci; RSA, ton wagalny) - niezależny autonomiczny read-out regulacji i zaangażowania społecznego (rama polywagalna).

- **H4:** bezpośredni związek - HF-HRV to kanoniczny wskaźnik regulacji pobudzenia. Konwergencja caregiver-leading asymetrii w SLOW-band EEG i HF-HRV wzmocniłaby interpretację ko-regulacji.
- **H2:** umiarkowany związek - ton wagalny wspiera zaangażowanie społeczne; synchronia HF-HRV kontekstualizuje, czy coupling TPJ zachodzi na tle wspólnego zaangażowania autonomicznego.

Uwaga: pasmo HF wiekowo-dostosowane (dzieci oddychają szybciej - szersze/wyższe pasmo).

---

## 9. Multiplicity i struktura wnioskowania

Ustalona hierarchia (nie płaska rodzina, nie w pełni osobne testy):

- **Poziom krawędzi (wewnątrz hipotezy):** dla analiz z wieloma krawędziami - BH-FDR albo NBS. Dla modelu 4-zmiennego z jedną primary krawędzią per hipoteza - nie dotyczy bezpośrednio.
- **Poziom hipotezy:** każda hipoteza ma JEDEN nominowany główny kontrast (H2: cg->child fast ΔDTF efekt grupy; H4: analogicznie HRV).
- **Poziom rodziny:** łagodny FDR na głównych kontrastach primary hipotez (H2, H4, ewentualnie H1/H3/H5/H6). Koszt mocy minimalny (kilka liczb), zysk wiarygodności duży.
- **Podejście Bayesowskie (preferowane):** logika FWE/FDR zastąpiona przez shrinkage / partial pooling w modelu hierarchicznym. Regularyzujące priory na współczynnikach AR pełnią analogiczną rolę. Hipotezy = z góry określone kontrasty na posteriorze.
- **Krawędzie eksploracyjne** (cross brain-heart) - jawnie oznaczone jako hypothesis-generating, bez twardej korekcji.

Opcja: preregistracja z jawnym podziałem primary (H2, H4) vs secondary/exploratory.

---

## 10. Zabezpieczenia metodologiczne (checklist)

1. **ΔDTF vs surogaty zawsze** - kontrola wspólnego bodźca (niezbywalne dla modelu bez zewnętrznego conditioning).
2. **HRV jako surowe (interpolowane) IBI, nie RMSSD, nie (już) obwiednia HF** - zaktualizowano w
   implementacji Stage 2: surowe IBI dzieli pasmo częstotliwości z obwiednią EEG (RSA ~0.2-1 Hz),
   obwiednia HF-IBI już nie (patrz §4, §5.1). Wiekowo-dostosowane pasmo HF zachowane wyłącznie
   jako metadane referencyjne.
3. **Artefakt dojrzewania w asymetrii** - test czy asymetria utrzymuje się po normalizacji z_vs_surrogate (niezależnej od skali).
4. **Znak efektu HRV w ASD** - nie zakładać TD>ASD; negatywna synchronia bywa adaptacyjna.
5. **Pasmo HF wiekowo-dostosowane** osobno dziecko/dorosły.
6. **Rząd MVAR** sprawdzony osobno dla EEG i HRV przed narzuceniem wspólnego p.
7. **Diagnostyka MCMC** (Rhat < 1.01, ESS > 400, pp_check, LOO) - jeśli k=4 niestabilne, fallback do 2-zmiennego.

---

## 11. Otwarte pytania do dyskusji ze współpracownikami

1. **Model 4-zmienny (wspólny H2+H4, conditioning krzyżowy) vs dwa osobne 2-zmienne?** Rekomendacja: zacząć od 4-zmiennego z uśrednianiem ACF po oknach, fallback do 2-zmiennego jeśli diagnostyka MCMC pokaże niestabilność (60 s na film, 46 diad - trzeba zweryfikować empirycznie).
2. **Pozycjonowanie względem Esposito:** rozszerzenie paradygmatu na nowe regiony+modalność (EEG+HRV vs fNIRS PFC), czy niezależne badanie sieci TPJ korzystające z tych samych bodźców? Wpływa na wagę kontrastu frontal-vs-temporoparietal.
3. **Czy dodać frontal-midline (Fz) jako ROI porównawczy** dla konceptualnej bliskości z Esposito, mimo słabej jakości sygnału Fz?
4. **Slow band w H4:** czy testować HRV coupling łącznie ze slow-band EEG (kanał wolnej ko-regulacji), czy tylko HRV-HRV?
5. **Preregistracja** - czy i w jakim zakresie; podział primary/secondary.
6. **Fallback jeśli k=4 niestabilne** - czy 2-zmienny EEG-only + osobny 2-zmienny HRV-only, z połączeniem na poziomie modeli brms (korelacja interbrain EEG coupling z interbrain HRV synchrony)?

---

## 12. Powiązania z istniejącym pipeline

- Indywidualne pasma (slow_cf, fast_cf, bw) z `04_band_assignment/band_assignments.csv`.
- IAF distance jako kowariant z `04_band_assignment/iaf_metrics.csv`.
- Istniejące metody synchronii HRV (WCC, DTW, CRQA, Granger) z notatek - do wykorzystania w wariancie 2-zmiennym HRV-only i jako walidacja krzyżowa.
- Podejście surogatowe i ΔDTF / z_vs_surrogate z `DTF_statistical_analysis_notes.md`.

---

*Notatka do dyskusji. Wszystkie decyzje wstępne, wymagają konsultacji zespołowej przed implementacją.*
