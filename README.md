# ASSR Phase Locking Factor (PLF) Analysis

Šis kodas skirtas analizuoti gyvūnų (pelių) smegenų atsaką į 
auditory steady-state stimulus (ASSR), vertinant Phase Locking Factor (PLF)
iš ECoG įrašų po vaistų injekcijų.

## Failai

- `heatmap.py` — PLF spektrogramų vizualizacija (2×3 grafikai)
- `statistics.py` — PLF statistinė analizė su boxplot ir post-hoc testais

## Duomenų struktūra

Kiekvienas `.txt` failas = vienas gyvūnas.  
Failai turi būti sudėti į atskirus aplankus pagal grupę:
## Konfigūracija

Prieš paleidžiant, `plf_heatmap.py` ir `plf_statistics.py` viršuje  
pakeisk `CONDITION_DIRS` kelius į savo duomenų aplanką:

```python
CONDITION_DIRS = {
    "Kontrolė":           r"/kelias/iki/SALINE_ASSR",
    "Ketaminas 20 mg/kg": r"/kelias/iki/KET20_ASSR",
    "MK-801 1 mg/kg":     r"/kelias/iki/MK1_ASSR",
}
```

## Paleidimas

```bash
pip install -r requirements.txt
python plf_heatmap.py
python plf_statistics.py
```

## Metodai

- Signalas: ECoG, 2000 Hz
- Epochos: 8s (16000 taškų)
- PLF skaičiuojamas 20–70 Hz diapazone naudojant STFT
- Lyginami du laiko langai: 0–30 min ir 60–90 min po injekcijos
- Statistika: Shapiro-Wilk normalumo testas → ANOVA arba Kruskal-Wallis
- Post-hoc: Tukey arba Dunn (Bonferroni korekcija)

## Grupės

| Grupė | Vaistas |
|-------|---------|
| Kontrolė | Saline |
| Ketaminas 20 mg/kg | Ketamine |
| MK-801 1 mg/kg | MK-801 |