# Testování fungování hybridního systému pro extrakci

Způsob testování netrénovaných modelů a porovnání s referenčními hodnotami extrakce s ideálními či slepými piloty je viditelný v souboru `data_analysis_scripts/evaluating_model.ipynb`. Následující části popisují postup, jak případně spustit vlastní trénování v daném souboru.

## Úvodní nastavení
Pokud jste dosud nestáhli obsah repozitáře:
1. Stáněte obsah repozitáře bakalářské práce do vybaného adresáře.
2. Nainstalujte potřebné knihovny Pythonu, ideálně do vytvořeného virtuálního prostředí (například dle souboru **requirements.txt** ```pip install -r requirements.txt```).
3. Vytvořte soubor **data** na úrovni hlavního hlavního adresáře, do něj vytvořte soubor **test**, do kterého nakopírujte vygenerované soubory s nahrávkami vytvořenými dle návodu [generovani.md](./generovani.md) (soubory s*.wav a y*.wav).

Po úspěšné přípravě dat a instalaci potřebných knihoven pokračujte v souboru `data_analysis_scripts/evaluating_model.ipynb`:
1. Spusťte první buňku souboru s potřebnými importy, buňka po úspěšném proběhnutí bude ukazovat symvol *Success*.
2. V následující druhé buňce upravte počet nahrávek ```number_of_samples = 600```, dle Vámi vybraného počtu. Následně buňku spusťtě.

## Spočítání referenčních hodnot SIR extrakce
1. Spusťte buňku s definicí funkce ```get_SIR_for_basic_pilots(number_of_samples)```.
2. Spusťte výpočet referenčních hodnot pomocí ```SIR_df = get_SIR_for_basic_pilots(number_of_samples)```. Výsledkem je tabulka s hodnotami uložená v proměnné ```SIR_df```.
3. Tabulku uložte pomocí ```SIR_df.to_csv('../results/SIR_test.csv', index=True)```.

ALTERNATIVNĚ: Pokud jsou data již předpočítaná, tabulku ```SIR_df``` načtěte:
```
SIR_df = pd.read_csv('../results/SIR_test.csv', index_col=0)
```

## Spočítání hodnot SIR extrakce s natrénovanými modely
1. Spusťte buňku s definicí funkce ```get_improvement_for_hybrid_system(hybrid_system, number_of_samples)```.
2. Dle předpřipravených buněk inicializujte vybraný hybridní systém na základě vstupních příznaků. Jako argument inicializace hybridního systému slouží cesta k natrénovanému modelu. Např. pro systém s modelem ILD 14 + 17 + 35:
```
hybrid_system = HybridSystem_IPD14_IPD17_IPD35(model_path="./models/ipd14_ipd17_ipd35_model.pt")
```
3. Zavolejte příkaz pro výpočet hodnot SIR extrakce a zlepšení SIR pro testovací dataset:
```
SIR_imp_model_pilot, SIR_model_pilot = get_improvement_for_hybrid_system(hybrid_system, number_of_samples)
```
4. Uložte spočítané hodnoty do tabulky s referenčními hodnotami, např.:
```
SIR_df['SIR_ipd14_ipd17_ipd35_model_pilot'] = SIR_model_pilot
SIR_df['imp_SIR_ipd14_ipd17_ipd35_model_pilot'] = SIR_imp_model_pilot
```
5. Tabulku uložte pomocí ```SIR_df.to_csv('../results/SIR_test.csv', index=True)```.

## Vizualizace výsledků
Následující buňky obsahují vizualizaci spočítaných výsledků - přehled průměrů a krabicové grafy srovnání modelů a referenčních pilotů.
