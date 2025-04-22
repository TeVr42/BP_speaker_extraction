# Trénování CNN modelu pro tvorbu pilotního signálu
Zde je popsán postup vlastního trénování modelu krok za krokem, jeho následné použítí je přiblíženo v souboru [testovani.md](./testovani.md).

## Příprava repozitáře
1. Stáněte obsah repozitáře bakalářské práce do vybaného adresáře.
2. Nainstalujte potřebné knihovny Pythonu, ideálně do vytvořeného virtuálního prostředí (například dle souboru **requirements.txt** ```pip install -r requirements.txt```).
3. Vytvořte soubor **data** na úrovni hlavního hlavního adresáře, do něj vytvořte soubor **train** (a případně **test**), do kterého nakopírujte vygenerované soubory s nahrávkami vytvořenými dle návodu [generovani.md](./generovani.md) (soubory s*.wav a y*.wav).

## Předzpracování dat
1. Vytvořte soubor **preprocessed_data** na úrovni hlavního hlavního adresáře se soubory práce.
1. Otevřete adresář **data_preprocessing_scripts**.

### Předzpracujte příznaky: 
1. Otevřete soubor **data_preprocessing_scripts/ILD_preprocessing.ipynb** pro předzpracování ILD příznaků nebo **data_preprocessing_scripts/ILD_preprocessing.ipynb** pro IPD příznaky.
2. Spusťte úvodní buňku s importy (1. buňka), pokud vše proběhlo v pořádku, buňka obsahuje znak Success.
3. Spusťte buňku s definicí funkce ```preprocess_data(number_of_data, name, mic1, mic2)```.
4. Spusťte vámi vybranou bunku pro předzpracování nebo vytvořte novou dle požadavků, jako parametry uveďte:
  - počet zpracovávaných dat (```number_of_data```, maximálně celkový počet vámi vygenerovaných dat),
  - požadovaný název nově vytvářených příznaků (```name```),
  - a čísla mikrofonů, ze kterých má být daný příznak generován (```mic1``` a ```mic2```, možnosti jsou v rozsahu 0-8 začínající horní řadou a končící dolní řadou mikrofonů).

### Předzpracujte piloty: 
1. Otevřete soubor **data_preprocessing_scripts/SIR_pilot_preprocessing.ipynb**
2. Spusťte úvodní buňku s importy a buňku s definicí funkce ```preprocess_data(number_of_data)```.
3. Spusťte buňku s příkazem preprocess_data pro vybraný počet zpracovávaných nahrávek (stejný počet jako u příznaků).

## Trénování sítě
1. Otevřete adresář **training_scripts** v něm soubor **model_training.ipynb**.
2. Spusťte buňku s importy (1. buňka), pokud vše proběhlo v pořádku, buňka obsahuje znak Success.
3. Spusťte buňku s nastavení seed (2. buňka).
4. Spusťte buňku s definováním funkce train_model (3. buňka).
5. Spusťte buňku s trénováním vybraného modelu nebo s trénováním s parametry dle Vašeho výběru, parametry jsou následující:
   - ```data_dir``` při dodržení výše popsané struktury se jedná o adresář ```"./preprocessed_data/"``` s předzpracovanými příznaky a piloty,
   - ```features_names``` list obsahující názvy příznaků (stringy), které slouží jako vstup do sítě při trénování (vytvořené při předzpracování dat, například ```["ipd17_", "ild17_]```),
   - ```model_name``` string s požadovaným názvem modelu,
   - ```epochs``` int obsahující požadovaný počet epoch (nepovinný, výchozí je 20),
   - ```batch_size``` int obsahující požadovanou velikost dávky při trénování (nepovinný, výchozí je 32),
   - ```learning_rate``` float obsahující požadovanou hodnotu learning ratu (nepovinný, výchozí je 0.001),
   - ```data_split_coeff``` float obsahující požadovaný koeficient rozdělení dat na trénovací a validační část, např. hodnota 0.7 označuje, že 70 % dat je použito jako trénovací, a 30 % jako validační (nepovinný, výchozí je 0.7),
   - Příklad:
```
train_model("../preprocessed_data/", ["ipd17_", "ild17_"], "ipd17_ild17_model_ii", epochs=12, batch_size=32, learning_rate=0.001, data_split_coeff=0.7)
```
6. Nechte dokončit trénování (na základě informačních výpisů je možné sledovat průběh).
7. Natrénovaný model je uložen ve složce **models**, (jeho použití popisuje návod trenovani.md).
8. Pro vizualizaci průběhu učení je možné zobrazit obsah adresáře **runs** vytvořeného nástrojem Tensorboard, příkaz konzole: ```tensorboard --logdir=runs```.
