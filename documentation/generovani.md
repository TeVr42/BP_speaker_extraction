# Generování syntetických dat
V rámci práce byla vytvořena vlastní databáze syntetických nahrávek řečníka v hlučném prostředí. Pro vlastní trénování a testování jednotlivých částí práce je nutné použít syntetické nahrávky v původní databáze nebo vytvořené obdobným způsobem, zde je postup pro vytvoření vlastních nahrávek.

Skripty pro následující úkony jsou obsaženy v adresáři **matlab_scripts**.

Syntetické nahrávky jsou tvořeny z databáze [Wall Street Journal dataset](https://doi.org/10.1109/ICASSP.2016.7471631), konkrétně z verze [MC-WSJ0-2mix](https://doi.org/10.1109/ICASSP.2018.8461639) ze sekce **cv** obsahující odhlučněné nahrávky 1 řečníka. Pro vlastní generování je nutné mít k dispozici tyto nahrávky nebo jejich obdobu.

## Příprava dat z WSJ0-2mix datasetu
Při využití MC-WSJ0-2mix nahrávek je před generováním dat nutné vytvořit adresář s obsahem originálních nahrávek pojmenovaných určeným způsobem.

1. Stáhněte soubor **wsj_data_clean.m**.
2. Vytvořte cílový adresář pro uložení předzpracovaných nahrávek (např. "wsj_data").
3. Připravte adresář s datasetem Wall Street Journal, tento adresář musí obsahovat soubory dvou mluvčích pojmenované **s1** a **s2**, přejmenujte je, pokud tomu tak není.
4. Spustě v terminálu Matlabu příkaz ```wsj_data_clean(baseFolder, destinationFolder)```:
  - ```baseFolder``` je proměnná typu string obsahující název cílového adresáře, např.  ```"./wsj_data"```, <br>
  - ```destinationFolder``` je proměnná typu string obsahující název adresáře s WSJ0-2mix datasetem, např.  ```"./wsj0-2mix/cv"```.
5. Pokud vše proběhlo v pořádku, cílový adresář nyní obsahuje předzpracované soubory typu .wav s 8 znakovými názvy (složené z písmen a číslic).

## Alternativa: Příprava dat s alternativním datasetem

**Altarnativně** je možné využít jiný dataset než WSJ0-2mix. Vlastní dataset musí obsahovat nahrávky samostatných řečníků z prostředí bez reverberací. Při použítí alternativního datasetu je nutné nejprve předpřipravit názvy těchto nahrávek.

1. Přejmenujte nahrávky dle následujících pravidel:
- jména nahrávek musí mít jednotnou délku, delší než 3 znaky, například 8 znaků,
- pokud 2 nahrávky pocházejí od stejného mluvčího, mají společné první 3 znaky, například *a12jk123* a *a12mo45s*,
- pokud 2 nahrávky pocházejí od různých mluvčích, první 3 znaky jsou rozdílné,
- soubory nesmí obsahovat více shodných nahrávek pod různými názvy.

2. Přejmenované nahrávky umístěte do nového adresáře, např.  ```"./wsj_data"```, který musí obsahovat pouze tyto nahrávky.

Takto připravený adresář s nahrávkami lze využít pro tvorbu nahrávek, dle následujícího postupu.

## Generování datasetu
Pro vygenerování konkrétních dat z přezpracovaných WSJ nahrávek.

1. Stáhněte soubor **generate_dataset.m**.
2. Z repozitáře https://github.com/ehabets/RIR-Generator stáhněte soubory **rir_generator.cpp** a **rir_generator_core.cpp**.
3. Spusťte kompilaci RIR generátoru pomocí příkazů:
 ```
mex -setup C++
mex rir_generator.cpp rir_generator_core.cpp
 ```
4. Pokud vše proběhlo v pořádku, pracovní adresář obsahuje soubor *rir_generator.mexw64*, pokud nastala chyba konzultujte s dokumentací daného [repozitáře](https://github.com/ehabets/RIR-Generator?tab=readme-ov-file).
5. Vytvořte cílový adresář pro uložení vygenerovaných nahrávek (např. "dataset_train").
6. Spusťte v Matlabu příkaz ```generate_dataset(nr_mix_samples, datafolder_name, wsj_folder) ```:<br>
  - ```nr_mix_samples``` je proměnná typu Integer obsahující počet požadovaných nahrávek, např.  ```100```, <br>
  - ```datafolder_name``` je proměnná typu string obsahující název cílového adresáře, např.  ```"./dataset_train"```, <br>
  - ```wsj_folder``` je proměnná typu string obsahující název adresáře s předzpracovanými WSJ nahrávkami, (viz návod Příprava dat), např.  ```"./wsj_data"```.
7. Pokud vše proběhlo v pořádku, cílový adresář nyní obsahuje stanovený počet vygenerovaných nahrávek hlavního řečníka v hlučném prostředí ve vybrané modelové situaci, konkrétně obsahuje následující nahrávky:
  - *s\*.wav* představující pouze nahrávku hlavního řečníka (složka SOI z příslušné mixované nahrávky),
  - *y\*.wav* představující nežádoucí hluk z mixované nehrávky (promluvy ostatních řečníků a šum),
  - *mix\*.wav* představující smíchané nahrávky hlavního řečníka v hlučném prostředí (= s*.wav + y*.wav).

