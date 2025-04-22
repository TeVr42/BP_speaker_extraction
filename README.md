# 🎓 Repozitář k bakalářské práci

**Hybridní systém pro extrakci řečového signálu na základě informovaného algoritmu FICA**  
👩‍🎓 **Autor:** Tereza Vrbová  
👨‍🏫 **Vedoucí práce:** prof. Ing. Zbyněk Koldovský, Ph.D.  

## 🎧 Vyzkoušejte si
Základní ukázku fungování navrhovaného hybridního systému poskytuje soubor `example.ipynb`, ukazuje fungování extrakce na ukázkových směsích řečníka v hlučné oblasti ve vytipované situaci (`example_data`) vytvořených navrženým algoritmem pro tvorbu databáze.

📍 **Stáhněte si** repozitář a **spusťte** připravené kódy v souboru [example.ipynb](./example.ipynb), můžete si tak **poslechnout fungování hybridního systému** na neznámých datech a vidět **extrakci v praxi**.

## 📖 Dokumentace
📄 Teoretické přiblížení fungování jednotlivých kódu obsahuje text práce.

Chcete se s kódy seznámit **více dopodrobna**? Prostudujte dokumentaci.

Obecně se práce zabývá problémem **extrakce řečového signálu** hlavního řečníka stojícího ve vybrané oblasti v hlučném prostředí s jinými řečníky. Kódy obsahují:  
- ✅ Finální software pro extrakci,
- 📊 Skripty pro vyhodnocení extrakce a analýzu problémů,
- 🤖 Skripty pro trénování neuronových sítí vyvíjených v rámci práce.

Dokumentace přibližuje **strukturu adresářů** a fungování jednotlivých kódů, případně kroky pro jejich použití.  

📜 **Detailní dokumenty:**  
- 📌 **Ukázka použití hybridního systému** na ukázkových nahrávkách → [example.ipynb](./example.ipynb)  
- 📁 **Struktura adresářů** → [struktura.md](./dokumentace/struktura.md)  
- 🎤 **Generování vlastních uměle smíchaných nahrávek** → [generovani.md](./dokumentace/generovani.md)  
- 🤖 **Trénování neuronové sítě** pro zisk postranní informace → [trenovani.md](./dokumentace/trenovani.md)  
- 📊 **Testování hybridního systému** na připravených nahrávkách → [testovani.md](./dokumentace/testovani.md)  

## ⚙️ Technické specifikace  
💻 Kódy práce jsou napsány v **Pythonu (3.12.4)** 🐍 a **Matlabu (R2024b)** 📊.  
📦 **Použité Python knihovny** jsou uvedeny v souboru **requirements.txt**.  

## 🎯 Komu je repozitář určen?  
Tento repozitář je určen především čtenářům bakalářské práce, kteří se zajímají o její konkrétní realizaci, především tedy:  
👨‍🏫 **vedoucímu práce**,  
🧐 **oponentovi práce**,  
⚖️ **hodnotící komisi**,  
ale také **nadšencům do zpracování signálů** 📊.
