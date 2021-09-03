# Fighting Words in den Parteiprogrammen zur Bundestagswahl 2021
## Abgleich der Wahlprogramme von CDU/CSU, SPD, AfD, LINKE, FDP und BÜNDNIS 90/DIE GRÜNEN

Artikel zum Code: https://medium.com/@inojk/die-sprache-der-wahlprogramme-389ea4b414de

Die jeweiligen Wahlprogramme sind Eigentum der Parteien & wurden am 2.9.2021 abgefragt.
Die Implementierung der Fighting Words ist adaptiert von https://github.com/jmhessel/FightingWords

Run the code:

    git clone https://github.com/JonathanKuelz/election_programmes.git
    pip install requirements.txt
    python wahlprogramme.py --save_as your_folder_name -n 1 2 3  # Or any other n_gram length
    

## Beispiel: Wordcloud für CDU_CSU, 1_grams
![CDU_CSU_previes](/data/1_grams/cdu_csu.png)
