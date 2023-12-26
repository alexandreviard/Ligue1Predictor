from script_automatisation import *



def update_data():

    mapping_equipe = {
        'Nimes': 'Nîmes',
        'Paris S-G': 'Paris Saint Germain',
        'Saint Etienne': 'Saint-Étienne'
    }
    
    base_actuelle = pd.read_csv("Fbref_alex/SOCCER_241223.csv", index_col=0)
    data = scrape_latest_ligue1_data()
    base_supplémentaire = data[0]

    base_updated = add_new_matches(base_initiale=base_actuelle, base_nouvelle=base_supplémentaire)

    #base_updated.to_csv("/home/onyxia/work/Projet-python/Fbref_alex/SOCCER_241223.csv")
    base_updated.to_csv("Fbref_alex/SOCCER_241223.csv")
    
    recup_matchweek = data[1]
    
    recup_matchweek=find_futur_matchweeks(recup_matchweek, mapping_equipe)
    #recup_matchweek.to_csv("/home/onyxia/work/Projet-python/Fbref_alex/recup_matchweek.csv")
    recup_matchweek.to_csv("Fbref_alex/recup_matchweek.csv")

    return

    
update_data()
