from script_automatisation import *

def update_data():

    base_actuelle = pd.read_csv("/home/onyxia/work/Projet-python/Fbref_alex/SOCCER_241223.csv")
    data = scrape_latest_ligue1_data()
    base_supplémentaire = data[0]

    base_updated = add_new_matches(base_initiale=base_actuelle, base_nouvelle=base_supplémentaire)

    #base_updated.to_csv("/home/onyxia/work/Projet-python/Fbref_alex/SOCCER_241223.csv")
    base_updated.to_csv("Fbref_alex/SOCCER_241223.csv")
    
    recup_matchweek = data[1]

    #recup_matchweek.to_csv("/home/onyxia/work/Projet-python/Fbref_alex/recup_matchweek.csv")
    recup_matcwheek.to_csv("Fbref_alex/SOCCER_241223.csv")

    return
