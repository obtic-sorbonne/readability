# scrapper pour remplir le dataset

import requests
from bs4 import BeautifulSoup
from google.colab import files

# URL de la page contenant les liens vers les articles pour le niveau A1
url = "https://www.podcastfrancaisfacile.com/textes?_niveau=delf-c1&_page=2"

# Faire une requête GET à la page principale
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Trouver tous les liens vers les articles contenant '/texte/' dans leur href
articles = soup.find_all('a', href=lambda href: href and '/texte/' in href)

print(f"Nombre de liens d'articles trouvés: {len(articles)}")

texts = []

# Pour chaque lien, faire une requête GET et extraire le texte
for article in articles:
    article_url = article.get('href')
    print(f"Traitement de l'URL de l'article: {article_url}")

    article_response = requests.get(article_url)
    article_soup = BeautifulSoup(article_response.content, 'html.parser')
    print(article_soup.prettify())
    # Extraire le contenu de l'article
    content = article_soup.find('div', class_='entry-content')
    if content:
        # Convertir le contenu en texte brut
        article_text = content.get_text(separator='\n')

        # Extraire le texte entre "🔹TEXTE" et "🔹VOCABULAIRE"
        start = article_text.find("🔹TEXTE")
        end = article_text.find("🔹VOCABULAIRE")

        if start != -1 and end != -1:
            extracted_text = article_text[start + len("🔹TEXTE"):end].strip()
            print(f"Texte extrait de l'article: {extracted_text[:100]}...")  # Afficher un extrait pour vérification
            texts.append(extracted_text)
        else:
            print("Les balises '🔹TEXTE' ou '🔹VOCABULAIRE' sont introuvables dans cet article.")
    else:
        print("Aucun contenu trouvé pour cet article.")

# Enregistrer tous les textes dans un fichier texte
with open('/content/A1.txt', 'w', encoding='utf-8') as file:
    for text in texts:
        file.write(text)
        file.write('\n\n')
