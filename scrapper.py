# scrapper pour remplir le dataset

import requests
from bs4 import BeautifulSoup
from google.colab import files


url = "https://www.podcastfrancaisfacile.com/textes?_niveau=delf-c1&_page=2"


response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')


articles = soup.find_all('a', href=lambda href: href and '/texte/' in href)

print(f"Nombre de liens d'articles trouvés: {len(articles)}")

texts = []


for article in articles:
    article_url = article.get('href')
    print(f"Traitement de l'URL de l'article: {article_url}")

    article_response = requests.get(article_url)
    article_soup = BeautifulSoup(article_response.content, 'html.parser')
    print(article_soup.prettify())
 
    content = article_soup.find('div', class_='entry-content')
    if content:

        article_text = content.get_text(separator='\n')

  
        start = article_text.find("🔹TEXTE")
        end = article_text.find("🔹VOCABULAIRE")

        if start != -1 and end != -1:
            extracted_text = article_text[start + len("🔹TEXTE"):end].strip()
            print(f"Texte extrait de l'article: {extracted_text[:100]}...")
            texts.append(extracted_text)
        else:
            print("Les balises '🔹TEXTE' ou '🔹VOCABULAIRE' sont introuvables dans cet article.")
    else:
        print("Aucun contenu trouvé pour cet article.")


with open('/content/A1.txt', 'w', encoding='utf-8') as file:
    for text in texts:
        file.write(text)
        file.write('\n\n')
