import pandas as pd
import statistics
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
# app.py
#!pip install flask gradio joblib pandas scikit-learn imblearn
import pandas as pd
import statistics
import gradio as gr
from flask import Flask, request, jsonify


lexique_path = '/Lexique383.tsv'
lexique = pd.read_csv(lexique_path, delimiter='\t')


def clean_text(text):
    return re.sub(r'[,\!\?\%\(\)\/\"]', '', text)


def count_syllables(word):  #estimation assez precise du nb de syllabes dans un mot
    voyelles = "aeiouyAEIOUY"
    n =len(word)

    if all(char in voyelles for char in word):
        return 1

    nb_syllabes= 0
    precendent_voyelle =False

    for i, char in enumerate(word):
        if char in voyelles:
            if i == 0 or not precedent_voyelle:
                nb_syllabes += 1
            precedent_voyelle = True
        else:
             precedent_voyelle= False

    return nb_syllabes


def get_text_features(text, lexique_dict):
    words = text.split()
    num_words = len(words)
    num_syllables = sum(count_syllables(word) for word in words)
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
    avg_syllables_per_word = num_syllables / num_words if num_words > 0 else 0

    # Ajoutez des caracteristiques
    
    freqs = [lexique_dict[word]['freqlemlivres'] if word in lexique_dict else 0 for word in words]
    avg_freq = sum(freqs) / num_words if num_words > 0 else 0
    if len(freqs) < 2:
        return avg_word_length, avg_syllables_per_word, avg_freq if freqs else 0, 0, 0, 0, 0
    else:
        quantiles = statistics.quantiles(freqs, n=5)
        return avg_word_length, avg_syllables_per_word, avg_freq, *quantiles

    return avg_word_length, avg_syllables_per_word, avg_freq,quantiles[0], quantiles[1], quantiles[2], quantiles[3]




lexique = lexique[['ortho', 'cgram', 'freqlemlivres']]


datasets = {}
levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
i=0
for level in levels:
    with open(f'/{level}.txt', 'r', encoding='utf-8') as file:
        datasets[level] = file.read().splitlines()

texts = []
labels = []
features = []

for level in levels:
    for text in datasets[level]:
        texts.append(text)
        labels.append(level)
        avg_word_length, avg_syllables_per_word,avg,q1,q2,q3,q4 = get_text_features(text,lexique)
        features.append([avg_word_length, avg_syllables_per_word,avg,q1,q2,q3,q4])   # on utilise les quantiles des frequences

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)


features = np.array(features)


X_combined = np.hstack((X_tfidf.toarray(), features))


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, labels)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)


model = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)  #mieux: n_est,max_depth=(500,50) ou (600,55) , Acc=0.54
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))




def transform_text_features(texts, vectorizer, get_text_features, lexique_dict):
 
    X_tfidf = vectorizer.transform(texts)

    features = np.array([get_text_features(text, lexique) for text in texts])

    X_combined = np.hstack((X_tfidf.toarray(), features))
    return X_combined

joblib.dump(model, 'random_forest_cefr_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(get_text_features, 'get_text_features.pkl')


# Fonction qui predit le niveau CECRL d'un nouveau texte
def predict_cefr_level(text):
    text_combined = transform_text_features([text], vectorizer, get_text_features, lexique)
    prediction = model.predict(text_combined)
    return prediction[0]

model = joblib.load('random_forest_cefr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
get_text_features=joblib.load('get_text_features.pkl')

def transform_text_features(texts, vectorizer, get_text_features, lexique_dict):
    X_tfidf = vectorizer.transform(texts)
    features = np.array([get_text_features(text, lexique_dict) for text in texts])
    X_combined = np.hstack((X_tfidf.toarray(), features))
    return X_combined

def predict_cefr_level(text):
    text_combined = transform_text_features([text], vectorizer, get_text_features, lexique)
    prediction = model.predict(text_combined)
    return prediction[0]

texte1 = "Bonjour comment vas tu?"
#"Le service propose une plateforme de contenus interactifs, ludiques et variés pour les élèves du CP à la Terminale"
#"Bonjour comment vas tu?"
#"Le réveillon du Nouvel An a toujours été ma journée préférée,” a dit l’Ingénieur, qui était venu se tenir avec eux"
#"Henriette était une femme petite et frêle, donc même un gardien ivre était une sécurité suffisante"

niveau_pred = predict_cefr_level(texte1)
print(f'Le niveau CECRL prédit pour le texte est: {niveau_pred}')




# Application 

app = Flask(__name__)

# Charger les fichiers de modèle
model = joblib.load('random_forest_cefr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
get_text_features = joblib.load('get_text_features.pkl')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    niveau_pred = predict_cefr_level(text)
    return jsonify({'niveau': niveau_pred})


def predict_and_display(text):
    niveau_pred = predict_cefr_level(text)
    return f'Le niveau CECRL prédit pour le texte est: {niveau_pred}'

# interface Gradio
iface = gr.Interface(
    fn=predict_and_display,
    inputs=gr.Textbox(lines=10, placeholder="Entrez le texte ici..."),
    outputs="text",
    title="Prédiction du Niveau CECRL",
    description="Entrez un texte en français pour prédire son niveau CECRL (A1, A2, B1, B2, C1, C2)."
)


def launch_gradio():
    iface.launch(share=True)

if __name__ == "__main__":
    from threading import Thread


    thread = Thread(target=launch_gradio)
    thread.start()


    app.run(debug=True, use_reloader=False)


'''
import pandas as pd
import statistics
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import re
from scipy.sparse import hstack

# Charger le lexique
lexique_path = '/Lexique383.tsv'
lexique = pd.read_csv(lexique_path, delimiter='\t')
lexique_dict = lexique.set_index('ortho').T.to_dict()

# Charger la liste des prénoms
prenoms_path = '/prenom.csv'
prenoms_df = pd.read_csv(prenoms_path)
prenoms_set = set(prenoms_df['prenom'].str.lower())

# Liste des principales villes de France
villes_france = [
    "paris", "marseille", "lyon", "toulouse", "nice", "nantes", "strasbourg",
    "montpellier", "bordeaux", "lille", "rennes", "reims", "le havre", "saint-étienne",
    "toulon", "grenoble", "dijon", "angers", "nîmes", "villeurbanne", "clermont-ferrand",
    "le mans", "aix-en-provence", "brest", "limoges", "tours", "amiens", "perpignan",
    "metz", "besançon", "boulogne-billancourt", "orléans", "mulhouse", "rouen", "caen",
    "nancy", "saint-denis", "argenteuil", "montreuil", "saint-paul", "nouméa"
]

# Fonction pour nettoyer le texte
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Fonction pour compter les syllabes
def count_syllables(word):
    voyelles = "aeiouyAEIOUY"
    n = len(word)
    if all(char in voyelles for char in word):
        return 1
    nb_syllabes = 0
    precedent_voyelle = False
    for i, char in enumerate(word):
        if char in voyelles:
            if i == 0 or not precedent_voyelle:
                nb_syllabes += 1
            precedent_voyelle = True
        else:
            precedent_voyelle = False
    return nb_syllabes

# Fonction pour filtrer les mots
def filter_words(text, lexique_dict, prenoms_set, villes_france):
    words = text.split()
    filtered_words = []
    removed_words = []
    zero_freq_words = []
    skip_next = False
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        
        # Filtrer les villes principales de France
        if word.lower() in villes_france:
            removed_words.append(word)
            continue
        
        # Ignorer les mots inconnus après certains préfixes
        if word.lower() in ["m.", "monsieur", "madame", "mme"] and (i + 1 < len(words)):
            next_word = words[i + 1]
            if next_word not in lexique_dict and next_word not in prenoms_set:
                removed_words.append(next_word)
                skip_next = True
                continue
        
        # Ignorer les mots inconnus adjacents aux prénoms s'ils sont suspects d'être des noms de famille
        if word in prenoms_set and (i + 1 < len(words)):
            next_word = words[i + 1]
            if next_word not in lexique_dict and next_word[0].isupper():
                removed_words.append(next_word)
                skip_next = True
                continue

        # Ajouter le mot s'il est dans le lexique ou s'il n'est pas un prénom
        if word in lexique_dict:
            if lexique_dict[word]['freqlemlivres'] == 0:
                zero_freq_words.append(word)
            filtered_words.append(word)
        elif word not in prenoms_set:
            filtered_words.append(word)
        else:
            removed_words.append(word)
    
    return ' '.join(filtered_words), removed_words, zero_freq_words

# Fonction pour extraire les caractéristiques textuelles
def get_text_features(text, lexique_dict):
    words = text.split()
    num_words = len(words)
    num_syllables = sum(count_syllables(word) for word in words)
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
    avg_syllables_per_word = num_syllables / num_words if num_words > 0 else 0

    # Calcul des fréquences et autres caractéristiques en considérant les mots inconnus comme rares
    freqs = [lexique_dict[word]['freqlemlivres'] if word in lexique_dict else 0 for word in words]
    avg_freq = sum(freqs) / num_words if num_words > 0 else 0

    num_verbs = sum(1 for word in words if word in lexique_dict and 'VER' in lexique_dict[word]['cgram'])
    num_nouns = sum(1 for word in words if word in lexique_dict and 'NOM' in lexique_dict[word]['cgram'])

    # Calcul du taux de répétition des mots
    unique_words = set(words)
    repetition_rate = 1 - (len(unique_words) / num_words) if num_words > 0 else 0

    if len(freqs) < 2:
        return avg_word_length, avg_syllables_per_word, avg_freq if freqs else 0, 0, 0, 0, 0,0,0,0,0,0, num_verbs, num_nouns, repetition_rate
    else:
        quantiles = statistics.quantiles(freqs, n=10)
        return avg_word_length, avg_syllables_per_word, avg_freq, *quantiles, num_verbs, num_nouns, repetition_rate

# Charger les datasets
datasets = {}
levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
for level in levels:
    with open(f'/{level}.txt', 'r', encoding='utf-8') as file:
        datasets[level] = file.read().splitlines()

# Extraire les textes et les labels
texts = []
labels = []
features = []
all_removed_words = []
all_zero_freq_words = []

for level in levels:
    for text in datasets[level]:
        filtered_text, removed_words, zero_freq_words = filter_words(text, lexique_dict, prenoms_set, villes_france)  # Filtrer les mots
        texts.append(filtered_text)
        labels.append(level)
        features.append(get_text_features(filtered_text, lexique_dict))
        all_removed_words.extend(removed_words)
        all_zero_freq_words.extend(zero_freq_words)

# Afficher les mots supprimés et les mots dont la fréquence est égale à 0
print(f'Mots supprimés : {set(all_removed_words)}')
print(f'Mots avec fréquence de 0 : {set(all_zero_freq_words)}')

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)

# Convertir les caractéristiques en array numpy
features = np.array(features)

# Combiner les caractéristiques TF-IDF avec les autres caractéristiques
X_combined = np.hstack((X_tfidf.toarray(), features))

# Utiliser SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, labels)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Optimiser les hyperparamètres du modèle RandomForest
param_grid = {
    'n_estimators': [700],
    'max_depth': [60],
    'min_samples_split': [3],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Entraîner le modèle RandomForest avec les meilleurs paramètres
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Sauvegarder les modèles et les transformateurs
joblib.dump(best_model, 'random_forest_cefr_model_optimized.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(get_text_features, 'get_text_features.pkl')

# Définir une correspondance numérique pour les niveaux CECRL
cefr_levels = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}

# Calculer les distances moyennes d'erreur pour chaque classe
error_distances = {level: [] for level in levels}

for i, (true_level, pred_level) in enumerate(zip(y_test, y_pred)):
    if true_level in cefr_levels and pred_level in cefr_levels:
        distance = abs(cefr_levels[pred_level] - cefr_levels[true_level])
        error_distances[true_level].append(distance)

# Calculer la distance moyenne d'erreur pour chaque classe
avg_error_distances = {level: np.mean(distances) if distances else 0 for level, distances in error_distances.items()}

# Afficher les distances moyennes d'erreur sous forme de tableau
error_table = pd.DataFrame.from_dict(avg_error_distances, orient='index', columns=['Average Error Distance'])
print(error_table)
'''





