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

    # Ajouter les caractéristiques lexiques
    freqs = [lexique_dict[word]['freqlemlivres'] if word in lexique_dict else 0 for word in words]
    avg_freq = sum(freqs) / num_words if num_words > 0 else 0
    if len(freqs) < 2:
        return avg_word_length, avg_syllables_per_word, avg_freq if freqs else 0, 0, 0, 0, 0
    else:
        quantiles = statistics.quantiles(freqs, n=5)
        return avg_word_length, avg_syllables_per_word, avg_freq, *quantiles

    return avg_word_length, avg_syllables_per_word, avg_freq,quantiles[0], quantiles[1], quantiles[2], quantiles[3]



# Donnees importantes
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
        features.append([avg_word_length, avg_syllables_per_word,avg,q1,q2,q3,q4])

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)

# Convertir les nouvelles caractéristiques en tableau numpy
features = np.array(features)


# Combiner les caractéristiques TF-IDF, et les caractéristiques lexicales
X_combined = np.hstack((X_tfidf.toarray(), features))

# Utiliser SMOTE pour équilibrer les classes (si nécessaire)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, labels)

# Séparer les données en ensemble d'entraînement et de test de manière stratifiée
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Construire le modèle Random Forest avec des hyperparamètres ajustés
model = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)  #mieux: n_est,max_depth=(500,50) ou (600,55) , Acc=0.54
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))



# Fonction pour transformer les caractéristiques du texte
def transform_text_features(texts, vectorizer, get_text_features, lexique_dict):
    # Vectoriser les textes avec TF-IDF
    X_tfidf = vectorizer.transform(texts)
    # Convertir les nouvelles caractéristiques en tableau numpy
    features = np.array([get_text_features(text, lexique) for text in texts])
    # Combiner les caractéristiques TF-IDF et les nouvelles caractéristiques
    X_combined = np.hstack((X_tfidf.toarray(), features))
    return X_combined

joblib.dump(model, 'random_forest_cefr_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(get_text_features, 'get_text_features.pkl')


# Fonction pour prédire le niveau CECRL d'un nouveau texte
def predict_cefr_level(text):
    text_combined = transform_text_features([text], vectorizer, get_text_features, lexique)
    prediction = model.predict(text_combined)
    return prediction[0]

# tests
texte1 = "Rappelons que les devoirs à la maison posent rarement des problèmes aux bons élèves. Mais pour ceux qui ont des difficultés, c’est une tout autre affaire : les devoirs sont donc un révélateur des inégalités. Que dire d’un enfant qui est face à un exercice qu’il n’a pas réussi à faire en classe ? Qu’il va y arriver quand il sera chez lui ? Ça revient à mettre sur ses épaules une pression énorme, d’autant plus que les élèves ont déjà des interrogations et des contrôles à tout-va… Même les petits de maternelle sont évalués ! Quand les enseignants estiment qu’un exercice peut se faire en un quart d’heure, ce ne sera pas le cas pour les élèves qui ont du mal."
texte2="Bonjour, tu vas bien?"

niveau_pred = predict_cefr_level(texte1)
print(f'Le niveau CECRL prédit pour le texte est: {niveau_pred}')
niveau_pred2 = predict_cefr_level(texte2)
print(f'Le niveau CECRL prédit pour le texte est: {niveau_pred2}')


