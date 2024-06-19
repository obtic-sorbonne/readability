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


