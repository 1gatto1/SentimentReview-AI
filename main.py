# pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Dataset di recensioni 
# In questo codice sono presenti SOLO 10 esempi di recensioni, in un codice reale è consigliato un numero maggiore
dati_recensioni = {
    'testo': [
        'Questo prodotto è fantastico!',
        'Pessima qualità, si è rotto subito.',
        'Molto soddisfatto dell’acquisto.',
        'Non vale il prezzo pagato.',
        'Servizio clienti eccellente!',
        'Materiali scadenti, esperienza negativa.',
        'Prodotto eccezionale, consigliato!',
        'Assolutamente deludente.',
        'Ottimo rapporto qualità/prezzo!',
        'Tempo perso, non lo consiglio.'
    ],
    'etichetta': [ # Associa ogni recensione a positivo o negativo (primo valore = prima recensione
        'positivo', 'negativo', 'positivo', 'negativo', 'positivo',
        'negativo', 'positivo', 'negativo', 'positivo', 'negativo'
    ]
}

tabella = pd.DataFrame(dati_recensioni)

modello = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

# cross-validation per valutazione più realistica
accuracy = cross_val_score(modello, tabella['testo'], tabella['etichetta'], cv=5)
print(f"Accuratezza media CV: {accuracy.mean() * 100:.1f}%")

# addestramento finale
X_train, X_test, y_train, y_test = train_test_split(tabella['testo'], tabella['etichetta'], test_size=0.3, random_state=42)
modello.fit(X_train, y_train)

# valutazione
predizioni = modello.predict(X_test)
print(classification_report(y_test, predizioni))

# test su una nuova recensione
nuova_recensione = "Ci ha messo tantissimo ad arrivare e pure danneggiato. Mai più!"
print(f"Nuova recensione: '{nuova_recensione}' ➜ {modello.predict([nuova_recensione])[0]}")
