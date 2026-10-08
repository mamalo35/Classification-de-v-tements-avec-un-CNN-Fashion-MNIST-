# Classification de vêtements avec un CNN (Fashion-MNIST)

Réseau de neurones convolutif (CNN) qui reconnaît automatiquement le type de vêtement sur une image, parmi 10 catégories.
Le premier modèle atteint **90,8 % de précision** sur le jeu de test.

## Objectif

Construire, entraîner et évaluer un modèle de deep learning de bout en bout :
préparation des données, conception de l'architecture, entraînement, analyse des erreurs et amélioration du modèle.

## Données

Le projet utilise **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)**, un dataset publié par Zalando Research :

- 70 000 images en niveaux de gris de 28 × 28 pixels ;
- 60 000 images d'entraînement et 10 000 images de test ;
- 10 classes équilibrées.

| Étiquette | Classe     |
|-----------|------------|
| 0         | T-shirt    |
| 1         | Pantalon   |
| 2         | Pull       |
| 3         | Robe       |
| 4         | Manteau    |
| 5         | Sandale    |
| 6         | Chemise    |
| 7         | Basket     |
| 8         | Sac        |
| 9         | Bottine    |

### Préparation

- Normalisation des pixels entre 0 et 1 (division par 255) pour stabiliser l'entraînement.
- Ajout de la dimension « canal » : `(28, 28)` → `(28, 28, 1)`, format attendu par les couches `Conv2D`.

## Modèle 1 : CNN de base

### Architecture

| Couche                          | Sortie        |
|---------------------------------|---------------|
| Entrée                          | 28 × 28 × 1   |
| Conv2D (32 filtres 3×3, ReLU)   | 26 × 26 × 32  |
| MaxPooling 2×2                  | 13 × 13 × 32  |
| Conv2D (64 filtres 3×3, ReLU)   | 11 × 11 × 64  |
| MaxPooling 2×2                  | 5 × 5 × 64    |
| Flatten                         | 1 600         |
| Dense (64, ReLU) + Dropout 0,5  | 64            |
| Dense (10, softmax)             | 10            |

Environ 122 000 paramètres entraînables. Optimiseur Adam, perte `sparse_categorical_crossentropy`, 10 époques.

### Résultats

**Précision sur le jeu de test : 90,8 %**

Précision par classe (issue de la matrice de confusion) :

| Classe   | Bonnes réponses |
|----------|-----------------|
| Sandale  | 98,4 %          |
| Sac      | 97,9 %          |
| Basket   | 97,6 %          |
| Pantalon | 97,0 %          |
| Bottine  | 95,8 %          |
| Robe     | 91,2 %          |
| Manteau  | 87,3 %          |
| T-shirt  | 86,4 %          |
| Pull     | 84,9 %          |
| Chemise  | 71,5 %          |

<!-- Ajouter ici l'image de la matrice de confusion, par exemple : -->
<!-- ![Matrice de confusion du modèle 1](images/matrice_confusion_v1.png) -->

### Analyse des erreurs

Le point faible du modèle est la classe **Chemise** : seulement 71,5 % de bonnes réponses.
Elle est surtout confondue avec le T-shirt (126 erreurs), le manteau (72) et le pull (58).
À l'inverse, 98 T-shirts sont pris pour des chemises.

Ces vêtements ont des silhouettes très proches, et les détails qui les distinguent (col, boutons) sont difficiles à voir en 28 × 28 pixels.
Les chaussures, les sacs et les pantalons, aux formes très distinctes, sont quasiment toujours bien reconnus.

## Modèle 2 : CNN amélioré

L'objectif de cette version est de réduire les confusions entre les hauts (chemise, T-shirt, pull, manteau).

### Changements par rapport au modèle 1

- **Augmentation de données** (`RandomFlip`, `RandomTranslation`) : le modèle s'entraîne sur des images légèrement retournées ou décalées, pour apprendre des caractéristiques plus robustes.
- **Deux convolutions par bloc** avant chaque pooling : le modèle peut repérer des détails fins (col, boutons) avant que la résolution ne soit réduite.
- **`padding="same"`** : les bords de l'image sont conservés lors des convolutions.
- **`BatchNormalization`** : renormalise les valeurs entre les couches pour un apprentissage plus stable.
- **Dropout après chaque bloc** convolutif pour limiter le surapprentissage.
- **`EarlyStopping`** (patience de 5 époques, jusqu'à 50 époques) : l'entraînement s'arrête automatiquement quand le modèle ne progresse plus, et conserve sa meilleure version.

### Architecture

| Bloc         | Couches                                                              |
|--------------|----------------------------------------------------------------------|
| Augmentation | RandomFlip (horizontal), RandomTranslation (10 %)                    |
| Bloc 1       | Conv2D 32 → BatchNormalization → Conv2D 32 → MaxPooling → Dropout 0,25 |
| Bloc 2       | Conv2D 64 → BatchNormalization → Conv2D 64 → MaxPooling → Dropout 0,25 |
| Décision     | Flatten → Dense 128 → Dropout 0,5 → Dense 10 (softmax)               |

### Résultats

A compléter

## Technologies

- Python
- Keras (backend TensorFlow)
- NumPy
- Matplotlib
- Google Colab (entraînement sur GPU)
