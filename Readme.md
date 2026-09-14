# Mini-Transformer from scratch (Projet 1)

Un GPT minimaliste codé de A à Z (tokenizer BPE, attention, pipeline de
données en streaming, entraînement, génération), piloté par une interface
Streamlit locale. Pensé pour être lisible, debuggable, et pour nourrir une
série d'articles LinkedIn sur "comment fonctionne un LLM".

## Contenu du projet

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit qui pilote tout le pipeline (corpus, entraînement, génération, visuels) |
| `download_corpus.py` | Télécharge un corpus Wikipédia FR en streaming (jamais chargé entier en RAM) |
| `prepare_data.py` | Entraîne le tokenizer BPE et tokenise le corpus en streaming vers `train.bin`/`val.bin` |
| `tokenizer.py` | Tokenizer BPE byte-level (librairie `tokenizers`) |
| `model.py` | Architecture Transformer décodeur, attention codée à la main |
| `train.py` | Boucle d'entraînement (lecture des `.bin` via `numpy.memmap`, reprise possible avec `--resume`) |
| `generate.py` | Génération de texte à partir d'un checkpoint |
| `plot_loss.py` | Courbe de loss (visuel pour LinkedIn) |
| `visualize_attention.py` | Heatmap d'attention (visuel pour LinkedIn) |

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install torch tokenizers datasets streamlit numpy matplotlib tqdm pandas
```

### Exploiter ton GPU Intel Arc (optionnel)

Par défaut, le script tourne sur CPU (ou CUDA/MPS si disponible). Pour
utiliser ton iGPU Arc via le backend XPU de PyTorch :

1. Installe `intel-extension-for-pytorch` en suivant la doc officielle
   à jour (la procédure change régulièrement selon les versions de
   PyTorch) : https://pytorch.org/docs/stable/notes/get_start_xpu.html
2. `train.py` (via `get_device()`) détecte automatiquement `torch.xpu`
   si disponible, sinon retombe sur CUDA, MPS, ou CPU.

Le CPU seul suffit largement pour ce Projet 1 (modèle de quelques
millions de paramètres).

## Utilisation

### Option A -- Interface Streamlit (recommandé)

```bash
streamlit run app.py
```

L'app propose 4 onglets :
- **📚 Corpus** : télécharger un extrait de Wikipédia FR en streaming (ou
  importer ton propre `.txt`), puis préparer le corpus (tokenizer BPE +
  tokenisation en streaming vers `train.bin`/`val.bin`).
- **🏋️ Entraînement** : configurer les hyperparamètres et lancer/arrêter
  l'entraînement (tourne en sous-processus, l'interface reste réactive),
  avec suivi live de la courbe de loss.
- **✍️ Génération** : tester un checkpoint entraîné (prompt, température,
  top-k, seed optionnel pour reproductibilité).
- **📊 Visuels LinkedIn** : générer la courbe de loss et la heatmap
  d'attention en un clic, avec bouton de téléchargement.

### Option B -- Ligne de commande

**1. Télécharger un corpus Wikipédia FR (streaming)**
```bash
python download_corpus.py --max_mb 500 --out corpus_wikipedia_fr.txt
```

**2. Préparer le corpus (tokenizer BPE + streaming vers .bin)**
```bash
python prepare_data.py --input corpus_wikipedia_fr.txt --out_dir data_prepared --vocab_size 8000
```

**3. Entraîner**
```bash
python train.py --data_dir data_prepared --steps 5000
```
Ajuste `--n_layer`, `--n_head`, `--n_embd`, `--block_size` pour changer
la taille du modèle. Avec les valeurs par défaut (4 couches, 128 dim),
tu es autour de 3M de paramètres. Utilise `--resume` pour reprendre un
entraînement interrompu depuis `--out_dir`.

**4. Générer du texte**
```bash
python generate.py --checkpoint_dir checkpoints --prompt "La France est" --tokens 300
```
Ajoute `--seed 42` pour un échantillonnage reproductible.

**5. Visuels pour LinkedIn**
```bash
python plot_loss.py --checkpoint_dir checkpoints
python visualize_attention.py --checkpoint_dir checkpoints --prompt "La France est un pays" --layer 0 --head 0
```

## Remplacer le corpus

Le pipeline part par défaut de Wikipédia FR (`wikimedia/wikipedia`,
config `20231101.fr`), téléchargée en streaming pour ne jamais saturer
la RAM ni le disque au-delà de la taille demandée (`--max_mb`). Tu peux
aussi importer n'importe quel texte brut UTF-8 (discours, corpus
littéraire libre de droits, articles que tu as écrits, etc.) via
l'onglet Corpus de l'app, ou en le passant directement à `prepare_data.py`
avec `--input`. Aucune autre modification n'est nécessaire : le tokenizer
BPE s'entraîne automatiquement sur le nouveau texte.

## Feuille de route articles LinkedIn (Projet 1)

1. **L'architecture** -- présentation du projet, pourquoi coder
   l'attention à la main plutôt qu'en appelant une fonction toute faite.
2. **Le pipeline de données à l'échelle** -- streaming Wikipédia +
   tokenizer BPE, comment traiter un corpus qui ne rentre pas en RAM.
3. **La boucle d'entraînement** -- captures d'écran de `plot_loss.py`
   (ou de l'onglet live de l'app), explication de la cross-entropy.
4. **La génération de texte et la visualisation de l'attention** --
   heatmap d'attention, effet de la température et du top-k.
5. **L'interface qui pilote tout ça** -- présentation de l'app
   Streamlit, du corpus au modèle entraîné sans ligne de commande.

## Prochaine étape : Projet 2

Une fois ce projet posté, on passe à un modèle plus gros (10-50M
paramètres) avec RoPE, RMSNorm, SwiGLU -- l'architecture type LLaMA.
