"""
Interface graphique locale pour le projet Mini-Transformer.

Permet de :
- télécharger un corpus Wikipédia FR (streaming) ou importer son propre texte
- préparer le corpus (tokenizer BPE + streaming vers train.bin/val.bin)
- régler les hyperparamètres et lancer/arrêter un entraînement
- suivre la courbe de loss en direct
- tester le modèle entraîné (génération de texte)
- générer les visuels pour LinkedIn (courbe de loss, heatmap d'attention)

Usage :
    streamlit run app.py

L'entraînement tourne dans un processus séparé (train.py lancé en
sous-processus) : l'interface reste réactive pendant que le modèle
s'entraîne, et tu peux naviguer entre les onglets ou arrêter
l'entraînement à tout moment.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).parent
PYTHON = sys.executable

# Force tous les sous-processus (train.py, generate.py, etc.) à utiliser
# l'UTF-8 pour leur entrée/sortie, indépendamment de l'encodage par défaut
# de la console Windows (cp1252) -- sinon un simple print() d'un caractère
# accentué "exotique" (nom propre étranger généré par le modèle, etc.)
# peut faire planter le sous-processus.
import os
SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

st.set_page_config(page_title="Mini-Transformer Lab", layout="wide")

# ----------------------------------------------------------------------
# État partagé entre les reruns Streamlit
# ----------------------------------------------------------------------
if "train_proc" not in st.session_state:
    st.session_state.train_proc = None
if "train_out_dir" not in st.session_state:
    st.session_state.train_out_dir = None
if "train_log_path" not in st.session_state:
    st.session_state.train_log_path = None

st.title("🧠 Mini-Transformer Lab")
st.caption("Wikipédia FR · Tokenizer BPE · Entraînement · Génération — pilotage local de ton mini-GPT from scratch")

tab_corpus, tab_train, tab_generate, tab_viz = st.tabs(
    ["📚 Corpus", "🏋️ Entraînement", "✍️ Génération", "📊 Visuels LinkedIn"]
)

# ----------------------------------------------------------------------
# Onglet Corpus
# ----------------------------------------------------------------------
with tab_corpus:
    st.subheader("🌊 Télécharger un corpus Wikipédia FR (streaming)")
    st.caption(
        "Le dataset Wikipédia FR fait plusieurs dizaines de Go sur Hugging Face -- on ne "
        "télécharge JAMAIS le dataset entier : le streaming écrit les articles sur disque "
        "au fil de l'eau, en s'arrêtant dès que la taille cible est atteinte."
    )

    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        max_mb = st.number_input("Taille cible (Mo)", min_value=10, max_value=50000, value=500, step=50)
    with col_dl2:
        corpus_name = st.text_input("Nom du fichier de sortie", value="corpus_wikipedia_fr.txt")

    if st.button("⬇️ Télécharger en streaming", type="primary"):
        with st.spinner("Streaming depuis Wikipédia FR... (peut prendre du temps selon la taille cible)"):
            result = subprocess.run(
                [PYTHON, str(PROJECT_DIR / "download_corpus.py"),
                 "--max_mb", str(max_mb), "--out", corpus_name],
                cwd=PROJECT_DIR, capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=SUBPROCESS_ENV,
            )
        if result.returncode == 0:
            st.success(result.stdout.strip().splitlines()[-2] if result.stdout else "Téléchargement terminé.")
        else:
            st.error("Échec du téléchargement :")
            st.code(result.stderr or result.stdout)

    st.divider()
    st.subheader("Ou importer ton propre texte")
    uploaded = st.file_uploader("Fichier .txt (UTF-8)", type=["txt"])
    if uploaded is not None:
        custom_name = st.text_input("Nom à donner au fichier", value=uploaded.name)
        if st.button("💾 Enregistrer ce fichier comme corpus"):
            content = uploaded.read().decode("utf-8", errors="replace")
            (PROJECT_DIR / custom_name).write_text(content, encoding="utf-8")
            st.success(f"Enregistré : {custom_name}")

    st.divider()
    st.subheader("Corpus disponibles dans le projet")
    txt_files_all = sorted(PROJECT_DIR.glob("*.txt"))
    if not txt_files_all:
        st.info("Aucun fichier .txt trouvé pour l'instant.")
    else:
        rows = [{"Fichier": f.name, "Taille": f"{f.stat().st_size / 1024:,.0f} Ko"} for f in txt_files_all]
        st.table(rows)

    st.divider()
    st.subheader("⚙️ Préparer le corpus (tokenizer BPE + streaming vers train.bin/val.bin)")
    if not txt_files_all:
        st.info("Télécharge ou importe d'abord un fichier .txt ci-dessus.")
    else:
        col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
        with col_p1:
            prep_input = st.selectbox("Fichier source", options=[f.name for f in txt_files_all], key="prep_input")
        with col_p2:
            prep_vocab = st.number_input("Taille du vocabulaire BPE", min_value=500, max_value=32000, value=8000, step=500)
        with col_p3:
            prep_out = st.text_input("Dossier de sortie", value="data_prepared")

        if st.button("⚙️ Préparer (peut prendre plusieurs minutes sur un gros fichier)"):
            with st.spinner("Entraînement du tokenizer BPE puis tokenisation en streaming..."):
                result = subprocess.run(
                    [PYTHON, str(PROJECT_DIR / "prepare_data.py"),
                     "--input", prep_input, "--out_dir", prep_out, "--vocab_size", str(prep_vocab)],
                    cwd=PROJECT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace", env=SUBPROCESS_ENV,
                )
            if result.returncode == 0:
                st.success(f"Corpus préparé dans `{prep_out}/` — utilise-le dans l'onglet Entraînement.")
                st.code(result.stdout[-1500:])
            else:
                st.error("Échec de la préparation :")
                st.code(result.stderr or result.stdout)

    prepared_dirs = sorted(p.parent.name for p in PROJECT_DIR.glob("*/meta.json"))
    if prepared_dirs:
        st.markdown("**Corpus préparés disponibles**")
        rows = []
        for d in prepared_dirs:
            meta_path = PROJECT_DIR / d / "meta.json"
            try:
                meta = json.loads(meta_path.read_text())
                rows.append({
                    "Dossier": d,
                    "Vocab BPE": meta.get("vocab_size"),
                    "Tokens train": f"{meta.get('n_train_tokens', 0):,}",
                    "Tokens val": f"{meta.get('n_val_tokens', 0):,}",
                })
            except (json.JSONDecodeError, OSError):
                continue
        if rows:
            st.table(rows)

# ----------------------------------------------------------------------
# Onglet Entraînement
# ----------------------------------------------------------------------
with tab_train:
    st.subheader("Configuration de l'entraînement")

    prepared_dirs = sorted(p.parent.name for p in PROJECT_DIR.glob("*/meta.json"))
    is_running = st.session_state.train_proc is not None and st.session_state.train_proc.poll() is None

    if is_running:
        st.warning(f"Un entraînement est en cours (checkpoint : `{st.session_state.train_out_dir}`). "
                   "Arrête-le avant d'en lancer un nouveau.")

    if not prepared_dirs:
        st.warning("Aucun corpus préparé trouvé. Va dans l'onglet 📚 Corpus pour en préparer un.")

    with st.form("train_config"):
        col1, col2 = st.columns(2)

        with col1:
            data_dir_choice = st.selectbox("Corpus préparé", options=prepared_dirs, disabled=is_running)
            if "out_dir_suggestion" not in st.session_state:
                st.session_state.out_dir_suggestion = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir = st.text_input(
                "Dossier de checkpoint",
                value=st.session_state.out_dir_suggestion,
                key="out_dir_input",
                disabled=is_running,
                help="Un nom unique par run évite d'écraser un ancien entraînement avec "
                     "d'autres hyperparamètres. Change-le si tu veux reprendre un dossier existant.",
            )
            steps = st.number_input("Nombre de steps", min_value=100, max_value=100000, value=3000, step=100, disabled=is_running)
            eval_interval = st.number_input("Intervalle d'évaluation", min_value=1, max_value=2000, value=200, step=10, disabled=is_running, help="Nombre de steps entre deux évaluations. Mettre 1 évalue à chaque step (très lent).")
            lr = st.select_slider(
                "Learning rate",
                options=[1e-4, 3e-4, 6e-4, 1e-3, 3e-3],
                value=3e-4,
                format_func=lambda x: f"{x:.0e}",
                disabled=is_running,
            )

        with col2:
            n_layer = st.slider("Nombre de couches (n_layer)", 1, 12, 4, disabled=is_running)
            n_head = st.slider("Nombre de têtes d'attention (n_head)", 1, 12, 4, disabled=is_running)
            n_embd = st.slider("Dimension d'embedding (n_embd)", 32, 512, 128, step=32, disabled=is_running)
            block_size = st.slider("Longueur de contexte (block_size)", 32, 512, 256, step=32, disabled=is_running)
            batch_size = st.slider("Batch size", 8, 128, 64, step=8, disabled=is_running)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1, step=0.05, disabled=is_running)

        if n_embd % n_head != 0:
            st.error(f"n_embd ({n_embd}) doit être divisible par n_head ({n_head}).")
            valid_config = False
        else:
            valid_config = True

        submitted = st.form_submit_button("🚀 Lancer l'entraînement", type="primary", disabled=is_running)

    if submitted and valid_config and not is_running:
        log_path = PROJECT_DIR / f"{out_dir}_train.log"
        cmd = [
            PYTHON, str(PROJECT_DIR / "train.py"),
            "--data_dir", data_dir_choice,
            "--out_dir", out_dir,
            "--steps", str(steps),
            "--eval_interval", str(eval_interval),
            "--lr", str(lr),
            "--n_layer", str(n_layer),
            "--n_head", str(n_head),
            "--n_embd", str(n_embd),
            "--block_size", str(block_size),
            "--batch_size", str(batch_size),
            "--dropout", str(dropout),
            "--no_live_plot",  # la fenêtre matplotlib live n'a pas de sens depuis un sous-processus lancé par Streamlit
        ]
        st.info("Commande lancée :\n" + " ".join(cmd))
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write("Commande : " + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=log_file, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=SUBPROCESS_ENV)
        st.session_state.train_proc = proc
        st.session_state.train_out_dir = out_dir
        st.session_state.train_log_path = str(log_path)
        # on prépare un nouveau nom suggéré pour le PROCHAIN run, pour ne pas
        # réutiliser accidentellement le même dossier deux fois de suite
        st.session_state.out_dir_suggestion = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        del st.session_state["out_dir_input"]
        st.rerun()

    if is_running:
        if st.button("⏹️ Arrêter l'entraînement"):
            proc = st.session_state.train_proc
            proc.terminate()
            try:
                proc.wait(timeout=5)  # on attend que le process soit VRAIMENT mort
            except subprocess.TimeoutExpired:
                proc.kill()  # il ignore SIGTERM -> on force
                proc.wait(timeout=5)
            st.session_state.train_proc = None
            st.rerun()

    st.divider()

    @st.fragment(run_every="2s" if is_running else None)
    def show_training_progress():
        out_dir_current = st.session_state.train_out_dir
        if out_dir_current is None:
            st.info("Aucun entraînement lancé pour l'instant.")
            return

        config_path = PROJECT_DIR / out_dir_current / "config.json"
        if config_path.exists():
            try:
                applied_config = json.loads(config_path.read_text())
                with st.expander("Hyperparamètres réellement appliqués (lus depuis config.json)", expanded=False):
                    st.json(applied_config)
            except json.JSONDecodeError:
                pass  # fichier en cours d'écriture, on retentera au prochain refresh

        history_path = PROJECT_DIR / out_dir_current / "history.json"
        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.markdown("**Courbe de loss**")
            if history_path.exists():
                try:
                    history = json.loads(history_path.read_text())
                    if history:
                        chart_data = {
                            "step": [h["step"] for h in history],
                            "train_loss": [h["train_loss"] for h in history],
                            "val_loss": [h["val_loss"] for h in history],
                        }
                        import pandas as pd
                        df = pd.DataFrame(chart_data).set_index("step")
                        st.line_chart(df)
                        last = history[-1]
                        st.metric("Dernière loss (train)", f"{last['train_loss']:.4f}")
                        st.metric("Dernière loss (val)", f"{last['val_loss']:.4f}")
                except (json.JSONDecodeError, KeyError):
                    st.info("En attente des premières données...")
            else:
                st.info("En attente des premières données...")

        with col_b:
            st.markdown("**Sortie console**")
            log_path_current = st.session_state.train_log_path
            if log_path_current and Path(log_path_current).exists():
                log_text = Path(log_path_current).read_text(encoding="utf-8", errors="replace")
                tail = "\n".join(log_text.strip().splitlines()[-15:])
                st.code(tail or "(pas encore de sortie)", language=None)

        proc = st.session_state.train_proc
        if proc is not None and proc.poll() is not None:
            st.success(f"Entraînement terminé (code retour {proc.poll()}).")

    show_training_progress()

# ----------------------------------------------------------------------
# Onglet Génération
# ----------------------------------------------------------------------
with tab_generate:
    st.subheader("Tester un modèle entraîné")

    checkpoint_dirs = sorted({p.parent.name for p in PROJECT_DIR.glob("*/model.pt")})

    if not checkpoint_dirs:
        st.info("Aucun checkpoint trouvé. Lance d'abord un entraînement dans l'onglet précédent.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            ckpt_dir = st.selectbox("Checkpoint", options=checkpoint_dirs)
            prompt = st.text_area("Prompt de départ", value="\n", height=100)
            n_tokens = st.slider("Nombre de tokens à générer", 20, 1000, 300, step=20)
        with col2:
            temperature = st.slider("Température", 0.1, 2.0, 0.8, step=0.1)
            top_k = st.slider("Top-k", 1, 100, 40, step=1)
            use_seed = st.checkbox(
                "Fixer le seed (comparaison reproductible)",
                help="Sans seed, chaque génération est différente même à paramètres identiques "
                     "(échantillonnage aléatoire). Fixe un seed pour isoler l'effet d'un seul "
                     "paramètre -- ex: comparer deux températures sur exactement le même tirage.",
            )
            seed = st.number_input("Seed", min_value=0, value=42, step=1, disabled=not use_seed)

        if st.button("✍️ Générer", type="primary"):
            with st.spinner("Génération en cours..."):
                cmd = [
                    PYTHON, str(PROJECT_DIR / "generate.py"),
                    "--checkpoint_dir", ckpt_dir,
                    "--prompt", prompt,
                    "--tokens", str(n_tokens),
                    "--temperature", str(temperature),
                    "--top_k", str(top_k),
                ]
                if use_seed:
                    cmd += ["--seed", str(seed)]
                result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace", env=SUBPROCESS_ENV)
            if result.returncode == 0:
                # la première ligne de generate.py est le log d'init du modèle, on l'enlève
                output_lines = result.stdout.splitlines()
                text_out = "\n".join(output_lines[1:]) if len(output_lines) > 1 else result.stdout
                st.text_area("Résultat", value=text_out, height=300)
            else:
                st.error("Erreur lors de la génération :")
                st.code(result.stderr or result.stdout)

# ----------------------------------------------------------------------
# Onglet Visuels LinkedIn
# ----------------------------------------------------------------------
with tab_viz:
    st.subheader("Générer les visuels pour tes posts")

    checkpoint_dirs = sorted({p.parent.name for p in PROJECT_DIR.glob("*/model.pt")})

    if not checkpoint_dirs:
        st.info("Aucun checkpoint trouvé. Lance d'abord un entraînement.")
    else:
        ckpt_dir = st.selectbox("Checkpoint", options=checkpoint_dirs, key="viz_ckpt")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Courbe de loss**")
            if st.button("Générer la courbe de loss"):
                out_png = PROJECT_DIR / f"{ckpt_dir}_loss_curve.png"
                result = subprocess.run(
                    [PYTHON, str(PROJECT_DIR / "plot_loss.py"),
                     "--checkpoint_dir", ckpt_dir, "--out", str(out_png)],
                    cwd=PROJECT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace", env=SUBPROCESS_ENV,
                )
                if result.returncode == 0 and out_png.exists():
                    st.image(str(out_png))
                    with open(out_png, "rb") as f:
                        st.download_button("Télécharger l'image", f, file_name=out_png.name)
                else:
                    st.error(result.stderr or result.stdout)

        with col2:
            st.markdown("**Heatmap d'attention**")
            attn_prompt = st.text_input("Prompt", value="Il était une fois")
            layer_idx = st.number_input("Couche", min_value=0, value=0, step=1)
            head_idx = st.number_input("Tête", min_value=0, value=0, step=1)
            if st.button("Générer la heatmap"):
                out_png = PROJECT_DIR / f"{ckpt_dir}_attention.png"
                result = subprocess.run(
                    [PYTHON, str(PROJECT_DIR / "visualize_attention.py"),
                     "--checkpoint_dir", ckpt_dir, "--prompt", attn_prompt,
                     "--layer", str(layer_idx), "--head", str(head_idx),
                     "--out", str(out_png)],
                    cwd=PROJECT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace", env=SUBPROCESS_ENV,
                )
                if result.returncode == 0 and out_png.exists():
                    st.image(str(out_png))
                    with open(out_png, "rb") as f:
                        st.download_button("Télécharger l'image", f, file_name=out_png.name)
                else:
                    st.error(result.stderr or result.stdout)