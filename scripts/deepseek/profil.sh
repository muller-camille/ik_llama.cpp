#!/bin/bash
# Chronomètre les opérations du modèle sur cette machine, pour savoir où part le temps.
#
#   ./profil.sh                      six passes sans drafter
#   ./profil.sh <fichier drafter>    six passes de plus, avec le drafter
#
# Le script mesure la version dans laquelle il se trouve. Un dossier ik_llama en premier argument en
# choisit une autre : ./profil.sh <dossier ik_llama> [fichier drafter]
# Autre modèle : MODEL=/mnt/ds0/...-R8Q6-00001-of-00009.gguf ./profil.sh ...
#
# Les passes comparent la version telle quelle (un thread par cœur physique) et la même avec un réglage
# changé, d'abord après une question courte, puis après un texte d'environ 4000 tokens : au-delà de
# 256 tokens de contexte, l'attention ne suit plus le même chemin dans le code.
#   VARS       les réglages à essayer tour à tour, par défaut "GGML_LAZY_CONCAT=0 GGML_FA_ROW_GROUPS=-1" :
#              - GGML_LAZY_CONCAT=0 recopie le cache en un bloc avant l'attention, comme avant ;
#              - GGML_FA_ROW_GROUPS=-1 tire au sort la grille de threads de l'attention à chaque nœud, et le
#                chronométrage dit ce que chacune a coûté (les tokens/s de ces passes ne se comparent donc
#                pas aux autres)
#              (autre exemple : VARS="GGML_FA_SKIP_MASKED=0 GGML_SHARE_ROWS=0" coupe ces optimisations tour à tour ;
#              VARS="GGML_INDEXER_512=0" compare l'indexeur AVX-512 à l'ancien, à mesurer avec CONTEXTE=20000 ;
#              VARS= ne fait que les quatre passes de base)
#   CONTEXTE   taille du texte long en tokens, 4000 par défaut ; CONTEXTE=0 saute ces passes
# Chaque passe note aussi le début du texte produit et son empreinte. Le texte peut différer d'une passe
# à l'autre dès le premier token : il suffit que deux tokens soient presque à égalité et qu'un arrondi
# change. Il doit rester un texte sensé.
#
# Durée : environ 2 minutes par passe. Le résultat est écrit dans profil-resultat.txt, dans le dossier
# où vous lancez le script : c'est ce fichier qu'il faut envoyer. Il ne contient ni nom de machine ni nom
# d'utilisateur.
set -euo pipefail

# --- réglages --------------------------------------------------------------
# Ce qui est propre à la machine (NFS, API_KEY, HOST...) se lit dans ~/.deepseek.conf, une ligne NOM=valeur
# par réglage, sans guillemets. Une variable d'environnement passe avant ce fichier.
ICI=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
MAISON=$(getent passwd "${SUDO_USER:-$(id -un)}" 2>/dev/null | cut -d: -f6 || true)
CONF=${DEEPSEEK_CONF:-${MAISON:-$HOME}/.deepseek.conf}
if [ -f "$CONF" ]; then
    while IFS='=' read -r nom valeur || [ -n "$nom" ]; do
        [[ $nom =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [ -n "${!nom+x}" ] || export "$nom=$valeur"
    done < "$CONF"
fi
# un dossier ik_llama en premier argument, sinon celui où se trouve ce script
if [ $# -gt 0 ] && [ -d "$1" ]; then
    IK_DIR=$1
    shift
else
    IK_DIR=${IK_DIR:-$ICI}
fi
# un réglage IK_DIR (variable ou ~/.deepseek.conf) fait travailler ce script avec une autre version que la sienne
if [ "$(cd "$IK_DIR" 2>/dev/null && pwd -P)" != "$(cd "$ICI" && pwd -P)" ]; then
    echo "== ATTENTION : les outils utilisés sont ceux de $IK_DIR, pas ceux du dossier de ce script ($ICI)"
fi
# un thread par cœur physique
COEURS=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$COEURS" -gt 0 ] 2>/dev/null || COEURS=$(nproc)
DRAFT=${1:-}
MODEL=${MODEL:-/mnt/ds0/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8-00001-of-00009.gguf}
MIRROR=${MIRROR-/mnt/ds1}
T=${T:-$COEURS}        # threads des passes principales
VARS=${VARS-GGML_LAZY_CONCAT=0 GGML_FA_ROW_GROUPS=-1}
CONTEXTE=${CONTEXTE:-4000}
CTX=${CTX:-131072}
SPEC=${SPEC:-dspark:n_max=4}
API_KEY=mesure         # clé du serveur que ce script lance pour lui seul, sur 127.0.0.1
PORT=${PORT_MESURE:-8080}
ESSAIS=${ESSAIS:-5}
N_PREDICT=${N_PREDICT:-200}
OUT=$PWD/profil-resultat.txt
LOG=/tmp/profil-llama-$$.log

if pgrep -x llama-server > /dev/null; then
    echo "Un llama-server tourne déjà :"
    pgrep -a -x llama-server | cut -c1-160
    echo "Arrêtez-le d'abord (kill <PID>), puis relancez ce script."
    exit 1
fi
[ -x "$IK_DIR/llama-server" ] || { echo "llama-server introuvable dans $IK_DIR"; exit 1; }
[ -z "$DRAFT" ] || [ -f "$DRAFT" ] || { echo "Drafter introuvable : $DRAFT"; exit 1; }
[ -f "$MODEL" ] || { echo "Modèle introuvable : $MODEL (variable MODEL)"; exit 1; }
IK_DIR=$(cd "$IK_DIR" && pwd)

PID=
arreter() {
    if [ -n "$PID" ]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=
    fi
}
trap arreter EXIT

# dire <texte> : à l'écran et dans le fichier de résultat
dire() { echo "$@" | tee -a "$OUT"; }

{
    echo "=== machine"
    "$IK_DIR/llama-server" --version 2>&1 | head -1
    lscpu | grep -E "^(Model name|Socket|Core|Thread)" || true
} > "$OUT" 2>&1
echo "== Résultat dans $OUT"

# remplir <tokens> : un texte d'à peu près cette longueur, en paragraphes numérotés (85 tokens environ chacun),
# déjà échappé pour JSON
remplir() {
    local n=$1 i=1
    while [ $((i*85)) -le "$n" ]; do
        printf 'Note %d. The archive of the city keeps the registers of the guilds, the maps of the river and the letters of the merchants who crossed the bridge every morning before the market opened. Each volume was copied by hand, checked against the older one and stored on the upper floor, away from the damp of the cellars.\\n\\n' "$i"
        i=$((i+1))
    done
}

# passe <threads> <drafter ou vide> <tokens de texte avant la question, ou 0> [VARIABLE=valeur]
passe() {
    local t=$1 draft=$2 contexte=$3 var=${4:-} pret=0 v i rep texte invite
    local args=(-m "$MODEL" -t "$t" -c "$CTX" -cram 100000 --jinja --swa-compress
                --api-key "$API_KEY" --host 127.0.0.1 --port "$PORT")
    [ -z "$MIRROR" ] || args+=(--numa-mirror "$MIRROR")
    [ -z "$draft" ]  || args+=(-md "$draft" --spec-type "$SPEC")

    dire ""
    if [ "$contexte" -gt 0 ]; then
        dire "=== $t threads, drafter : ${draft:-aucun}, contexte long${var:+, $var}"
    else
        dire "=== $t threads, drafter : ${draft:-aucun}${var:+, $var}"
    fi
    ( cd "$IK_DIR" && exec env GGML_PROFILE=1 $var numactl --interleave=all ./llama-server "${args[@]}" > "$LOG" 2>&1 ) &
    PID=$!
    echo "   serveur lancé (PID $PID), chargement..."
    for i in $(seq 1 300); do
        if ! kill -0 "$PID" 2>/dev/null; then
            dire "Le serveur s'est arrêté pendant le chargement. Fin du journal :"
            tail -5 "$LOG" | tee -a "$OUT"
            PID=
            return 1
        fi
        if curl -s "http://127.0.0.1:$PORT/health" -H "Authorization: Bearer $API_KEY" 2>/dev/null | grep -q '"status":"ok"'; then
            pret=1
            break
        fi
        sleep 2
    done
    [ $pret -eq 1 ] || { dire "Le serveur ne répond pas après 10 minutes."; arreter; return 1; }

    # la requête passe par un fichier : un texte long ne tient pas sur une ligne de commande
    invite=/tmp/profil-invite-$$.json
    {
        printf '{"prompt":"'
        [ "$contexte" -le 0 ] || remplir "$contexte"
        printf 'Write a long essay about the history of Paris.\\n\\n'
        # after a long text, the title of the essay: without it the model goes on with the text
        [ "$contexte" -le 0 ] || printf '# The History of Paris\\n\\n'
        printf '","n_predict":%d,"temperature":0,"cache_prompt":true}' "$N_PREDICT"
    } > "$invite"
    local valeurs=()
    for i in $(seq 1 "$ESSAIS"); do
        rep=$(curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" \
                -d @"$invite" || true)
        v=$(echo "$rep" | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2 || true)
        if [ "$i" = 1 ] && [ "$contexte" -gt 0 ]; then
            # le premier essai lit le texte, les suivants le retrouvent dans le cache du serveur
            dire "texte lu : $(echo "$rep" | grep -o '"prompt_n":[0-9]*' | head -1 | cut -d: -f2) tokens, $(echo "$rep" | grep -o '"prompt_per_second":[0-9.]*' | head -1 | cut -d: -f2 | xargs printf '%.1f' 2>/dev/null || echo '?') tokens/s"
        fi
        if [ -z "$v" ]; then
            dire "essai $i : pas de réponse"
        else
            dire "essai $i : $(printf '%.2f' "$v") tokens/s"
            valeurs+=("$v")
        fi
        if [ "$i" = 1 ]; then
            # le texte produit : le même d'une passe à l'autre si le calcul est le même
            texte=$(echo "$rep" | grep -o '"content":"\(\\.\|[^"\\]\)*"' | head -1 | cut -c12- | sed 's/"$//' || true)
            dire "texte : $(echo "$texte" | cut -c1-150)"
            dire "empreinte du texte : $(echo "$texte" | md5sum | cut -c1-12)"
        fi
    done
    if [ ${#valeurs[@]} -gt 0 ]; then
        dire "$(printf '%s\n' "${valeurs[@]}" | awk '{ s += $1 } END { printf "moyenne sur %d essais : %.2f tokens/s", NR, s/NR }')"
    fi

    arreter   # le tableau est écrit à l'arrêt du serveur
    grep -E "draft acceptance rate|statistics dspark" "$LOG" | tail -2 >> "$OUT" || true
    if grep -q "^ggml_profile:" "$LOG"; then
        grep "^ggml_profile:" "$LOG" >> "$OUT"
        echo "   chronométrage enregistré"
    else
        dire "Pas de chronométrage dans le journal : cette version est-elle assez récente ?"
    fi
    rm -f "$LOG" "$invite"
}

# une passe qui échoue n'empêche pas les suivantes
passe "$T" "" 0 || true
for var in $VARS; do
    passe "$T" "" 0 "$var" || true
done
if [ "$CONTEXTE" -gt 0 ]; then
    passe "$T" "" "$CONTEXTE" || true
    for var in $VARS; do
        passe "$T" "" "$CONTEXTE" "$var" || true
    done
fi
if [ -n "$DRAFT" ]; then
    passe "$T" "$DRAFT" 0 || true
    for var in $VARS; do
        passe "$T" "$DRAFT" 0 "$var" || true
    done
    if [ "$CONTEXTE" -gt 0 ]; then
        passe "$T" "$DRAFT" "$CONTEXTE" || true
        for var in $VARS; do
            passe "$T" "$DRAFT" "$CONTEXTE" "$var" || true
        done
    fi
fi
echo "== Terminé. Envoyez le fichier $OUT"
