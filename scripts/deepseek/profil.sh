#!/bin/bash
# Chronomètre les opérations du modèle sur cette machine, pour savoir où part le temps.
#
#   ./profil.sh                      quatre passes sans drafter
#   ./profil.sh <fichier drafter>    trois passes de plus, avec le drafter
#
# Le script mesure la version dans laquelle il se trouve. Un dossier ik_llama en premier argument en
# choisit une autre : ./profil.sh <dossier ik_llama> [fichier drafter]
# Autre modèle : MODEL=/mnt/ds0/...-R8Q6-00001-of-00009.gguf ./profil.sh ...
#
# Les passes comparent la version telle quelle (un thread par cœur physique), la même sans le partage des
# lignes entre threads (GGML_SHARE_ROWS=0), la même avec l'ancien calcul des hyper-connexions
# (GGML_HC_PROJ=0) et la même avec deux fois plus de threads. Chaque passe note aussi le début du texte
# produit et son empreinte. Le texte peut différer d'une passe à l'autre dès le premier token : il suffit
# que deux tokens soient presque à égalité et qu'un arrondi change. Il doit rester un texte sensé.
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
# un thread par cœur physique
COEURS=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$COEURS" -gt 0 ] 2>/dev/null || COEURS=$(nproc)
DRAFT=${1:-}
MODEL=${MODEL:-/mnt/ds0/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8-00001-of-00009.gguf}
MIRROR=${MIRROR-/mnt/ds1}
T=${T:-$COEURS}        # threads des passes principales
T2=${T2:-$((2*T))}     # threads de la passe de comparaison
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

# passe <threads> <drafter ou vide> [VARIABLE=valeur]
passe() {
    local t=$1 draft=$2 var=${3:-} pret=0 v i rep texte
    local args=(-m "$MODEL" -t "$t" -c "$CTX" -cram 100000 --jinja --swa-compress
                --api-key "$API_KEY" --host 127.0.0.1 --port "$PORT")
    [ -z "$MIRROR" ] || args+=(--numa-mirror "$MIRROR")
    [ -z "$draft" ]  || args+=(-md "$draft" --spec-type "$SPEC")

    dire ""
    dire "=== $t threads, drafter : ${draft:-aucun}${var:+, $var}"
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

    local valeurs=()
    for i in $(seq 1 "$ESSAIS"); do
        rep=$(curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" \
                -d '{"prompt":"Write a long essay about the history of Paris.\n\n","n_predict":'"$N_PREDICT"',"temperature":0}' || true)
        v=$(echo "$rep" | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2 || true)
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
    rm -f "$LOG"
}

# une passe qui échoue n'empêche pas les suivantes
passe "$T"  ""                         || true
passe "$T"  "" GGML_SHARE_ROWS=0       || true
passe "$T"  "" GGML_HC_PROJ=0          || true
passe "$T2" ""                         || true
if [ -n "$DRAFT" ]; then
    passe "$T"  "$DRAFT"                 || true
    passe "$T"  "$DRAFT" GGML_SHARE_ROWS=0 || true
    passe "$T"  "$DRAFT" GGML_HC_PROJ=0  || true
fi
echo "== Terminé. Envoyez le fichier $OUT"
