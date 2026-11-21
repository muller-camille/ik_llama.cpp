#!/bin/bash
# Cherche le meilleur réglage du drafter DSpark sur cette machine : longueur des brouillons (n_max) et
# seuil de confiance en dessous duquel un brouillon est coupé (p_min, 0,75 par défaut).
#
#   ./drafter.sh <fichier drafter>
#
# Le script mesure la version dans laquelle il se trouve. Un dossier ik_llama en premier argument en
# choisit une autre : ./drafter.sh <dossier ik_llama> <fichier drafter>
# Autre modèle : MODEL=/mnt/ds0/...-R8Q6-00001-of-00009.gguf ./drafter.sh ...
# Autres réglages : REGLAGES="n_max=4 n_max=5,p_min=0.4" ./drafter.sh ...
#
# Une passe par réglage, environ 2 minutes chacune. Si le drafter converti (même nom avec -R8) est à côté
# du drafter, une passe de plus le mesure avec le premier réglage. Le résultat est écrit dans
# drafter-resultat.txt, dans le dossier où vous lancez le script : c'est ce fichier qu'il faut envoyer.
# Il ne contient ni nom de machine ni nom d'utilisateur.
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
DRAFT=${1:?"Usage : $0 [dossier ik_llama] <fichier drafter>"}
MODEL=${MODEL:-/mnt/ds0/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8-00001-of-00009.gguf}
MIRROR=${MIRROR-/mnt/ds1}
T=${T:-$COEURS}
CTX=${CTX:-131072}
# le premier est le réglage habituel ; le drafter a été entraîné pour des blocs de 5 tokens
REGLAGES=${REGLAGES:-"n_max=4 n_max=4,p_min=0.5 n_max=4,p_min=0.3 n_max=4,p_min=0 n_max=5,p_min=0.5 n_max=5,p_min=0.3 n_max=5,p_min=0"}
API_KEY=mesure   # clé du serveur que ce script lance pour lui seul, sur 127.0.0.1
PORT=${PORT_MESURE:-8080}
ESSAIS=${ESSAIS:-5}
N_PREDICT=${N_PREDICT:-200}
OUT=$PWD/drafter-resultat.txt
LOG=/tmp/drafter-llama-$$.log

if pgrep -x llama-server > /dev/null; then
    echo "Un llama-server tourne déjà :"
    pgrep -a -x llama-server | cut -c1-160
    echo "Arrêtez-le d'abord (kill <PID>), puis relancez ce script."
    exit 1
fi
[ -x "$IK_DIR/llama-server" ] || { echo "llama-server introuvable dans $IK_DIR"; exit 1; }
[ -f "$DRAFT" ] || { echo "Drafter introuvable : $DRAFT"; exit 1; }
[ -f "$MODEL" ] || { echo "Modèle introuvable : $MODEL"; exit 1; }
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
    echo "modèle : $(basename "$MODEL"), $T threads"
} > "$OUT" 2>&1
echo "== Résultat dans $OUT"
RESUME=()

# passe <drafter> <réglage>
passe() {
    local draft=$1 reglage=$2 pret=0 v i rep texte moyenne=
    local args=(-m "$MODEL" -t "$T" -c "$CTX" -cram 100000 --jinja --swa-compress
                --api-key "$API_KEY" --host 127.0.0.1 --port "$PORT"
                -md "$draft" --spec-type "dspark:$reglage")
    [ -z "$MIRROR" ] || args+=(--numa-mirror "$MIRROR")

    dire ""
    dire "=== $(basename "$draft"), $reglage"
    ( cd "$IK_DIR" && exec numactl --interleave=all ./llama-server "${args[@]}" > "$LOG" 2>&1 ) &
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
            texte=$(echo "$rep" | grep -o '"content":"\(\\.\|[^"\\]\)*"' | head -1 | cut -c12- | sed 's/"$//' || true)
            dire "texte : $(echo "$texte" | cut -c1-150)"
        fi
    done
    if [ ${#valeurs[@]} -gt 0 ]; then
        moyenne=$(printf '%s\n' "${valeurs[@]}" | awk '{ s += $1 } END { printf "%.2f", s/NR }')
        dire "moyenne sur ${#valeurs[@]} essais : $moyenne tokens/s"
    fi

    arreter   # les statistiques du drafter sont écrites à l'arrêt du serveur
    grep -E "draft acceptance rate|statistics dspark" "$LOG" | tail -2 | tee -a "$OUT" || true
    RESUME+=("$(printf '%8s tokens/s  %-22s %s' "${moyenne:--}" "$reglage" "$(basename "$draft")")")
    rm -f "$LOG"
}

# une passe qui échoue n'empêche pas les suivantes
PREMIER=
for r in $REGLAGES; do
    [ -n "$PREMIER" ] || PREMIER=$r
    passe "$DRAFT" "$r" || true
done
R8=${DRAFT%.gguf}-R8.gguf
if [ -f "$R8" ]; then
    passe "$R8" "$PREMIER" || true
fi

dire ""
dire "=== résumé"
for l in ${RESUME[@]+"${RESUME[@]}"}; do
    dire "$l"
done
echo "== Terminé. Envoyez le fichier $OUT"
