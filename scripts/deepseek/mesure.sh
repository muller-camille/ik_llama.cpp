#!/bin/bash
# Lance un serveur seul, fait 5 mesures de vitesse, affiche la moyenne, puis arrête le serveur.
#
#   ./mesure.sh                      mesure sans drafter
#   ./mesure.sh <fichier drafter>    mesure avec ce drafter
#
# Le script mesure la version dans laquelle il se trouve. Un dossier ik_llama en premier argument en
# choisit une autre : ./mesure.sh <dossier ik_llama> [fichier drafter]
#
# Réglages par variables : T (threads, un par cœur physique par défaut), MODEL (premier fichier du modèle),
# MIRROR (copie des poids pour l'autre socket, vide = aucune), SPEC (réglage du drafter), CONTEXTE (nombre
# de tokens d'un texte placé avant la question, 0 par défaut : la vitesse avec un long contexte n'est pas
# celle d'une question courte).
# Exemples : MODEL=/mnt/ds0/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8Q6-00001-of-00009.gguf ./mesure.sh
#            CONTEXTE=4000 ./mesure.sh
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
T=${T:-$COEURS}
CTX=${CTX:-131072}
UBATCH=${UBATCH:-2048}   # tokens du prompt lus ensemble, comme deepseek.sh
SPEC=${SPEC:-dspark:n_max=4}
CLE=mesure   # clé du serveur que ce script lance pour lui seul, sur 127.0.0.1
PORT=${PORT_MESURE:-8080}
ESSAIS=${ESSAIS:-5}
N_PREDICT=${N_PREDICT:-200}
CONTEXTE=${CONTEXTE:-0}
LOG=/tmp/mesure-llama-$$.log

# un autre serveur sur la machine fausserait les mesures (et pourrait répondre à notre place)
if pgrep -x llama-server > /dev/null; then
    echo "Un llama-server tourne déjà :"
    pgrep -a -x llama-server | cut -c1-160
    echo "Arrêtez-le d'abord (kill <PID>), puis relancez ce script."
    exit 1
fi
[ -x "$IK_DIR/llama-server" ] || { echo "llama-server introuvable dans $IK_DIR"; exit 1; }
[ -z "$DRAFT" ] || [ -f "$DRAFT" ] || { echo "Drafter introuvable : $DRAFT"; exit 1; }
[ -f "$MODEL" ] || { echo "Modèle introuvable : $MODEL (variable MODEL)"; exit 1; }

ARGS=(-m "$MODEL" -t "$T" -c "$CTX" -cram 100000 --jinja --swa-compress
      -b "$((UBATCH > 2048 ? UBATCH : 2048))" -ub "$UBATCH"
      --api-key "$CLE" --host 127.0.0.1 --port "$PORT")
[ -z "$MIRROR" ] || ARGS+=(--numa-mirror "$MIRROR")
[ -z "$DRAFT" ]  || ARGS+=(-md "$DRAFT" --spec-type "$SPEC")

cd "$IK_DIR"
echo "== $(./llama-server --version 2>&1 | head -1), $T threads, drafter : ${DRAFT:-aucun}"
echo "== modèle : $(basename "$MODEL")"
[ "$CONTEXTE" -le 0 ] || echo "== contexte : environ $CONTEXTE tokens avant la question"
numactl --interleave=all ./llama-server "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true; echo "== Serveur arrêté. Journal complet : $LOG"' EXIT

echo "== Serveur lancé (PID $PID), chargement..."
pret=0
for i in $(seq 1 300); do
    if ! kill -0 $PID 2>/dev/null; then
        echo "Le serveur s'est arrêté pendant le chargement. Fin du journal :"
        tail -5 "$LOG"
        exit 1
    fi
    if curl -s "http://127.0.0.1:$PORT/health" -H "Authorization: Bearer $CLE" 2>/dev/null | grep -q '"status":"ok"'; then
        pret=1
        break
    fi
    sleep 2
done
[ $pret -eq 1 ] || { echo "Le serveur ne répond pas après 10 minutes."; exit 1; }

# La requête passe par un fichier : un texte long ne tient pas sur une ligne de commande. Le texte est fait
# de paragraphes numérotés, 85 tokens environ chacun.
INVITE=/tmp/mesure-invite-$$.json
{
    printf '{"prompt":"'
    i=1
    while [ $((i*85)) -le "$CONTEXTE" ]; do
        printf 'Note %d. The archive of the city keeps the registers of the guilds, the maps of the river and the letters of the merchants who crossed the bridge every morning before the market opened. Each volume was copied by hand, checked against the older one and stored on the upper floor, away from the damp of the cellars.\\n\\n' "$i"
        i=$((i+1))
    done
    printf 'Write a long essay about the history of Paris.\\n\\n'
    # after a long text, the title of the essay: without it the model goes on with the text
    [ "$CONTEXTE" -le 0 ] || printf '# The History of Paris\\n\\n'
    printf '","n_predict":%d,"temperature":0,"cache_prompt":true}' "$N_PREDICT"
} > "$INVITE"

valeurs=()
for i in $(seq 1 "$ESSAIS"); do
    rep=$(curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" -H "Authorization: Bearer $CLE" \
            -d @"$INVITE" || true)
    v=$(echo "$rep" | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2 || true)
    if [ "$i" = 1 ] && [ "$CONTEXTE" -gt 0 ]; then
        # le premier essai lit le texte, les suivants le retrouvent dans le cache du serveur
        echo "texte lu : $(echo "$rep" | grep -o '"prompt_n":[0-9]*' | head -1 | cut -d: -f2) tokens"
    fi
    if [ -z "$v" ]; then
        echo "essai $i : pas de réponse"
    else
        printf "essai %d : %.2f tokens/s\n" "$i" "$v"
        valeurs+=("$v")
    fi
done

if [ ${#valeurs[@]} -gt 0 ]; then
    printf '%s\n' "${valeurs[@]}" | awk '{ s += $1 } END { printf "== moyenne sur %d essais : %.2f tokens/s\n", NR, s/NR }'
fi
rm -f "$INVITE"
grep -E "draft acceptance rate|statistics dspark" "$LOG" | tail -2 || true
