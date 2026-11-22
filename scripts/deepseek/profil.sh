#!/bin/bash
# Chronomètre les opérations du modèle sur cette machine, pour savoir où part le temps.
#
#   ./profil.sh                      toutes les passes, avec le drafter rangé à côté du modèle
#   ./profil.sh <fichier drafter>    les mêmes, avec ce drafter
#   DRAFT= ./profil.sh               sans drafter (en csh : env DRAFT= ./profil.sh)
#
# Le script mesure la version dans laquelle il se trouve. Un dossier ik_llama en premier argument en
# choisit une autre : ./profil.sh <dossier ik_llama> [fichier drafter]
# Autre modèle : MODEL=/mnt/ds0/...-R8Q6-00001-of-00009.gguf ./profil.sh ...
#
# Les passes, dans cet ordre :
#   1. la version telle quelle, un thread par cœur physique : sans puis avec le drafter, après une question
#      courte puis après un texte long (au-delà de 256 tokens de contexte l'attention ne suit plus le même
#      chemin dans le code, et le texte long donne la vitesse de lecture d'un prompt) ;
#   2. les mêmes avec un réglage changé à la fois (VARS), après le texte long, sans puis avec le drafter ;
#   3. les passes du 1. avec les autres nombres de threads (THREADS) : un par processeur logique, soit le
#      double sur une machine à hyperthreading.
# Sur une machine à 2 x 16 cœurs et hyperthreading, avec un drafter et les réglages par défaut : 14 passes.
#
#   THREADS    les nombres de threads à mesurer, par défaut les cœurs physiques puis les processeurs
#              logiques (32 et 64 sur 2 x 16 cœurs à hyperthreading) ; le premier est celui des passes 1 et 2
#   VARS       les réglages à essayer tour à tour, par défaut "UBATCH=512 GGML_INDEXER_512=0 GGML_FA_ROW_GROUPS=-1" :
#              - UBATCH=512 lit le prompt par lots de 512 tokens au lieu de 2048 : ce n'est pas une variable du
#                serveur mais son option -ub, la ligne « texte lu » dit ce que cela change ;
#              - GGML_INDEXER_512=0 reprend l'ancien indexeur à la place de celui en AVX-512 ;
#              - GGML_FA_ROW_GROUPS=-1 tire au sort la grille de threads de l'attention à chaque nœud, et le
#                chronométrage dit ce que chacune a coûté (les tokens/s de ces passes ne se comparent donc
#                pas aux autres)
#              (autres exemples : VARS="GGML_FA_SKIP_MASKED=0 GGML_SHARE_ROWS=0 GGML_LAZY_CONCAT=0" coupe ces
#              optimisations tour à tour ; VARS= ne fait que les passes 1 et 3)
#   CONTEXTE   taille du texte long en tokens, 16000 par défaut ; CONTEXTE=0 saute les passes au texte long
#   UBATCH     tokens du prompt lus ensemble dans toutes les passes, 2048 par défaut comme deepseek.sh
#   DRAFT      le drafter : un chemin, ou un nom de fichier cherché à côté du modèle (celui de
#              ~/.deepseek.conf) ; vide = sans drafter ; par défaut le dspark-*-R8.gguf à côté du modèle,
#              sinon le dspark-*.gguf
#   ESSAIS     requêtes par passe, 5 par défaut ; N_PREDICT, tokens produits par requête, 200
# Chaque passe note aussi le début du texte produit et son empreinte. Le texte peut différer d'une passe
# à l'autre dès le premier token : il suffit que deux tokens soient presque à égalité et qu'un arrondi
# change. Il doit rester un texte sensé.
#
# Durée : 2 à 3 minutes par passe, une demi-heure pour les 14. Le résultat est écrit dans
# profil-resultat.txt, dans le dossier où vous lancez le script, et se termine par un tableau d'une ligne
# par passe : c'est ce fichier qu'il faut envoyer. Il ne contient ni nom de machine ni nom d'utilisateur.
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
# un thread par cœur physique, puis un par processeur logique si l'hyperthreading en donne davantage
COEURS=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$COEURS" -gt 0 ] 2>/dev/null || COEURS=$(nproc)
LOGIQUES=$(nproc)
MODEL=${MODEL:-/mnt/ds0/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8-00001-of-00009.gguf}
MIRROR=${MIRROR-/mnt/ds1}
if [ -z "${THREADS:-}" ]; then
    THREADS=${T:-$COEURS}
    [ "$LOGIQUES" -le "$THREADS" ] || THREADS="$THREADS $LOGIQUES"
fi
T=${THREADS%% *}       # threads des passes principales
# le drafter : en argument, sinon DRAFT, sinon celui qui est rangé à côté du modèle
if [ $# -gt 0 ]; then
    DRAFT=$1
elif [ -z "${DRAFT+x}" ]; then
    DRAFT=
    for f in "$(dirname "$MODEL")"/dspark-*-R8.gguf "$(dirname "$MODEL")"/dspark-*.gguf; do
        if [ -f "$f" ]; then DRAFT=$f; break; fi
    done
fi
# un nom sans dossier, comme dans ~/.deepseek.conf, se cherche à côté du modèle
if [ -n "$DRAFT" ] && [ ! -f "$DRAFT" ] && [ -f "$(dirname "$MODEL")/$DRAFT" ]; then
    DRAFT=$(dirname "$MODEL")/$DRAFT
fi
VARS=${VARS-UBATCH=512 GGML_INDEXER_512=0 GGML_FA_ROW_GROUPS=-1}
CONTEXTE=${CONTEXTE:-16000}
CTX=${CTX:-131072}
UBATCH=${UBATCH:-2048}
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
[ -z "$DRAFT" ] || [ -f "$DRAFT" ] || { echo "Drafter introuvable : $DRAFT (DRAFT= pour s'en passer)"; exit 1; }
[ -f "$MODEL" ] || { echo "Modèle introuvable : $MODEL (variable MODEL)"; exit 1; }
for n in $THREADS; do
    [[ $n =~ ^[1-9][0-9]*$ ]] || { echo "THREADS=$THREADS : des nombres de threads, séparés par des espaces."; exit 1; }
done
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

# les passes : d'abord le nombre de threads principal, puis les réglages, puis les autres nombres de threads
DRAFTERS=("")
NOM_DRAFT=aucun
if [ -n "$DRAFT" ]; then
    DRAFTERS+=("$DRAFT")
    NOM_DRAFT=$(basename "$DRAFT")
fi
CONTEXTES="0"
[ "$CONTEXTE" -le 0 ] || CONTEXTES="0 $CONTEXTE"
N_VARS=$(set -- $VARS; echo $#)
N_THREADS=$(set -- $THREADS; echo $#)
N_PASSES=$(( ${#DRAFTERS[@]} * (N_THREADS * $(set -- $CONTEXTES; echo $#) + N_VARS) ))

{
    echo "=== machine"
    "$IK_DIR/llama-server" --version 2>&1 | head -1
    lscpu | grep -E "^(Model name|Socket|Core|Thread)" || true
    echo "threads : $THREADS ; drafter : $NOM_DRAFT ; texte long : $CONTEXTE tokens ; lots de $UBATCH tokens ; réglages : ${VARS:-aucun}"
} > "$OUT" 2>&1
echo "== Résultat dans $OUT"
echo "== $N_PASSES passes, environ $(( (N_PASSES*5 + 1)/2 )) minutes : threads $THREADS, drafter $NOM_DRAFT"
RESUME=()

# remplir <tokens> : un texte d'à peu près cette longueur, en paragraphes numérotés (85 tokens environ chacun),
# déjà échappé pour JSON
remplir() {
    local n=$1 i=1
    while [ $((i*85)) -le "$n" ]; do
        printf 'Note %d. The archive of the city keeps the registers of the guilds, the maps of the river and the letters of the merchants who crossed the bridge every morning before the market opened. Each volume was copied by hand, checked against the older one and stored on the upper floor, away from the damp of the cellars.\\n\\n' "$i"
        i=$((i+1))
    done
}

# resumer <threads> <drafter ou vide> <contexte> <réglage> <tokens lus> <lecture> <tokens/s produits> <acceptés> :
# la ligne de la passe dans le tableau final
resumer() {
    RESUME+=("$(printf '%7s  %-7s  %-8s  %-24s  %7s  %8s  %8s  %6s' "$1" "$([ -n "$2" ] && echo oui || echo non)" \
        "$([ "$3" -gt 0 ] && echo long || echo court)" "${4:--}" "$5" "$6" "$7" "$8")")
}

N_PASSE=0
# passe <threads> <drafter ou vide> <tokens de texte avant la question, ou 0> [VARIABLE=valeur]
passe() {
    local t=$1 draft=$2 contexte=$3 var=${4:-} pret=0 v i rep texte invite
    local lu="-" lecture="-" moyenne="-" accepte="-"
    N_PASSE=$((N_PASSE + 1))
    local args=(-m "$MODEL" -t "$t" -c "$CTX" -cram 100000 --jinja --swa-compress
                --api-key "$API_KEY" --host 127.0.0.1 --port "$PORT")
    [ -z "$MIRROR" ] || args+=(--numa-mirror "$MIRROR")
    [ -z "$draft" ]  || args+=(-md "$draft" --spec-type "$SPEC")
    # la taille des lots du prompt, une option du serveur : celle de UBATCH, ou d'un UBATCH=<n> parmi les réglages
    v=$UBATCH
    case $var in
        UBATCH=*) v=${var#UBATCH=} ;;
    esac
    args+=(-b "$((v > 2048 ? v : 2048))" -ub "$v")

    dire ""
    echo "== passe $N_PASSE sur $N_PASSES"
    if [ "$contexte" -gt 0 ]; then
        dire "=== $t threads, drafter : $([ -n "$draft" ] && echo "$NOM_DRAFT" || echo aucun), contexte long${var:+, $var}"
    else
        dire "=== $t threads, drafter : $([ -n "$draft" ] && echo "$NOM_DRAFT" || echo aucun)${var:+, $var}"
    fi
    ( cd "$IK_DIR" && exec env GGML_PROFILE=1 $var numactl --interleave=all ./llama-server "${args[@]}" > "$LOG" 2>&1 ) &
    PID=$!
    echo "   serveur lancé (PID $PID), chargement..."
    for i in $(seq 1 300); do
        if ! kill -0 "$PID" 2>/dev/null; then
            dire "Le serveur s'est arrêté pendant le chargement. Fin du journal :"
            tail -5 "$LOG" | tee -a "$OUT"
            PID=
            resumer "$t" "$draft" "$contexte" "$var" "-" "-" "échec" "-"
            return 1
        fi
        if curl -s "http://127.0.0.1:$PORT/health" -H "Authorization: Bearer $API_KEY" 2>/dev/null | grep -q '"status":"ok"'; then
            pret=1
            break
        fi
        sleep 2
    done
    if [ $pret -ne 1 ]; then
        dire "Le serveur ne répond pas après 10 minutes."
        arreter
        resumer "$t" "$draft" "$contexte" "$var" "-" "-" "échec" "-"
        return 1
    fi

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
            lu=$(echo "$rep" | grep -o '"prompt_n":[0-9]*' | head -1 | cut -d: -f2 || true)
            lecture=$(echo "$rep" | grep -o '"prompt_per_second":[0-9.]*' | head -1 | cut -d: -f2 | xargs printf '%.1f' 2>/dev/null || echo '?')
            dire "texte lu : $lu tokens, $lecture tokens/s"
            lu=${lu:-?}
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
        moyenne=$(printf '%s\n' "${valeurs[@]}" | awk '{ s += $1 } END { printf "%.2f", s/NR }')
    else
        moyenne="échec"
    fi

    arreter   # le tableau est écrit à l'arrêt du serveur
    grep -E "draft acceptance rate|statistics dspark" "$LOG" | tail -2 >> "$OUT" || true
    if [ -n "$draft" ]; then
        accepte=$(grep -o "draft acceptance rate = [0-9.]*" "$LOG" | tail -1 | awk '{ printf "%.0f%%", 100*$NF }' || true)
        accepte=${accepte:--}
    fi
    resumer "$t" "$draft" "$contexte" "$var" "$lu" "$lecture" "$moyenne" "$accepte"
    if grep -q "^ggml_profile:" "$LOG"; then
        grep "^ggml_profile:" "$LOG" >> "$OUT"
        echo "   chronométrage enregistré"
    else
        dire "Pas de chronométrage dans le journal : cette version est-elle assez récente ?"
    fi
    rm -f "$LOG" "$invite"
}

# une passe qui échoue n'empêche pas les suivantes
# 1. la version telle quelle, au nombre de threads principal
for draft in "${DRAFTERS[@]}"; do
    for contexte in $CONTEXTES; do
        passe "$T" "$draft" "$contexte" || true
    done
done
# 2. un réglage changé à la fois, après le texte long (la question courte si CONTEXTE=0)
for var in $VARS; do
    for draft in "${DRAFTERS[@]}"; do
        passe "$T" "$draft" "${CONTEXTES##* }" "$var" || true
    done
done
# 3. les autres nombres de threads
for t in ${THREADS#* }; do
    [ "$t" != "$T" ] || continue
    for draft in "${DRAFTERS[@]}"; do
        for contexte in $CONTEXTES; do
            passe "$t" "$draft" "$contexte" || true
        done
    done
done

{
    echo ""
    echo "=== résumé, une ligne par passe"
    echo "lus : tokens du texte long ; lecture et sortie : tokens/s lus et produits ; draft : part acceptée des tokens du drafter"
    printf '%7s  %-7s  %-8s  %-24s  %7s  %8s  %8s  %6s\n' threads drafter contexte variante lus lecture sortie draft
    printf '%s\n' "${RESUME[@]}"
} | tee -a "$OUT"
echo "== Terminé. Envoyez le fichier $OUT"
