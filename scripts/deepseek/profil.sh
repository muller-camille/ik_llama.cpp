#!/bin/bash
# Mesure sur cette machine ce que le serveur donne selon la façon de le lancer, et chronomètre les
# opérations du modèle. Les réglages par défaut de deepseek.sh viennent de son résultat, et c'est lui qu'il
# faut relancer pour savoir si un autre réglage ferait mieux : rien ne se mesure ailleurs que sur le serveur.
#
#   ./profil.sh                      toutes les passes, avec le drafter rangé à côté du modèle
#   ./profil.sh <fichier drafter>    les mêmes, avec ce drafter
#   DRAFT= ./profil.sh               sans drafter (en csh : env DRAFT= ./profil.sh)
#
# Le script mesure la version dans laquelle il se trouve. Un dossier ik_llama en premier argument en
# choisit une autre : ./profil.sh <dossier ik_llama> [fichier drafter]
# Autre modèle : MODEL=/mnt/ds0/...-R8Q6-00001-of-00009.gguf ./profil.sh ...
#
# Ce qui est mesuré, sans puis avec le drafter :
#   1. les nombres de threads (THREADS), après une question courte puis après un texte long (au-delà de 256
#      tokens de contexte l'attention ne suit plus le même chemin dans le code, et le texte long donne la
#      vitesse de lecture d'un prompt). Sur une machine à hyperthreading : un thread par cœur physique, puis
#      un par processeur logique. Dans chaque cas la première passe est celle de deepseek.sh sans réglage,
#      la passe de référence : un thread par cœur physique, ou T (et TB) s'ils sont réglés ;
#   2. un réglage changé à la fois (VARS), après le texte long, avec les threads de la passe de référence.
# Chaque passe fait plusieurs essais : ESSAIS requêtes de N_PREDICT tokens, chacune sur un autre sujet (le
# premier est celui de toutes les mesures précédentes, l'histoire de Paris), et après le texte long les
# LECTURES premières lisent chacune un texte neuf, que le serveur ne peut pas reprendre de son cache. Le
# résultat donne chaque essai, la moyenne et les extrêmes. Plusieurs sujets, parce qu'avec le drafter la
# vitesse tient au texte : sur un seul, deux passes qui n'écrivent pas le même ne se comparent pas.
#
# Avec un drafter, le résultat donne aussi pour chaque passe les pas du drafter (une proposition et sa
# vérification par le modèle) : leur durée (ms/pas) et les tokens produits par pas (tok/pas).
#
# Sur une machine à 2 x 16 cœurs et hyperthreading, avec un drafter : 32 passes, environ 2 heures, pendant
# lesquelles le serveur habituel doit rester arrêté. RAPIDE=1 en fait 14 en 25 minutes. Ctrl+C arrête le
# script et écrit le résumé des passes déjà faites.
#
#   RAPIDE     1 = la version courte : 2 essais par passe, une seule lecture du texte long, et les trois
#              réglages GGML_AMX=0, GGML_AMX_ATTN=1 et UBATCH=512
#   THREADS    les nombres de threads à mesurer. Par défaut les cœurs physiques puis les processeurs
#              logiques (32 et 64 sur 2 x 16 cœurs à hyperthreading). a/b = a threads pour un token seul et
#              b pour les lots de tokens (options -t et -tb du serveur) : mesuré avec rhel8-b5379, 32/64
#              perd 11 ms par pas du drafter, le temps de réveiller les threads à chaque changement
#   T, TB      les threads de deepseek.sh s'ils sont réglés (~/.deepseek.conf) : ceux de la passe de référence
#   VARS       les réglages à essayer tour à tour. Par défaut tous ceux qui se comparent :
#              - UBATCH=512 et UBATCH=4096 lisent le prompt par lots de 512 ou de 4096 tokens au lieu de 2048 :
#                ce n'est pas une variable du serveur mais son option -ub, la ligne « texte lu » dit ce que
#                cela change ;
#              - GGML_AMX=0 lit le prompt sans les tuiles AMX du processeur (Sapphire Rapids et suivants) :
#                les produits des matrices Q8_0_R8 par un lot de tokens reprennent les noyaux AVX-512 ;
#              - GGML_AMX_ATTN=1 fait passer aussi l'attention d'un lot de tokens par les tuiles AMX : c'est
#                le seul réglage de la liste qui ajoute quelque chose au lieu de le couper. Il n'est pas
#                encore celui de deepseek.sh : cette passe dit ce qu'il donne (ligne « texte lu », et
#                FLASH_ATTN_EXT dans le chronométrage). Les requêtes, les lignes du cache et les poids de
#                l'attention y sont arrondis à 8 bits significatifs (bf16) ;
#              - GGML_INDEXER_512=0 reprend l'ancien indexeur à la place de celui en AVX-512 ;
#              - GGML_FA_ROW_GROUPS=-1 tire au sort la grille de threads de l'attention à chaque nœud, et le
#                chronométrage dit ce que chacune a coûté (les tokens/s de ces passes ne se comparent donc
#                pas aux autres) ;
#              - GGML_LAZY_CONCAT=0 recopie le cache en un bloc avant l'attention ; GGML_FA_SKIP_MASKED=0
#                calcule l'attention sur tout le cache ; GGML_SHARE_ROWS=0, GGML_HC_PROJ=0,
#                GGML_SERIAL_NODES=0 et IQK_PREFETCH=0 coupent chacun une optimisation de la lecture des poids
#              Hors de la liste par défaut, la demande des poids en avance se règle aussi :
#              IQK_PREFETCH=<octets>[,<octets>[,<niveau>]] : la distance à laquelle les noyaux demandent les
#              poids (4096 par défaut), une deuxième demande plus loin, vers le cache L2 (0 = aucune), et le
#              niveau de la première (0 = cache L1, par défaut ; 1 = L2 ; 3 = sans passer par L2).
#              IQK_PREFETCH=1024, IQK_PREFETCH=4096,32768 ou IQK_PREFETCH=16384,0,1 par exemple
#              (VARS="GGML_INDEXER_512=0" n'en essaie qu'un ; VARS= ne mesure que les nombres de threads)
#   ESSAIS     requêtes par passe, 5 par défaut ; LECTURES, lectures d'un texte neuf parmi elles, 3 ;
#              N_PREDICT, tokens produits par requête, 200
#   CONTEXTE   taille du texte long en tokens, 16000 par défaut ; CONTEXTE=0 saute les passes au texte long
#   UBATCH     tokens du prompt lus ensemble dans toutes les passes, 2048 par défaut comme deepseek.sh
#   DRAFT      le drafter : un chemin, ou un nom de fichier cherché à côté du modèle (celui de
#              ~/.deepseek.conf) ; vide = sans drafter ; par défaut le dspark-*-R8.gguf à côté du modèle,
#              sinon le dspark-*.gguf
# Chaque passe note aussi le début du premier texte produit et son empreinte. Le texte peut différer d'une
# passe à l'autre dès le premier token : il suffit que deux tokens soient presque à égalité et qu'un arrondi
# change. Il doit rester un texte sensé.
#
# Le résultat est écrit dans profil-resultat.txt, dans le dossier où vous lancez le script : c'est ce
# fichier qu'il faut envoyer. Il ne contient ni nom de machine ni nom d'utilisateur, et se termine par
#   - un tableau d'une ligne par passe ;
#   - pour chaque colonne, le nombre de threads le plus rapide et les chiffres des autres, avec le réglage
#     de deepseek.sh qui le donne ;
#   - ce que chaque réglage de VARS change par rapport à la passe de référence.
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
# voulue <drafter ou vide> : les threads de deepseek.sh dans ce cas. T (et TB) s'ils sont réglés, sinon un par
# cœur physique, avec ou sans drafter, comme deepseek.sh
voulue() {
    local t=${T:-$COEURS}
    [ -z "${TB:-}" ] || [ "$TB" = "$t" ] || t=$t/$TB
    echo "$t"
}
# les nombres de threads à mesurer : a = a threads, a/b = a pour un token seul et b pour les lots de tokens
if [ -z "${THREADS:-}" ]; then
    THREADS=$COEURS
    [ "$LOGIQUES" -le "$COEURS" ] || THREADS="$COEURS $LOGIQUES"
    # les threads réglés pour deepseek.sh se mesurent aussi
    for v in "$(voulue "")" "$(voulue drafter)"; do
        case " $THREADS " in
            *" $v "*) ;;
            *) THREADS="$v $THREADS" ;;
        esac
    done
fi
# reference <drafter ou vide> : les threads de la passe de référence, ceux de deepseek.sh s'ils sont parmi
# les nombres mesurés, sinon le premier
reference() {
    local v
    v=$(voulue "$1")
    case " $THREADS " in
        *" $v "*) echo "$v" ;;
        *)        echo "${THREADS%% *}" ;;
    esac
}
# en_clair <threads> : « 32 threads », ou « 32 threads et 64 pour les lots de tokens »
en_clair() {
    case $1 in
        */*) echo "${1%%/*} threads et ${1#*/} pour les lots de tokens" ;;
        *)   echo "$1 threads" ;;
    esac
}
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
RAPIDE=${RAPIDE:-0}
if [ "$RAPIDE" = 1 ]; then
    VARS=${VARS-GGML_AMX=0 GGML_AMX_ATTN=1 UBATCH=512}
    ESSAIS=${ESSAIS:-2}
    LECTURES=${LECTURES:-1}
fi
VARS=${VARS-GGML_AMX=0 GGML_AMX_ATTN=1 UBATCH=512 UBATCH=4096 GGML_INDEXER_512=0 GGML_FA_ROW_GROUPS=-1 GGML_LAZY_CONCAT=0 GGML_FA_SKIP_MASKED=0 GGML_SHARE_ROWS=0 GGML_HC_PROJ=0 GGML_SERIAL_NODES=0 IQK_PREFETCH=0}
CONTEXTE=${CONTEXTE:-16000}
CTX=${CTX:-131072}
UBATCH=${UBATCH:-2048}
SPEC=${SPEC:-dspark:n_max=4}
API_KEY=mesure         # clé du serveur que ce script lance pour lui seul, sur 127.0.0.1
PORT=${PORT_MESURE:-8080}
ESSAIS=${ESSAIS:-5}
LECTURES=${LECTURES:-3}
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
    [[ $n =~ ^[1-9][0-9]*(/[1-9][0-9]*)?$ ]] || { echo "THREADS=$THREADS : des nombres de threads séparés par des espaces, 32 ou 32/64."; exit 1; }
done
for n in "$ESSAIS" "$LECTURES" "$N_PREDICT"; do
    [[ $n =~ ^[1-9][0-9]*$ ]] || { echo "ESSAIS, LECTURES et N_PREDICT sont des nombres, au moins 1."; exit 1; }
done
[ "$LECTURES" -le "$ESSAIS" ] || LECTURES=$ESSAIS
IK_DIR=$(cd "$IK_DIR" && pwd)

PID=
arreter() {
    if [ -n "$PID" ]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=
    fi
}
# conclure : ce que le tableau dit, pour n'avoir rien à y chercher. Les colonnes d'une ligne : 1 threads,
# 2 drafter, 3 contexte, 4 réglage changé, 6 lecture, 7 sortie.
conclure() {
    printf '%s\n' "${RESUME[@]}" | awk '
        function nombre(x) { return x ~ /^[0-9]+(\.[0-9]+)?$/ }
        function ecart(a, b) { return sprintf("%+.1f %%", 100*(a - b)/b) }
        # retenir : un chiffre de plus pour ce cas, et le plus haut
        function retenir(cas, t, v) {
            if (!nombre(v)) return
            if (!(cas in n)) ordre[++nc] = cas
            n[cas]++
            threads[cas, n[cas]] = t
            chiffre[cas, n[cas]] = v
            if (n[cas] == 1 || v + 0 > haut[cas] + 0) { haut[cas] = v; qui[cas] = t }
        }
        { ligne[NR] = $0 }
        $4 == "-" {
            mode = ($2 == "oui" ? "avec le drafter" : "sans drafter")
            if ($3 == "court") {
                retenir(mode ", question courte, tokens/s produits", $1, $7)
            } else {
                retenir(mode ", texte long, tokens/s lus", $1, $6)
                retenir(mode ", après le texte long, tokens/s produits", $1, $7)
            }
            ref_lecture[$1, $2, $3] = $6
            ref_sortie[$1, $2, $3] = $7
        }
        END {
            titre = 0
            for (i = 1; i <= nc; i++) {
                cas = ordre[i]
                if (n[cas] < 2) continue
                if (!titre) {
                    print ""
                    print "=== le plus rapide des nombres de threads, colonne par colonne (passes sans réglage changé)"
                    titre = 1
                }
                autres = ""
                for (j = 1; j <= n[cas]; j++) {
                    if (threads[cas, j] == qui[cas]) continue
                    autres = autres (autres == "" ? "" : ", ") chiffre[cas, j] " avec " threads[cas, j]
                }
                printf "%s : %s avec %s threads (%s)\n", cas, haut[cas], qui[cas], autres
            }
            if (titre) {
                print "a/b : a threads pour un token seul, b pour les lots de tokens."
                print "Pour que deepseek.sh prenne un de ces nombres : T=a dans ~/.deepseek.conf, et TB=b pour a/b."
            }
            titre = 0
            for (i = 1; i <= NR; i++) {
                split(ligne[i], f, " ")
                if (f[4] == "-") continue
                cle = f[1] SUBSEP f[2] SUBSEP f[3]
                if (!(cle in ref_sortie)) continue
                lecture = (nombre(f[6]) && nombre(ref_lecture[cle]))
                sortie = (nombre(f[7]) && nombre(ref_sortie[cle]))
                if (!lecture && !sortie) continue
                if (!titre) {
                    print ""
                    print "=== un réglage changé à la fois, à côté de la passe de référence (mêmes threads, même contexte)"
                    titre = 1
                }
                printf "%s, %s :", f[4], (f[2] == "oui" ? "avec le drafter" : "sans drafter")
                if (lecture) printf " lu à %s au lieu de %s (%s)", f[6], ref_lecture[cle], ecart(f[6], ref_lecture[cle])
                if (lecture && sortie) printf " ;"
                if (sortie) printf " produit à %s au lieu de %s (%s)", f[7], ref_sortie[cle], ecart(f[7], ref_sortie[cle])
                printf "\n"
            }
        }'
}

# à la fin, ou à l'arrêt par Ctrl+C : le serveur s'arrête et le résumé des passes faites s'écrit
RESUME=()
finir() {
    arreter
    [ ${#RESUME[@]} -gt 0 ] || return 0
    {
        echo ""
        echo "=== résumé, une ligne par passe"
        echo "lus : tokens du texte long ; lecture : tokens/s lus, moyenne des lectures ; sortie : tokens/s produits,"
        echo "moyenne, plus bas et plus haut des essais ; draft : part acceptée des tokens du drafter sur la passe ;"
        echo "ms/pas : durée d'un pas du drafter (sa proposition et la vérification) ; tok/pas : tokens produits par pas."
        echo "Avec le drafter, sortie = 1000 x tok/pas / ms/pas. threads a/b : a pour un token seul, b pour les lots."
        printf '%7s  %-7s  %-8s  %-22s  %6s  %8s  %7s  %7s  %7s  %5s  %7s  %7s\n' threads drafter contexte variante lus lecture sortie min max draft ms/pas tok/pas
        printf '%s\n' "${RESUME[@]}"
        conclure
    } | tee -a "$OUT"
    RESUME=()
}
trap finir EXIT
trap 'echo ""; echo "== Arrêt demandé."; exit 130' INT TERM

# dire <texte> : à l'écran et dans le fichier de résultat
dire() { echo "$@" | tee -a "$OUT"; }

# les passes : la référence de chaque cas, puis les autres nombres de threads, puis les réglages
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
# durée : par passe le chargement et les essais (20 tokens/s), plus les lectures du texte long (250 tokens/s)
S_COURTE=$(( 40 + ESSAIS*N_PREDICT/20 ))
S_LONGUE=$(( S_COURTE + LECTURES*CONTEXTE/250 ))
if [ "$CONTEXTE" -gt 0 ]; then
    S_TOTAL=$(( ${#DRAFTERS[@]} * (N_THREADS*(S_COURTE + S_LONGUE) + N_VARS*S_LONGUE) ))
else
    S_TOTAL=$(( N_PASSES*S_COURTE ))
fi
if [ "$S_TOTAL" -ge 5400 ]; then
    DUREE="$(( (S_TOTAL + 900)/1800 / 2 )) h$([ $(( (S_TOTAL + 900)/1800 % 2 )) -eq 1 ] && echo " 30" || true)"
else
    DUREE="$(( (S_TOTAL + 150)/300 * 5 )) minutes"
fi

{
    echo "=== machine"
    "$IK_DIR/llama-server" --version 2>&1 | head -1
    lscpu | grep -E "^(Model name|Socket|Core|Thread)" || true
    # ce dont dépendent les prochaines optimisations : AMX (noyau et processeur), pages de 2 Mo du RAM disk
    echo "noyau : $(uname -r) ; instructions : $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' |
        grep -E '^(amx_.*|avx512_(vnni|bf16|fp16)|avx_vnni)$' | tr '\n' ' ')"
    echo "RAM disk du modèle : $(findmnt -n -o OPTIONS -T "$MODEL" 2>/dev/null || echo inconnu)"
    grep -E "^(ShmemHugePages|ShmemPmdMapped)" /proc/meminfo | tr -s ' ' | paste -sd ';' || true
    echo "threads : $THREADS ; drafter : $NOM_DRAFT ; texte long : $CONTEXTE tokens ; lots de $UBATCH tokens"
    echo "passes de référence : $(reference "") threads sans drafter$([ -z "$DRAFT" ] || echo ", $(reference "$DRAFT") avec")"
    echo "essais par passe : $ESSAIS, dont $LECTURES lectures d'un texte neuf ; réglages : ${VARS:-aucun}"
} > "$OUT" 2>&1
echo "== Résultat dans $OUT"
echo "== $N_PASSES passes, environ $DUREE : threads $THREADS, drafter $NOM_DRAFT, $ESSAIS essais par passe"
[ "$RAPIDE" = 1 ] || [ "$S_TOTAL" -lt 2400 ] || echo "   (RAPIDE=1 : la version courte ; Ctrl+C écrit le résumé des passes faites)"

# les sujets des essais, la question et le titre : l'essai i prend le sujet i, puis on recommence au premier.
# Le premier est celui de toutes les mesures faites jusqu'à rhel8-b5374.
SUJETS=("the history of Paris" "how a steam engine works" "the life of a honey bee" "the rules of chess"
        "the making of bread" "the water cycle" "the building of a cathedral" "the travels of Marco Polo")
TITRES=("The History of Paris" "How a Steam Engine Works" "The Life of a Honey Bee" "The Rules of Chess"
        "The Making of Bread" "The Water Cycle" "The Building of a Cathedral" "The Travels of Marco Polo")

# remplir <tokens> : un texte d'à peu près cette longueur, en paragraphes numérotés (85 tokens environ chacun),
# déjà échappé pour JSON
remplir() {
    local n=$1 i=1
    while [ $((i*85)) -le "$n" ]; do
        printf 'Note %d. The archive of the city keeps the registers of the guilds, the maps of the river and the letters of the merchants who crossed the bridge every morning before the market opened. Each volume was copied by hand, checked against the older one and stored on the upper floor, away from the damp of the cellars.\\n\\n' "$i"
        i=$((i+1))
    done
}

# resumer <threads> <drafter ou vide> <contexte> <réglage> <tokens lus> <lecture> <sortie> <min> <max> <acceptés>
# <ms par pas> <tokens par pas> : la ligne de la passe dans le tableau final
resumer() {
    RESUME+=("$(printf '%7s  %-7s  %-8s  %-22s  %6s  %8s  %7s  %7s  %7s  %5s  %7s  %7s' "$1" "$([ -n "$2" ] && echo oui || echo non)" \
        "$([ "$3" -gt 0 ] && echo long || echo court)" "${4:--}" "$5" "$6" "$7" "$8" "$9" "${10}" "${11:--}" "${12:--}")")
}

# stats <valeurs> : moyenne, plus bas, plus haut, nombre
stats() {
    printf '%s\n' "$@" | awk 'NR == 1 { mn = $1; mx = $1 } { s += $1; if ($1 < mn) mn = $1; if ($1 > mx) mx = $1 }
                              END { printf "%.2f %.2f %.2f %d", s/NR, mn, mx, NR }'
}

N_PASSE=0
# passe <threads, a ou a/b> <drafter ou vide> <tokens de texte avant la question, ou 0> [VARIABLE=valeur]
passe() {
    local t=$1 draft=$2 contexte=$3 var=${4:-} pret=0 v i rep texte invite
    local lu="-" lecture="-" moyenne="-" accepte="-" ms_pas="-" tok_pas="-"
    local s pn pm stat pas gen acc dur tok_total=0 ms_total=0
    N_PASSE=$((N_PASSE + 1))
    local args=(-m "$MODEL" -t "${t%%/*}" -c "$CTX" -cram 100000 --jinja --swa-compress
                --api-key "$API_KEY" --host 127.0.0.1 --port "$PORT")
    # a/b : b threads pour les lots de plusieurs tokens
    case $t in
        */*) args+=(-tb "${t#*/}") ;;
    esac
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
        dire "=== $(en_clair "$t"), drafter : $([ -n "$draft" ] && echo "$NOM_DRAFT" || echo aucun), contexte long${var:+, $var}"
    else
        dire "=== $(en_clair "$t"), drafter : $([ -n "$draft" ] && echo "$NOM_DRAFT" || echo aucun)${var:+, $var}"
    fi
    ( cd "$IK_DIR" && exec env GGML_PROFILE=1 $var numactl --interleave=all ./llama-server "${args[@]}" > "$LOG" 2>&1 ) &
    PID=$!
    echo "   serveur lancé (PID $PID), chargement..."
    for i in $(seq 1 300); do
        if ! kill -0 "$PID" 2>/dev/null; then
            dire "Le serveur s'est arrêté pendant le chargement. Fin du journal :"
            tail -5 "$LOG" | tee -a "$OUT"
            PID=
            resumer "$t" "$draft" "$contexte" "$var" "-" "-" "échec" "-" "-" "-"
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
        resumer "$t" "$draft" "$contexte" "$var" "-" "-" "échec" "-" "-" "-"
        return 1
    fi

    # la requête passe par un fichier : un texte long ne tient pas sur une ligne de commande
    invite=/tmp/profil-invite-$$.json
    local valeurs=() lectures=() k l mini="-" maxi="-" n
    for i in $(seq 1 "$ESSAIS"); do
        # après le texte long, les LECTURES premiers essais lisent chacun un texte neuf : sa première ligne
        # change, le serveur ne retrouve donc rien dans son cache. Les essais suivants reprennent le dernier.
        k=$(( i < LECTURES ? i : LECTURES ))
        s=$(( (i - 1) % ${#SUJETS[@]} ))
        {
            printf '{"prompt":"'
            if [ "$contexte" -gt 0 ]; then
                printf 'Volume %d.\\n\\n' "$k"
                remplir "$contexte"
            fi
            printf 'Write a long essay about %s.\\n\\n' "${SUJETS[$s]}"
            # after a long text, the title of the essay: without it the model goes on with the text
            [ "$contexte" -le 0 ] || printf '# %s\\n\\n' "${TITRES[$s]}"
            printf '","n_predict":%d,"temperature":0,"cache_prompt":true}' "$N_PREDICT"
        } > "$invite"
        rep=$(curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" \
                -d @"$invite" || true)
        v=$(echo "$rep" | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2 || true)
        # pour la durée d'un pas du drafter : les tokens produits et le temps passé à les produire, sur la passe
        pn=$(echo "$rep" | grep -o '"predicted_n":[0-9]*' | head -1 | cut -d: -f2 || true)
        pm=$(echo "$rep" | grep -o '"predicted_ms":[0-9.]*' | head -1 | cut -d: -f2 || true)
        if [ -n "$pn" ] && [ -n "$pm" ]; then
            tok_total=$((tok_total + pn))
            ms_total=$(awk -v a="$ms_total" -v b="$pm" 'BEGIN { printf "%.3f", a + b }')
        fi
        if [ "$contexte" -gt 0 ] && [ "$i" -le "$LECTURES" ]; then
            lu=$(echo "$rep" | grep -o '"prompt_n":[0-9]*' | head -1 | cut -d: -f2 || true)
            l=$(echo "$rep" | grep -o '"prompt_per_second":[0-9.]*' | head -1 | cut -d: -f2 || true)
            if [ -n "$l" ]; then
                dire "texte lu : ${lu:-?} tokens, $(printf '%.1f' "$l") tokens/s"
                lectures+=("$l")
            else
                dire "texte lu : pas de réponse"
            fi
            lu=${lu:-?}
        fi
        if [ -z "$v" ]; then
            dire "essai $i : pas de réponse"
        else
            dire "essai $i : $(printf '%.2f' "$v") tokens/s (${SUJETS[$s]})"
            valeurs+=("$v")
        fi
        if [ "$i" = 1 ]; then
            # le premier texte produit : le même d'une passe à l'autre si le calcul est le même
            texte=$(echo "$rep" | grep -o '"content":"\(\\.\|[^"\\]\)*"' | head -1 | cut -c12- | sed 's/"$//' || true)
            dire "texte : $(echo "$texte" | cut -c1-150)"
            dire "empreinte du texte : $(echo "$texte" | md5sum | cut -c1-12)"
        fi
    done
    if [ ${#lectures[@]} -gt 0 ]; then
        read -r lecture mini maxi n <<< "$(stats "${lectures[@]}")"
        dire "lecture, moyenne sur $n essais : $lecture tokens/s (de $mini à $maxi)"
        lecture=$(printf '%.1f' "$lecture")
    fi
    mini="-"; maxi="-"
    if [ ${#valeurs[@]} -gt 0 ]; then
        read -r moyenne mini maxi n <<< "$(stats "${valeurs[@]}")"
        dire "moyenne sur $n essais : $moyenne tokens/s (de $mini à $maxi)"
    else
        moyenne="échec"
    fi

    # Les pages de 2 Mo du RAM disk ne servent que si le serveur les a prises entières (ShmemPmdMapped) et non
    # par morceaux de 4 ko : cela ne se lit que pendant qu'il tourne, une fois les poids parcourus.
    if [ -z "${PAGES_DITES:-}" ]; then
        PAGES_DITES=1
        dire "pages du modèle, serveur en marche : $(grep -E '^(ShmemHugePages|ShmemPmdMapped)' /proc/meminfo | tr -s ' ' | paste -sd ';')"
    fi
    arreter   # le tableau est écrit à l'arrêt du serveur
    grep -E "draft acceptance rate|statistics ${SPEC%%:*}:" "$LOG" | tail -2 >> "$OUT" || true
    if [ -n "$draft" ]; then
        accepte=$(grep -o "draft acceptance rate = [0-9.]*" "$LOG" | tail -1 | awk '{ printf "%.0f%%", 100*$NF }' || true)
        accepte=${accepte:--}
        # Sur toute la passe : les pas du drafter, ce qu'ils durent et ce qu'ils donnent. Les tokens/s en sont le
        # quotient ; la durée d'un pas tient au réglage, les tokens par pas à ce que le modèle accepte du texte.
        stat=$(grep "statistics ${SPEC%%:*}:" "$LOG" | tail -1 || true)
        pas=$(echo "$stat" | grep -o '#gen drafts = [0-9]*' | grep -o '[0-9]*$' || true)
        gen=$(echo "$stat" | grep -o '#gen tokens = [0-9]*' | grep -o '[0-9]*$' || true)
        acc=$(echo "$stat" | grep -o '#acc tokens = [0-9]*' | grep -o '[0-9]*$' || true)
        # dur(b,g,a) : le deuxième temps est celui que le drafter a mis à proposer
        dur=$(echo "$stat" | sed -n 's/.*dur(b,g,a) = [0-9.]*, \([0-9.]*\),.*/\1/p')
        if [ "${pas:-0}" -gt 0 ] && [ "$tok_total" -gt 0 ]; then
            ms_pas=$(awk -v m="$ms_total" -v p="$pas" 'BEGIN { printf "%.1f", m/p }')
            tok_pas=$(awk -v n="$tok_total" -v p="$pas" 'BEGIN { printf "%.2f", n/p }')
            [ "${gen:-0}" -le 0 ] || accepte=$(awk -v a="${acc:-0}" -v g="$gen" 'BEGIN { printf "%.0f%%", 100*a/g }')
            dire "drafter : $pas pas de $ms_pas ms$(awk -v d="${dur:-0}" -v p="$pas" 'BEGIN { if (d > 0) printf " dont %.1f pour proposer", d/p }') ;" \
                 "par pas $(awk -v g="${gen:-0}" -v a="${acc:-0}" -v p="$pas" 'BEGIN { printf "%.2f tokens proposés, %.2f acceptés", g/p, a/p }')" \
                 "($accepte), $tok_pas produits"
        fi
    fi
    resumer "$t" "$draft" "$contexte" "$var" "$lu" "$lecture" "$moyenne" "$mini" "$maxi" "$accepte" "$ms_pas" "$tok_pas"
    if grep -q "^ggml_profile:" "$LOG"; then
        grep "^ggml_profile:" "$LOG" >> "$OUT"
        echo "   chronométrage enregistré"
    else
        dire "Pas de chronométrage dans le journal : cette version est-elle assez récente ?"
    fi
    rm -f "$LOG" "$invite"
}

# une passe qui échoue n'empêche pas les suivantes
# 1. la passe de référence de chaque cas : ce que deepseek.sh lance
for draft in "${DRAFTERS[@]}"; do
    for contexte in $CONTEXTES; do
        passe "$(reference "$draft")" "$draft" "$contexte" || true
    done
done
# 2. les autres nombres de threads
for t in $THREADS; do
    for draft in "${DRAFTERS[@]}"; do
        [ "$t" != "$(reference "$draft")" ] || continue
        for contexte in $CONTEXTES; do
            passe "$t" "$draft" "$contexte" || true
        done
    done
done
# 3. un réglage changé à la fois, après le texte long (la question courte si CONTEXTE=0)
for var in $VARS; do
    for draft in "${DRAFTERS[@]}"; do
        passe "$(reference "$draft")" "$draft" "${CONTEXTES##* }" "$var" || true
    done
done

finir
echo "== Terminé. Envoyez le fichier $OUT"
