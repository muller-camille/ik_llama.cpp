#!/bin/bash
# DeepSeek-V4.1 : recharge le modèle en RAM depuis le dossier NFS si nécessaire, puis lance le serveur.
# À lancer avec votre compte habituel, pas en root :   ./deepseek.sh
# Le script n'utilise sudo que pour les montages et le vidage du cache.
#
# Réglages, dans ~/.deepseek.conf ou en variables :
#   NFS      dossier des fichiers convertis et du drafter (obligatoire)
#   API_KEY  clé demandée aux clients (obligatoire si HOST n'est pas 127.0.0.1)
#   HOST     127.0.0.1 par défaut (cette machine seulement) ; 0.0.0.0 = toutes les interfaces réseau
#   PORT     8080
#   P        préfixe du modèle à charger, DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8 par défaut
#   M        préfixe du modèle à lancer, P par défaut ; ...-R8Q6 pour le modèle de recompresser.sh
#   DRAFT    fichier du drafter dans NFS, vide = sans drafter ; par défaut dspark-...-Q8_0-R8.gguf s'il y
#            est (le drafter converti, lu plus vite), sinon dspark-...-Q8_0.gguf
#   SPEC     réglage du drafter, dspark:n_max=4 par défaut : 4 tokens au plus par brouillon, coupé au premier
#            token dont le drafter est sûr à moins de 75 % (p_min). Mesuré avec rhel8-b5399 sur cinq sujets :
#            30,4 tokens/s ; p_min=0.5, 0.3 ou 0 donnent 30,8, 30,1 et 30,6, n_max=5 30,1 (p_min=0.3) et 28,8
#            (p_min=0). Aucun n'est plus rapide : un pas donne plus de tokens et dure d'autant plus longtemps.
#   T        threads. Par défaut un par cœur physique, avec ou sans drafter : c'est ce que profil.sh a mesuré
#            de plus rapide sur 2 x 16 cœurs à hyperthreading (rhel8-b5379, moyenne de cinq sujets). Sans
#            drafter, 22,9 tokens/s à 32 threads contre 21,5 à 64 ; avec, 29,6 contre 28,3, et après un prompt
#            de 12 600 tokens 24,2 contre 23,6 : un pas du drafter dure 80 ms à 32 threads et 83 à 64. Le
#            prompt se lit à la même vitesse (361 et 358 tokens/s). Avec le drafter le résultat tient au
#            texte : sur le sujet des mesures précédentes, 64 threads restent devant (31,5 contre 29,8, et
#            33,4 contre 30,9 avec rhel8-b5371), sur trois des quatre autres c'est 32.
#            profil.sh refait ces mesures sur votre machine et dit le plus rapide colonne par colonne.
#   TB       threads quand le modèle calcule plusieurs tokens ensemble (lecture d'un prompt, vérification
#            des tokens du drafter), option -tb du serveur ; T par défaut. À laisser égal à T avec un
#            drafter : mesuré à 32/64, un pas dure 94 ms au lieu de 83, le temps de réveiller les threads.
#   CTX      contexte, 131072 par défaut. Les conversations gardées dans CACHE_DIR ne tiennent pas à cette
#            taille : après un changement de CTX, le serveur reprend celles qui tiennent dans le nouveau
#            contexte et laisse les autres dans le dossier, pour un serveur au contexte plus grand.
#   UBATCH   tokens du prompt lus ensemble (option -ub du serveur), 2048 par défaut. Avec --swa-compress, les
#            19 dernières couches de DeepSeek-V4.1 ne sont calculées que sur les 256 derniers tokens de chaque
#            lot : plus le lot est grand, moins de tokens du prompt y passent. Mesuré sur 2 x 16 cœurs avec un
#            prompt de 15761 tokens : 245 tokens/s par lots de 512, 329 par lots de 2048. La vitesse du texte
#            produit ne change pas. UBATCH=512 revient au réglage du serveur.
#   WEBUI    page web du serveur : llamacpp (celle de llama.cpp, par défaut), auto (celle d'ik_llama.cpp)
#            ou none (aucune)
#   CACHE_DIR  dossier où le serveur garde les conversations qu'il met de côté (son cache de prompts, limité
#            à 100000 Mio). Vide par défaut : elles restent dans la mémoire du serveur et se perdent quand
#            il s'arrête. Avec CACHE_DIR=/mnt/dscache/cache elles vont dans des fichiers sur un RAM disk,
#            /mnt/dscache, que ce script monte pour votre compte seul : le serveur les retrouve à son
#            redémarrage. À son arrêt (un seul Ctrl+C : il finit d'abord la réponse en cours, un deuxième
#            l'arrête sans rien écrire) il y ajoute la conversation en cours. S'il est tué ou plante, une
#            conversation revient dans l'état où elle a été écrite la dernière fois, et la suite se relit.
#            Rien ne survit à un redémarrage de la machine.
#   CACHE_DISK taille du RAM disk /mnt/dscache, 120G par défaut (il n'occupe que ce qu'on y écrit). Quand
#            il est plein, le serveur retire les conversations les plus anciennes.
#   GGML_AMX_ATTN  1 = l'attention d'un prompt passe elle aussi par les tuiles AMX du processeur. Mesuré avec
#            rhel8-b5390 : un prompt de 12 616 tokens lu à 452 tokens/s au lieu de 359, la production de
#            texte inchangée. Ce n'est pas le réglage par défaut : le calcul y est arrondi à 8 bits
#            significatifs, le texte produit après un même prompt n'est plus le même, et la qualité des
#            réponses n'a pas été comparée. Ce n'est pas un réglage de ce script mais une variable du
#            serveur : comme toute ligne NOM=valeur de ~/.deepseek.conf, elle lui est passée telle quelle.
#
# Le serveur lancé est celui de la version dans laquelle se trouve ce script ; un dossier ik_llama en
# premier argument en choisit une autre.
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
# les cœurs physiques (la moitié des processeurs logiques avec l'hyperthreading)
COEURS=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$COEURS" -gt 0 ] 2>/dev/null || COEURS=$(nproc)
NFS=${NFS:?"Indiquez le dossier des fichiers convertis : NFS=... dans ~/.deepseek.conf ou devant la commande"}
P=${P:-DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8}
M=${M:-$P}
# le drafter : sa conversion (llama-quantize --repack ... Q8_0_R8) si elle est dans NFS, sinon l'original
if [ -z "${DRAFT+x}" ]; then
    DRAFT=dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
    [ ! -f "$NFS/${DRAFT%.gguf}-R8.gguf" ] || DRAFT=${DRAFT%.gguf}-R8.gguf
fi
SPEC=${SPEC:-dspark:n_max=4}
API_KEY=${API_KEY:-}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
# Les threads, d'après les mesures de rhel8-b5379 (cinq sujets par passe) : un par cœur physique, avec ou sans
# drafter. Un par processeur logique donne 21.5 tokens/s au lieu de 22.9 sans drafter et 28.3 au lieu de
# 29.6 avec ; TB différent de T perd 11 ms à chaque pas du drafter.
T=${T:-$COEURS}
TB=${TB:-}
CTX=${CTX:-131072}
UBATCH=${UBATCH:-2048}
WEBUI=${WEBUI:-llamacpp}
CACHE_DIR=${CACHE_DIR:-}
CACHE_DISK=${CACHE_DISK:-120G}

# --- vérifications ---------------------------------------------------------
[ "$(id -u)" -ne 0 ] || { echo "Lancez ce script avec votre compte habituel, pas en root."; exit 1; }
[ -x "$IK_DIR/llama-server" ] || { echo "llama-server introuvable dans $IK_DIR"; exit 1; }
case $HOST in
    127.0.0.1|localhost|::1) ;;
    *) [ -n "$API_KEY" ] || { echo "HOST=$HOST ouvre le serveur au réseau : indiquez API_KEY (dans ~/.deepseek.conf ou devant la commande)."; exit 1; } ;;
esac
for f in "$NFS/$P-00001-of-00009.gguf" "$NFS/$P-00008-of-00009.gguf" ${DRAFT:+"$NFS/$DRAFT"}; do
    [ -f "$f" ] || { echo "Fichier introuvable : $f (le NFS est-il monté ?)"; exit 1; }
done
sudo -v   # demande le mot de passe sudo une fois, dès le début

# --- RAM disks -------------------------------------------------------------
# monter <dossier> <options> : ne monte que si rien n'est déjà monté à cet endroit
monter() {
    sudo mkdir -p "$1"
    if mountpoint -q "$1"; then
        echo "$1 est déjà monté, je le garde tel quel."
    else
        sudo mount -t tmpfs -o "$2" tmpfs "$1"
    fi
}
monter /mnt/ramdisk size=200G,huge=always,mpol=interleave
monter /mnt/ds0     size=300G,huge=always,mpol=bind:0
monter /mnt/ds1     size=300G,huge=always,mpol=bind:1
mkdir -p /mnt/ramdisk/model
# le cache de prompts du serveur, s'il est demandé sur son RAM disk (le serveur crée le dossier lui-même)
case $CACHE_DIR in
    /mnt/dscache|/mnt/dscache/*) monter /mnt/dscache "size=$CACHE_DISK,mpol=interleave,mode=0700,uid=$(id -u),gid=$(id -g)" ;;
esac

# --- chargement (ne recopie que ce qui manque) -----------------------------
echo "== Poids et drafter vers le socket 0"
rsync -a --size-only --inplace --info=progress2 "$NFS"/$P-0000[1-79]-of-00009.gguf ${DRAFT:+"$NFS/$DRAFT"} /mnt/ds0/
echo "== Copie vers le socket 1"
rsync -a --size-only --inplace --info=progress2 /mnt/ds0/$P-0000[1-79]-of-00009.gguf ${DRAFT:+"/mnt/ds0/$DRAFT"} /mnt/ds1/
echo "== Table engram"
rsync -a --size-only --inplace --info=progress2 "$NFS/$P-00008-of-00009.gguf" /mnt/ramdisk/model/
ln -sfn "/mnt/ramdisk/model/$P-00008-of-00009.gguf" "/mnt/ds0/$P-00008-of-00009.gguf"

sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo "== Modèle en RAM. Mémoire partagée par socket :"
grep -H "Shmem:" /sys/devices/system/node/node*/meminfo
free -g

if [ ! -e "/mnt/ds0/$M-00001-of-00009.gguf" ]; then
    echo "Le modèle $M n'est pas en RAM. S'il vient de recompresser.sh, relancez recompresser.sh : il le réinstalle sans le recalculer."
    exit 1
fi

# --- serveur ---------------------------------------------------------------
ARGS=(-m "/mnt/ds0/$M-00001-of-00009.gguf" --numa-mirror /mnt/ds1
      -t "$T" -c "$CTX" -cram 100000 --jinja --swa-compress --host "$HOST" --port "$PORT" --webui "$WEBUI")
[ -z "$TB" ]      || ARGS+=(-tb "$TB")
[ -z "$API_KEY" ] || ARGS+=(--api-key "$API_KEY")
[ -z "$DRAFT" ]   || ARGS+=(-md "/mnt/ds0/$DRAFT" --spec-type "$SPEC")
[ -z "$CACHE_DIR" ] || ARGS+=(-cram-dir "$CACHE_DIR")
ARGS+=(-b "$((UBATCH > 2048 ? UBATCH : 2048))" -ub "$UBATCH")
echo "== Démarrage du serveur sur $HOST:$PORT, $T threads${TB:+ ($TB pour les lots de tokens)}, modèle $M (Ctrl+C pour l'arrêter)"
cd "$IK_DIR"
exec numactl --interleave=all ./llama-server "${ARGS[@]}"
