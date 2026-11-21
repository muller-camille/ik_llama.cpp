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
#   DRAFT    fichier du drafter dans NFS, vide = sans drafter
#   SPEC     réglage du drafter, dspark:n_max=4 par défaut
#   T        threads, un par cœur physique par défaut
#   CTX      contexte, 131072 par défaut
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
# un thread par cœur physique
COEURS=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$COEURS" -gt 0 ] 2>/dev/null || COEURS=$(nproc)
NFS=${NFS:?"Indiquez le dossier des fichiers convertis : NFS=... dans ~/.deepseek.conf ou devant la commande"}
P=${P:-DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8}
M=${M:-$P}
DRAFT=${DRAFT-dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf}
SPEC=${SPEC:-dspark:n_max=4}
API_KEY=${API_KEY:-}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
T=${T:-$COEURS}
CTX=${CTX:-131072}

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
      -t "$T" -c "$CTX" -cram 100000 --jinja --swa-compress --host "$HOST" --port "$PORT")
[ -z "$API_KEY" ] || ARGS+=(--api-key "$API_KEY")
[ -z "$DRAFT" ]   || ARGS+=(-md "/mnt/ds0/$DRAFT" --spec-type "$SPEC")
echo "== Démarrage du serveur sur $HOST:$PORT, $T threads, modèle $M (Ctrl+C pour l'arrêter)"
cd "$IK_DIR"
exec numactl --interleave=all ./llama-server "${ARGS[@]}"
