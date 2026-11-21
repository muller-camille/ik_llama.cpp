#!/bin/bash
# Recharge DeepSeek-V4.1 (fichiers convertis + drafter) en RAM depuis le dossier NFS, une copie des poids
# par socket. À lancer en root après un redémarrage : sudo ./charger-deepseek.sh
#
# Réglages, dans ~/.deepseek.conf ou en variables : NFS (dossier des fichiers, obligatoire), P (préfixe du
# modèle), DRAFT (fichier du drafter dans NFS, vide = aucun).
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
DRAFT=${DRAFT-dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf}

[ "$(id -u)" -eq 0 ] || { echo "À lancer en root (sudo)."; exit 1; }
for f in "$NFS/$P-00001-of-00009.gguf" "$NFS/$P-00008-of-00009.gguf" ${DRAFT:+"$NFS/$DRAFT"}; do
    [ -f "$f" ] || { echo "Fichier introuvable : $f (le NFS est-il monté ?)"; exit 1; }
done

# monter <dossier> <options> : ne monte que si rien n'est déjà monté à cet endroit
monter() {
    mkdir -p "$1"
    if mountpoint -q "$1"; then
        echo "$1 est déjà monté, je le garde tel quel."
    else
        mount -t tmpfs -o "$2" tmpfs "$1"
    fi
}
monter /mnt/ramdisk size=200G,huge=always,mpol=interleave
monter /mnt/ds0     size=300G,huge=always,mpol=bind:0
monter /mnt/ds1     size=300G,huge=always,mpol=bind:1
mkdir -p /mnt/ramdisk/model

echo "== Poids et drafter vers le socket 0"
rsync -a --size-only --inplace --info=progress2 "$NFS"/$P-0000[1-79]-of-00009.gguf ${DRAFT:+"$NFS/$DRAFT"} /mnt/ds0/
echo "== Copie vers le socket 1"
rsync -a --size-only --inplace --info=progress2 /mnt/ds0/$P-0000[1-79]-of-00009.gguf ${DRAFT:+"/mnt/ds0/$DRAFT"} /mnt/ds1/
echo "== Table engram"
rsync -a --size-only --inplace --info=progress2 "$NFS/$P-00008-of-00009.gguf" /mnt/ramdisk/model/
ln -sfn "/mnt/ramdisk/model/$P-00008-of-00009.gguf" "/mnt/ds0/$P-00008-of-00009.gguf"

sync
echo 3 > /proc/sys/vm/drop_caches
echo "== Terminé. Mémoire partagée par socket :"
grep -H "Shmem:" /sys/devices/system/node/node*/meminfo
free -g
