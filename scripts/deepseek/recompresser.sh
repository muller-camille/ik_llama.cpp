#!/bin/bash
# Recompresse les tenseurs Q8_0 lus à chaque token (attention, expert partagé, sortie) et installe le
# résultat en RAM À CÔTÉ du modèle actuel : rien n'est modifié ni supprimé, on choisit ensuite l'un ou
# l'autre avec -m.
#
#   ./recompresser.sh                               Q6_K (le pas le plus prudent)
#   TYPE=q5_k_r4 SUF=R8Q5 ./recompresser.sh         Q5_K
#   TYPE=q4_k_r4 SUF=R8Q4 ./recompresser.sh         Q4_K
#
# Le script utilise la version dans laquelle il se trouve ; un dossier ik_llama en premier argument en
# choisit une autre. Il lui faut NFS, le dossier des fichiers convertis (dans ~/.deepseek.conf ou en variable).
#
# Seul le fichier 9 est réécrit (7,3 Go en Q6_K au lieu de 9,2) ; les fichiers 1 à 8 sont partagés avec
# le modèle actuel par des liens, sans mémoire supplémentaire. Durée : quelques minutes.
# L'effet sur la qualité des réponses n'est pas mesuré : comparez vos propres requêtes avant de l'adopter.
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
TYPE=${TYPE:-q6_k_r4}
SUF=${SUF:-R8Q6}
NFS=${NFS:?"Indiquez le dossier des fichiers convertis : NFS=... dans ~/.deepseek.conf ou devant la commande"}
DS0=${DS0:-/mnt/ds0}
DS1=${DS1:-/mnt/ds1}
P=${P:-DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8}
THREADS=${THREADS:-$(nproc)}
OLD=$P
NEW=${P%-R8}-$SUF

# tous les tenseurs Q8_0 du fichier 9, sauf la table d'entrée token_embd
RX='^(blk\.[0-9]+\.(attn_kv|attn_q_[ab]|attn_output_[ab]|ffn_(up|gate|down)_shexp|engram_kv|attn_compressor_(gate|kv)|indexer\.(proj|attn_k|attn_q_b))|output)\.weight$'

f() { printf '%s-%05d-of-00009.gguf' "$1" "$2"; }   # f <préfixe> <numéro> : nom d'un fichier du jeu

# --- vérifications ---------------------------------------------------------
[ -x "$IK_DIR/llama-quantize" ] || { echo "llama-quantize introuvable dans $IK_DIR"; exit 1; }
for n in 1 2 3 4 5 6 7 8 9; do
    [ -e "$DS0/$(f "$OLD" $n)" ] || { echo "Fichier introuvable : $DS0/$(f "$OLD" $n) (le modèle est-il chargé en RAM ?)"; exit 1; }
    [ -f "$NFS/$(f "$OLD" $n)" ] || { echo "Fichier introuvable : $NFS/$(f "$OLD" $n) (le NFS est-il monté ?)"; exit 1; }
done
libre() { df --output=avail -BG "$1" | tail -1 | tr -dc '0-9'; }
for d in "$NFS" "$DS0" "$DS1"; do
    [ "$(libre "$d")" -ge 10 ] || { echo "Moins de 10 Go libres dans $d : il en faut une dizaine pour le nouveau fichier."; exit 1; }
done

# --- recompression : seul le fichier 9 est écrit ---------------------------
echo "== Liens vers les fichiers 1 à 8 dans $NFS"
for n in 1 2 3 4 5 6 7 8; do
    ln -sfn "$(f "$OLD" $n)" "$NFS/$(f "$NEW" $n)"
done
if [ -f "$NFS/$(f "$NEW" 9)" ]; then
    echo "== $NFS/$(f "$NEW" 9) existe déjà, je le garde (supprimez-le pour le refaire)."
else
    echo "== Recompression en $TYPE (journal : $NFS/$NEW.log)"
    ( cd "$IK_DIR" && ./llama-quantize --allow-requantize --keep-split --partial-requant \
        --custom-q "$RX=$TYPE" "$DS0/$(f "$OLD" 1)" "$NFS/$NEW.gguf" COPY "$THREADS" ) > "$NFS/$NEW.log" 2>&1 \
        || { echo "La recompression a échoué. Fin du journal :"; tail -5 "$NFS/$NEW.log"; exit 1; }
fi
[ -f "$NFS/$(f "$NEW" 9)" ] || { echo "Le fichier 9 n'a pas été produit. Fin du journal :"; tail -5 "$NFS/$NEW.log"; exit 1; }
if [ -f "$NFS/$NEW.log" ]; then
    echo "== Tenseurs du fichier 9 (type d'origine -> nouveau type) :"
    grep -oE "type = +[a-z0-9_]+, (converting to [a-z0-9_]+|size)" "$NFS/$NEW.log" \
        | sed 's/, size/ -> inchangé/; s/, converting to/ ->/' | sort | uniq -c || true
fi
ls -l "$NFS/$(f "$OLD" 9)" "$NFS/$(f "$NEW" 9)" | awk '{ printf "   %6.2f Go  %s\n", $5/1e9, $NF }'

# --- installation en RAM, à côté du modèle actuel --------------------------
echo "== Installation en RAM"
for n in 1 2 3 4 5 6 7; do
    ln -f "$DS0/$(f "$OLD" $n)" "$DS0/$(f "$NEW" $n)"     # lien physique : aucune mémoire en plus
    ln -f "$DS1/$(f "$OLD" $n)" "$DS1/$(f "$NEW" $n)"
done
ln -sfn "$(readlink -f "$DS0/$(f "$OLD" 8)")" "$DS0/$(f "$NEW" 8)"   # la table reste partagée
rsync -a --size-only --inplace --info=progress2 "$NFS/$(f "$NEW" 9)" "$DS0/"
rsync -a --size-only --inplace --info=progress2 "$DS0/$(f "$NEW" 9)" "$DS1/"

echo
echo "== Terminé. Le modèle recompressé se lance avec :"
echo "   -m $DS0/$(f "$NEW" 1) --numa-mirror $DS1"
echo "   ou, avec deepseek.sh : M=$NEW ./deepseek.sh"
echo "   Le modèle actuel reste disponible sous $DS0/$(f "$OLD" 1)"
