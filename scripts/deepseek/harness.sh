#!/bin/bash
# DeepSeek Harness (dsh, https://github.com/deepseek-ai/deepseek-harness) branché sur le serveur de deepseek.sh :
# l'agent, son interface web et sa ligne de commande travaillent avec le modèle local au lieu de l'API DeepSeek.
#
#   ./harness.sh                      interface web, sur 127.0.0.1:3080 (l'adresse à ouvrir est affichée)
#   ./harness.sh headless "tâche"     une tâche, la réponse sur la sortie, puis fin
#   ./harness.sh <profil> [options]   tout autre profil ou option de dsh (dsh --help)
#
# Le serveur (deepseek.sh) doit tourner. dsh demande Node.js 24.9 ou plus récent :
#   npm install -g @deepseek-ai/dsh
#
# ATTENTION : l'agent exécute sur cette machine, avec votre compte, les commandes que le modèle décide de
# lancer. Lancez-le dans un dossier de travail, avec un compte qui n'a que les droits voulus.
#
# Réglages, dans ~/.deepseek.conf ou en variables :
#   SERVEUR     adresse du serveur du modèle ; par défaut http://HOST:PORT, ceux de deepseek.sh
#               (127.0.0.1 si HOST n'est pas réglé ou vaut 0.0.0.0, et 8080)
#   API_KEY     la clé du serveur (celle de deepseek.sh)
#   CTX         contexte du serveur, 131072 par défaut : dsh résume la conversation avant de le dépasser
#   SORTIE      nombre maximal de tokens d'une réponse, 32768 par défaut
#   EFFORT      réflexion par défaut : off, low, high (par défaut) ou max ; se change aussi dans l'interface
#   H_HOST      adresse d'écoute de l'interface web, 127.0.0.1 par défaut (cette machine seulement)
#   H_PORT      port de l'interface web, 3080
#   DSH         commande qui lance dsh, « dsh » par défaut
#   TELEMETRIE  1 = laisse dsh envoyer sa télémétrie à DeepSeek ; par défaut elle est coupée
#
# Rien n'est écrit dans les profils de dsh : les réglages lui sont passés en surcouche (--patch), dans un
# fichier refait à chaque lancement. La clé n'y figure pas, elle passe par l'environnement.
set -euo pipefail

# --- réglages --------------------------------------------------------------
MAISON=$(getent passwd "${SUDO_USER:-$(id -un)}" 2>/dev/null | cut -d: -f6 || true)
CONF=${DEEPSEEK_CONF:-${MAISON:-$HOME}/.deepseek.conf}
if [ -f "$CONF" ]; then
    while IFS='=' read -r nom valeur || [ -n "$nom" ]; do
        [[ $nom =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [ -n "${!nom+x}" ] || export "$nom=$valeur"
    done < "$CONF"
fi
PORT=${PORT:-8080}
# le serveur écoute sur HOST ; s'il écoute sur toutes les interfaces, on le joint par celle de la machine
HOTE=${HOST:-127.0.0.1}
case $HOTE in
    0.0.0.0|::|localhost) HOTE=127.0.0.1 ;;
esac
SERVEUR=${SERVEUR:-http://$HOTE:$PORT}
SERVEUR=${SERVEUR%/}
SERVEUR=${SERVEUR%/v1}
API_KEY=${API_KEY:-}
CTX=${CTX:-131072}
SORTIE=${SORTIE:-32768}
EFFORT=${EFFORT:-high}
H_HOST=${H_HOST:-127.0.0.1}
H_PORT=${H_PORT:-3080}
DSH=${DSH:-dsh}
TELEMETRIE=${TELEMETRIE:-0}

# --- vérifications ---------------------------------------------------------
if [ "$(id -u)" -eq 0 ] && [ "${EN_ROOT:-0}" != 1 ]; then
    echo "Lancez ce script avec votre compte habituel, pas en root : l'agent exécute des commandes."
    echo "(EN_ROOT=1 passe outre, pour un conteneur.)"
    exit 1
fi
case $EFFORT in
    off|low|high|max) ;;
    *) echo "EFFORT=$EFFORT : off, low, high ou max."; exit 1 ;;
esac
for n in "$CTX" "$SORTIE"; do
    [[ $n =~ ^[0-9]+$ ]] || { echo "CTX et SORTIE sont des nombres de tokens."; exit 1; }
done
[ "$SORTIE" -lt "$CTX" ] || { echo "SORTIE ($SORTIE) doit rester sous CTX ($CTX)."; exit 1; }
# shellcheck disable=SC2086
if ! $DSH --version > /dev/null 2>&1; then
    echo "dsh ne se lance pas (commande : $DSH)."
    echo "Il lui faut Node.js 24.9 ou plus récent (ici : $(node --version 2>/dev/null || echo "pas de node"))."
    echo "Installation :  npm install -g @deepseek-ai/dsh"
    exit 1
fi
if ! curl -sf -m 5 "$SERVEUR/health" > /dev/null 2>&1; then
    echo "== ATTENTION : pas de réponse de $SERVEUR/health. Le serveur (deepseek.sh) tourne-t-il ?"
fi

# --- surcouche pour dsh ----------------------------------------------------
# Le serveur tient la place de l'API de DeepSeek pour l'adaptateur DeepSeek de dsh (/v1/messages) : le modèle
# « deepseek-flash » de dsh est DeepSeek-V4.1 Flash, celui que sert deepseek.sh. Seule cette entrée est
# remplacée ; les autres fournisseurs réglés dans dsh restent tels quels.
DSH_HOME=${DSH_HOME:-${MAISON:-$HOME}/.dsh}
export DSH_HOME
mkdir -p "$DSH_HOME"
SURCOUCHE=$DSH_HOME/ik-llama.patch.yml
cat > "$SURCOUCHE" <<EOF
# Écrit par harness.sh (ik_llama.cpp) à chaque lancement : les réglages se changent dans ~/.deepseek.conf.
- id: llm-deepseek
  config:
    baseURL: $SERVEUR/v1
    apiKeyEnv: IK_LLAMA_API_KEY
    reasoningEffort: $EFFORT
    maxTokens: $SORTIE
    models:
      - id: deepseek-flash
        name: DeepSeek-V4.1 Flash (local)
        contextWindow: $CTX
        systemPromptUpdate: in-history
EOF

# la clé : dsh en veut une même pour un serveur qui n'en demande pas
export IK_LLAMA_API_KEY=${API_KEY:-aucune}
[ "$TELEMETRIE" = 1 ] || export DSH_TELEMETRY_DISABLED=1

# --- lancement -------------------------------------------------------------
if [ $# -eq 0 ]; then
    set -- web --host "$H_HOST" --port "$H_PORT" --no-open
    echo "== Interface web de dsh sur $H_HOST:$H_PORT, modèle servi par $SERVEUR (Ctrl+C pour l'arrêter)"
    echo "   Ouvrez l'adresse affichée ci-dessous ; depuis un autre poste :  ssh -L $H_PORT:127.0.0.1:$H_PORT $(id -un)@$(uname -n)"
fi
PROFIL=$1
shift
# shellcheck disable=SC2086
exec $DSH "$PROFIL" --patch "$SURCOUCHE" "$@"
