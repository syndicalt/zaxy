#!/bin/sh
# Zaxy installer — governed active memory for agent fleets.
#
#   curl -fsSL https://zaxy.com/install.sh | sh
#
# Installs the `zaxy` CLI and registers its MCP server with every agent harness
# it detects on this machine (Claude Code, Codex, opencode, OpenClaw, Hermes,
# Z.ai ZCode, Pi). Uses the embedded LadybugDB backend — zero database setup;
# each project self-provisions a local store on first use.
#
# Options (env vars, for piped/CI use):
#   ZAXY_VERSION=3.0.2     pin a version (default: latest)
#   ZAXY_CLIENTS=codex,... only configure these harnesses (default: auto-detect)
#   ZAXY_NO_CONFIGURE=1    install the binary only; skip harness registration
#   ZAXY_INSTALLER=uv|pipx|pip   force an install method (default: prefer uv)
set -eu

# ---- output helpers (color only on a TTY) ----------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B="$(printf '\033[1m')"; G="$(printf '\033[32m')"; Y="$(printf '\033[33m')"
  R="$(printf '\033[31m')"; D="$(printf '\033[0m')"
else
  B=''; G=''; Y=''; R=''; D=''
fi
info() { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s\n' "$B" "$D" "$*"; }
warn() { printf '%swarn:%s %s\n' "$Y" "$D" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$R" "$D" "$*" >&2; exit 1; }
has()  { command -v "$1" >/dev/null 2>&1; }

PKG="zaxy-memory"
SPEC="$PKG"
[ -n "${ZAXY_VERSION:-}" ] && SPEC="$PKG==$ZAXY_VERSION"

info "${B}Zaxy${D} — governed active memory for agent fleets"
info ""

# ---- 1. install the zaxy CLI ------------------------------------------------
install_with_uv() {
  if ! has uv; then
    step "Installing uv (Python toolchain)…"
    curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 \
      || die "could not install uv; install it from https://astral.sh/uv or set ZAXY_INSTALLER=pipx"
    # uv installs to ~/.local/bin by default
    export PATH="$HOME/.local/bin:$PATH"
  fi
  step "Installing $SPEC with uv…"
  uv tool install --force "$SPEC" >/dev/null
  export PATH="$HOME/.local/bin:$(uv tool dir 2>/dev/null || echo "$HOME/.local/bin"):$PATH"
}

install_with_pipx() { step "Installing $SPEC with pipx…"; pipx install --force "$SPEC" >/dev/null; }
install_with_pip()  {
  step "Installing $SPEC with pip (--user)…"
  python3 -m pip install --user --upgrade "$SPEC" >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
}

if has zaxy && [ -z "${ZAXY_VERSION:-}" ] && [ -z "${ZAXY_INSTALLER:-}" ]; then
  step "zaxy already installed ($(zaxy --version 2>/dev/null || echo present)); re-registering harnesses."
else
  case "${ZAXY_INSTALLER:-auto}" in
    uv)   install_with_uv ;;
    pipx) install_with_pipx ;;
    pip)  install_with_pip ;;
    auto)
      if has uv || has curl; then install_with_uv
      elif has pipx; then install_with_pipx
      elif has python3; then install_with_pip
      else die "need one of: uv, pipx, or python3+pip. See https://zaxy.com/docs"
      fi ;;
    *) die "unknown ZAXY_INSTALLER='$ZAXY_INSTALLER' (use uv|pipx|pip)" ;;
  esac
fi

has zaxy || die "installed, but 'zaxy' is not on PATH. Add ~/.local/bin to PATH and re-run."
info "${G}✓${D} zaxy installed: $(command -v zaxy)"
info ""

# ---- 2. register with detected agent harnesses ------------------------------
if [ "${ZAXY_NO_CONFIGURE:-0}" = "1" ]; then
  info "Skipping harness registration (ZAXY_NO_CONFIGURE=1)."
  info "Run ${B}zaxy install${D} yourself when ready."
  exit 0
fi

step "Registering the Zaxy MCP server with your agent harnesses…"
if [ -n "${ZAXY_CLIENTS:-}" ]; then
  zaxy install --clients "$ZAXY_CLIENTS"
else
  zaxy install
fi

info ""
info "${G}Done.${D} Restart your agent(s), then in any project run:"
info "  ${B}zaxy checkout${D}    # see the memory your agent will use"
info "Docs: https://zaxy.com/docs"
