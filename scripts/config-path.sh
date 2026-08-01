#!/bin/bash

default_voice_config_path() {
  local config_home
  config_home="${XDG_CONFIG_HOME:-$HOME/.config}"
  printf '%s/speech-agent-workbench/config.json\n' "$config_home"
}
