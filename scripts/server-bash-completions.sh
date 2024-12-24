#!/bin/bash

_pumaguard_server_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="start stop restart status"

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "--help --version" -- ${cur}) )
        return 0
    fi

    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

complete -F _pumaguard_server_completions pumaguard-server
