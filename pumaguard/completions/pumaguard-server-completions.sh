#!/bin/bash

_pumaguard_server_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="-h --help --debug --notebook --watch-method " \
        "--model-path --completion"

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi

    case "${prev}" in
        --notebook)
            return 0
            ;;
        --model-path)
            COMPREPLY=( $(compgen -A directory \
                -o plusdirs -o nospace -- ${cur}) )
            return 0
            ;;
        --watch-method)
            opts="inotify os"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        --completion)
            opts="bash"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

complete -F _pumaguard_server_completions pumaguard.pumaguard-server
complete -F _pumaguard_server_completions pumaguard-server
