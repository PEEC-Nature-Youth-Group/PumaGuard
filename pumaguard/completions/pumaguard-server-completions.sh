#!/bin/bash

if [[ -z "$( declare -f _comp_compgen 2> /dev/null )" ]]; then
  echo "sourcing"
  source ${SNAP}/usr/share/bash-completion/bash_completion
fi

_pumaguard_server_completions() {
    local cur prev opts

    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts=(
        -h
        --help
        --completion
        --debug
        --model-path
        --notebook
        --watch-method
    )

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
        return 0
    fi

    case "${prev}" in
        --notebook)
            return 0
            ;;
        --model-path)
            COMPREPLY=( $(compgen -d -o dirnames -o nospace -- "${cur}") )
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
            _comp_compgen -a filedir -d
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    return 0
}

complete -F _pumaguard_server_completions pumaguard.pumaguard-server
complete -F _pumaguard_server_completions pumaguard-server
