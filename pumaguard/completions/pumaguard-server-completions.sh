#!/bin/bash

_pumaguard_server_completions() {
    local cur prev opts

    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD - 1]}"
    opts=(
        -h
        --help
        --debug
        --notebook
        --watch-method
        --model-path
        --completion
        --settings
    )
    if [[ ${cur} == -* ]]; then
        COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
        return 0
    fi

    case "${prev}" in
    --notebook)
        return 0
        ;;
    --model-path)
        COMPREPLY=($(compgen -d -o dirnames -o nospace -- "${cur}"))
        return 0
        ;;
    --watch-method)
        opts="inotify os"
        COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
        return 0
        ;;
    --completion)
        opts="bash"
        COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
        return 0
        ;;
    *)
        COMPREPLY=($(compgen -d -- "${cur}"))
        return 0
        ;;
    esac

    COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
    return 0
}

complete -F _pumaguard_server_completions pumaguard.pumaguard-server
complete -F _pumaguard_server_completions pumaguard-server
