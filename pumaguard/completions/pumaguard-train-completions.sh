#!/bin/bash

_pumaguard_train_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="-h --help --debug --model-path --model-output \
        --notebook --completion"

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
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
        --model-output)
            COMPREPLY=( $(compgen -d -o dirnames -o nospace -- "${cur}") )
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

complete -F _pumaguard_train_completions pumaguard.pumaguard-train
complete -F _pumaguard_train_completions pumaguard-train
