#!/bin/bash

_pumaguard_completions() {
    local cur prev opts

    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts=(
        -h
        --help
        --completion
        --data-directory
        --debug
        --epochs
        --model-output
        --model-path
        --no-load-previous-session
        --notebook
        --settings
    )

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts[*]}" -- "${cur}") )
        return 0
    fi

    case "${prev}" in
        --completion)
            opts=(bash)
            COMPREPLY=( $(compgen -W "${opts[*]}" -- "${cur}") )
            return 0
            ;;
        --epochs|--notebook)
            return 0
            ;;
        --model-path|--model-output|--data-directory)
            COMPREPLY=( $(compgen -d -o dirnames -o nospace -- "${cur}") )
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts[*]}" -- "${cur}") )
    return 0
}

complete -F _pumaguard_completions pumaguard.pumaguard
complete -F _pumaguard_completions pumaguard
