#!/bin/bash

if [[ -z "$( declare -f _filedir 2> /dev/null )" ]]; then
    echo "sourcing (SNAP = ${SNAP})"
    source ${SNAP}/usr/share/bash-completion/bash_completion
fi

_pumaguard_classify_completions() {
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
        --notebook)
            return 0
            ;;
        --model-path)
            COMPREPLY=( $(compgen -d -o dirnames -o nospace -- "${cur}") )
            return 0
            ;;
        *)
            _filedir
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts[*]}" -- "${cur}") )
    return 0
}

complete -F _pumaguard_classify_completions pumaguard.pumaguard-classify
complete -F _pumaguard_classify_completions pumaguard-classify
