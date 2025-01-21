#!/bin/bash

_pumaguard_classify_completions() {
    local cur prev opts

    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD - 1]}"
    opts=(
        -h
        --help
        --completion
        --debug
        --model-path
        --notebook
        --settings
    )

    if [[ ${cur} == -* ]]; then
        COMPREPLY=($(compgen -W "${opts[*]}" -- "${cur}"))
        return 0
    fi

    case "${prev}" in
    --completion)
        opts=(bash)
        COMPREPLY=($(compgen -W "${opts[*]}" -- "${cur}"))
        return 0
        ;;
    --notebook)
        return 0
        ;;
    --model-path)
        COMPREPLY=($(compgen -d -o dirnames -o nospace -- "${cur}"))
        return 0
        ;;
    *)
        # Suggest files, but avoid the trailing space for directories
        if [[ "${cur}" == */ ]]; then
            # Only suggest directories if the input ends with a slash
            COMPREPLY=($(compgen -d -- "${cur}"))
        else
            # Suggest files and directories without a trailing space
            COMPREPLY=($(compgen -f -- "${cur}"))
        fi
        return 0
        ;;
    esac

    COMPREPLY=($(compgen -W "${opts[*]}" -- "${cur}"))
    return 0
}

complete -F _pumaguard_classify_completions pumaguard.pumaguard-classify
complete -F _pumaguard_classify_completions pumaguard-classify
