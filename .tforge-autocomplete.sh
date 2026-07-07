_tforge_completion() {
    local cur prev words cword
    if [ -n "${ZSH_VERSION:-}" ]; then
        words=(${(f)"$(printf '%s\n' "${words[@]}")"})
        cur="${words[CURRENT]}"
        prev="${words[CURRENT-1]}"
        cword=$((CURRENT-1))
    else
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        cword="$COMP_CWORD"
    fi

    local subcommands="check validate lint status generate autocomplete"
    local curriculums="arraysmith tensorsmith hpcsmith infra all"
    local tiers="basic intermediate advanced applications"

    if [ "$cword" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${subcommands}" -- "${cur}") )
        return 0
    fi

    if [ "${words[1]}" = "check" ] || [ "${words[1]}" = "generate" ]; then
        if [ "$cword" -eq 2 ]; then
            COMPREPLY=( $(compgen -W "${curriculums}" -- "${cur}") )
            return 0
        elif [ "$cword" -eq 3 ]; then
            COMPREPLY=( $(compgen -W "${tiers}" -- "${cur}") )
            return 0
        fi
    fi

    COMPREPLY=()
    return 0
}

if [ -n "${ZSH_VERSION:-}" ]; then
    autoload -Uz compinit && compinit
    autoload -Uz bashcompinit && bashcompinit
fi
complete -F _tforge_completion tforge
