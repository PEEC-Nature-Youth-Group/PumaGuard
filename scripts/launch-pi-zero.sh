#!/bin/bash

set -e -u

declare force=0
declare vm_name=""

while (( $# > 0 )); do
    case $1 in
        -h|--help)
            cat <<EOF
Script to launch a Raspberry Pi Zero W virtual machine.

Options:

--name VM_NAME      The name of the VM (required)
--force             Delete existing VM
EOF
            exit
            ;;
        --name)
            shift
            vm_name="$1"
            ;;
        --force)
            force=$(( (force + 1)%2 ))
            ;;
    esac
    shift
done

if [[ -z ${vm_name} ]]; then
    echo "missing VM name"
    exit 1
fi

if (( force == 1 )); then
    multipass delete --purge "${vm_name}"
fi
multipass launch core24 --memory 512M --disk 64G --name "${vm_name}"
