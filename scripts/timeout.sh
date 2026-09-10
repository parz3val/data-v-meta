#!/bin/sh
# portable timeout (macOS has no coreutils timeout): timeout.sh <secs> <cmd...>
s="$1"; shift
perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$s" "$@"
