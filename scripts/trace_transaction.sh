#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <txid>"
  exit 1
fi

TXID=$1

echo "🔍 Tracing transaction: $TXID"

python3 - <<'PY'
import sys
sys.path.append('.')
from forensic.transaction_analyzer import TransactionAnalyzer
from forensic.public_key_tracer import PublicKeyTracer
from forensic.address_clusterer import cluster_addresses

txid = sys.argv[1] if len(sys.argv) > 1 else None
if not txid:
    print('txid missing')
    sys.exit(1)

ta = TransactionAnalyzer(txid)
analysis = ta.analyze_transaction()
print('Inputs:', len(analysis.get('inputs', [])))
print('Outputs:', len(analysis.get('outputs', [])))
print('Public keys:', len(analysis.get('public_keys', [])))

tracer = PublicKeyTracer()
for pk in analysis.get('public_keys', []):
    print('Tracing pubkey', pk[:12] + '...')
    hist = tracer.trace_public_key(pk)
    print('Found transactions:', len(hist.get('transactions', [])))

addresses = (analysis.get('inputs', []) or []) + (analysis.get('outputs', []) or [])
clusters, cluster_analysis = cluster_addresses(addresses)
print('Clusters:', cluster_analysis)

PY "$TXID"
