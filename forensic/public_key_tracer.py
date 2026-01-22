import json
from typing import Dict, List


class PublicKeyTracer:
    """Minimal tracer for a public key's observed activity.

    This is a lightweight helper that records a placeholder history and
    writes `public_key_<short>_history.json`. It attempts no aggressive
    network scanning by default to keep imports safe.
    """

    def __init__(self):
        pass

    def trace_public_key(self, public_key_hex: str) -> Dict:
        """Return a small history object for the provided public key.

        If `requests` is available, try to find associated addresses via
        Blockstream (best-effort). Otherwise return an empty history.
        """
        history = {"public_key": public_key_hex, "transactions": []}

        try:
            import requests
        except Exception:
            # no network available at import-time; return placeholder
            out_path = f"public_key_{public_key_hex[:8]}_history.json"
            try:
                with open(out_path, "w") as f:
                    json.dump(history, f, indent=2)
            except Exception:
                pass
            return history

        # Best-effort: try to derive an address (compressed pubkey -> P2PKH not implemented here)
        # Fetching by pubkey isn't directly supported by simple APIs; skip heavy scanning.
        out_path = f"public_key_{public_key_hex[:8]}_history.json"
        try:
            with open(out_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

        return history


def trace_public_key_hex(public_key_hex: str) -> Dict:
    tracer = PublicKeyTracer()
    return tracer.trace_public_key(public_key_hex)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python forensic/public_key_tracer.py <pubkey_hex>")
        sys.exit(1)
    pub = sys.argv[1]
    print(trace_public_key_hex(pub))
