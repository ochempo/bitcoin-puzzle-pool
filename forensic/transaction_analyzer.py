import json
from typing import List, Dict, Optional

def _hex_like(s: str) -> bool:
    return isinstance(s, str) and all(c in "0123456789abcdefABCDEF" for c in s)


class TransactionAnalyzer:
    """Lightweight transaction fetcher and analyzer.

    Note: network calls are only made from `fetch_transaction()`; importing
    this module does not require network libraries to be present.
    """

    def __init__(self, txid: str):
        self.txid = txid
        self.tx_json: Dict = {}
        self.input_addresses: List[str] = []
        self.output_addresses: List[str] = []
        self.public_keys: List[str] = []

    def fetch_transaction(self) -> Dict:
        """Fetch transaction JSON from Blockstream public API.

        Raises RuntimeError if `requests` is not available or fetch fails.
        """
        try:
            import requests
        except Exception:
            raise RuntimeError("requests library required to fetch transaction data")

        url = f"https://blockstream.info/api/tx/{self.txid}/raw"
        # Try JSON-friendly endpoint first
        try:
            r = requests.get(f"https://blockstream.info/api/tx/{self.txid}", timeout=15)
            if r.status_code == 200:
                try:
                    self.tx_json = r.json()
                    return self.tx_json
                except Exception:
                    # fallback to raw
                    pass

            r2 = requests.get(url, timeout=15)
            if r2.status_code == 200:
                # Raw hex
                self.tx_json = {"raw_hex": r2.text}
                return self.tx_json

            raise RuntimeError(f"Failed to fetch tx {self.txid}: {r.status_code if 'r' in locals() else 'unknown'}")
        except Exception as e:
            raise RuntimeError(str(e))

    def get_public_keys(self) -> List[str]:
        """Heuristically extract public keys from fetched transaction JSON.

        Returns list of hex strings (may be empty).
        """
        if not self.tx_json:
            return []

        # Try to find common fields
        for vin in self.tx_json.get("vin", []) or self.tx_json.get("inputs", []) or []:
            script = None
            if isinstance(vin, dict):
                script = vin.get("scriptSig") or vin.get("witness") or vin.get("txinwitness")

            if isinstance(script, dict):
                asm = script.get("asm")
                if isinstance(asm, str):
                    for part in asm.split():
                        if _hex_like(part) and len(part) in (66, 130):
                            self.public_keys.append(part)

            if isinstance(script, list):
                for part in script:
                    if _hex_like(part) and len(part) in (66, 130):
                        self.public_keys.append(part)

        # dedupe
        self.public_keys = list(dict.fromkeys(self.public_keys))
        return self.public_keys

    def analyze_transaction(self) -> Dict:
        """Pull addresses and public keys into a summary and write to JSON."""
        if not self.tx_json:
            try:
                self.fetch_transaction()
            except Exception as e:
                return {"error": str(e)}

        # outputs
        for vout in (self.tx_json.get("vout") or self.tx_json.get("outputs") or []):
            if isinstance(vout, dict):
                addr = vout.get("scriptpubkey_address") or vout.get("address")
                if addr:
                    self.output_addresses.append(addr)

        # inputs
        for vin in (self.tx_json.get("vin") or self.tx_json.get("inputs") or []):
            if isinstance(vin, dict):
                prev = vin.get("prevout") or vin.get("prev_out") or {}
                if isinstance(prev, dict):
                    addr = prev.get("scriptpubkey_address") or prev.get("address")
                    if addr:
                        self.input_addresses.append(addr)

        self.get_public_keys()

        analysis = {
            "txid": self.txid,
            "inputs": self.input_addresses,
            "outputs": self.output_addresses,
            "public_keys": self.public_keys,
        }

        out_path = f"transaction_{self.txid}_analysis.json"
        try:
            with open(out_path, "w") as f:
                json.dump(analysis, f, indent=2)
        except Exception:
            pass

        return analysis

    def follow_funding(self, depth: int = 1) -> Dict:
        """Follow funding chain by fetching previous transactions up to `depth` levels.

        Returns a dict {'txid': <txid>, 'chain': {prev_txid: tx_json, ...}}
        and writes `transaction_<txid>_chain.json`.
        """
        try:
            import requests
        except Exception:
            return {"error": "requests required for follow_funding"}

        chain = {}

        def fetch_tx(txid: str) -> Optional[Dict]:
            try:
                r = requests.get(f"https://blockstream.info/api/tx/{txid}", timeout=15)
                if r.status_code == 200:
                    try:
                        return r.json()
                    except Exception:
                        return {"raw": r.text}
            except Exception:
                return None

        # ensure base tx is fetched
        if not self.tx_json:
            try:
                self.fetch_transaction()
            except Exception as e:
                return {"error": str(e)}

        to_visit = [self.txid]
        visited = set()

        for level in range(depth):
            next_level = []
            for tx in to_visit:
                if tx in visited:
                    continue
                visited.add(tx)
                tx_json = fetch_tx(tx)
                if tx_json is None:
                    continue
                chain[tx] = tx_json

                # find previous txids in vin
                vins = tx_json.get("vin") or tx_json.get("inputs") or []
                for vin in vins:
                    if isinstance(vin, dict):
                        prev_txid = vin.get("txid") or vin.get("txid_hex")
                        if prev_txid and prev_txid not in visited:
                            next_level.append(prev_txid)
            to_visit = next_level

        out = {"txid": self.txid, "chain": chain}
        out_path = f"transaction_{self.txid}_chain.json"
        try:
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
        except Exception:
            pass

        return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python forensic/transaction_analyzer.py <txid>")
        sys.exit(1)

    txid = sys.argv[1]
    ta = TransactionAnalyzer(txid)
    try:
        print(ta.analyze_transaction())
    except Exception as e:
        print("Error:", e)
