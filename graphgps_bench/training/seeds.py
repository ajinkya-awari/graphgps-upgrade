from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedManifest:
    seed: int
    python_random_probe: float
    manifest_hash: str


def seed_manifest(seed: int) -> SeedManifest:
    rng = random.Random(seed)
    probe = rng.random()
    payload = {"seed": seed, "python_random_probe": probe}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return SeedManifest(seed=seed, python_random_probe=probe, manifest_hash=digest)
