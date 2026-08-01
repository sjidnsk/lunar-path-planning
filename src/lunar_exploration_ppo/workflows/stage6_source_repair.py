"""One-time, fail-closed Stage 6A Update 9 -> 10 source-repair bridge."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    lexical_absolute,
    require_plain_path,
    secure_read_bytes,
)


SOURCE_REPAIR_AMENDMENT_NAME: Final = "source-repair-amendment.json"
SOURCE_REPAIR_SCHEMA: Final = "stage6_source_repair_amendment/v1"
SOURCE_REPAIR_ID: Final = "stage6a_frontier_empty_candidate_update10/v1"
SOURCE_REPAIR_CONTINUATION_NAME: Final = "source-repair-continuation.json"
SOURCE_REPAIR_CONTINUATION_SCHEMA: Final = "stage6_source_repair_continuation/v1"
SOURCE_REPAIR_CONTINUATION_ID: Final = (
    "stage6a_evaluation_device_equivalence_update10/v1"
)
SOURCE_REPAIR_SUPPLEMENT_NAME: Final = "source-repair-supplement.json"
SOURCE_REPAIR_SUPPLEMENT_SCHEMA: Final = "stage6_source_repair_supplement/v1"
SOURCE_REPAIR_SUPPLEMENT_ID: Final = "stage6a_initial_ratio_gate_update33/v1"
SOURCE_REPAIR_CLOSURE_NAME: Final = "source-repair-closure.json"
SOURCE_REPAIR_CLOSURE_SCHEMA: Final = "stage6_source_repair_closure/v1"
SOURCE_REPAIR_CLOSURE_ID: Final = (
    "stage6a_proxy_safety_acceptance_contract_update33/v1"
)
SOURCE_REPAIR_FRONTIER_RECOVERY_NAME: Final = (
    "source-repair-frontier-recovery.json"
)
SOURCE_REPAIR_FRONTIER_RECOVERY_SCHEMA: Final = (
    "stage6_source_repair_frontier_recovery/v1"
)
SOURCE_REPAIR_FRONTIER_RECOVERY_ID: Final = (
    "stage6a_irregular_frontier_empty_update47/v1"
)
SOURCE_REPAIR_SENSOR_ACCELERATION_NAME: Final = (
    "source-repair-sensor-acceleration.json"
)
SOURCE_REPAIR_SENSOR_ACCELERATION_SCHEMA: Final = (
    "stage6_source_repair_sensor_acceleration/v1"
)
SOURCE_REPAIR_SENSOR_ACCELERATION_SUMMARY_SCHEMA: Final = (
    "stage6_source_repair_summary/v6"
)
SOURCE_REPAIR_SENSOR_ACCELERATION_ID: Final = (
    "stage6a_sensor_hotpath_exact_coverable_cache_update50/v1"
)
FIXED_FORMAL_RUN_ID: Final = "s6-standard-single-r1-20260718T220434Z"
FIXED_SEED: Final = 20260716
LAST_ORIGIN_UPDATE: Final = 9
FIRST_REPAIRED_UPDATE: Final = 10
NEXT_TRANSACTION_KEY: Final = "0009:20260716:update:010"
LAST_CONTINUATION_UPDATE: Final = 32
FIRST_SUPPLEMENT_UPDATE: Final = 33
SUPPLEMENT_NEXT_TRANSACTION_KEY: Final = "0032:20260716:update:033"
FIRST_CLOSURE_UPDATE: Final = 33
CLOSURE_NEXT_TRANSACTION_KEY: Final = SUPPLEMENT_NEXT_TRANSACTION_KEY
LAST_FRONTIER_RECOVERY_PARENT_UPDATE: Final = 46
FIRST_FRONTIER_RECOVERY_UPDATE: Final = 47
FRONTIER_RECOVERY_NEXT_TRANSACTION_KEY: Final = "0046:20260716:update:047"
LAST_SENSOR_ACCELERATION_PARENT_UPDATE: Final = 49
FIRST_SENSOR_ACCELERATION_UPDATE: Final = 50
SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY: Final = "0049:20260716:update:050"
SENSOR_ACCELERATION_FAILED_RESOURCE_ATTEMPT: Final = 1
SENSOR_ACCELERATION_NEXT_RESOURCE_ATTEMPT: Final = 2
SENSOR_ACCELERATION_NEXT_SEGMENT_INDEX: Final = 7

_FIXED_LINEAGE_AUDIT_SHA256 = (
    "95573b284888e363a3dfe87db012f0de47fe890a1557cd3736989b5b689e66b2"
)
_FIXED_ORIGIN_IMMUTABLE_SHA256 = (
    "29c47d78999f6e9e18f41cd3a2c12e29065d403a380a39b60b79503497bc9503"
)
_FIXED_ORIGIN_CHECKPOINT_LINEAGE_SHA256 = (
    "9d7362c0c681acd5f36d9819aa507e1de26d8c507959fce6c823d60b781cfce2"
)

_FIXED_PREFIX_BINDINGS: dict[str, dict[str, object]] = {
    "config.json": {
        "path": "config.json",
        "sha256": "7b1c37673c105c34e7bd71d04d4f508d3a167cdf77426b0b35c7aff86ba8adae",
        "size_bytes": 5298,
    },
    "lineage_audit.json": {
        "path": "lineage_audit.json",
        "sha256": _FIXED_LINEAGE_AUDIT_SHA256,
        "size_bytes": 19634,
    },
    "checkpoints/index.jsonl": {
        "path": "checkpoints/index.jsonl",
        "sha256": "886c8c5561d8f1b2a16f44e119056d1d55980493d61f12e6152f3c1466b89960",
        "size_bytes": 4617,
    },
    "job-state.jsonl": {
        "path": "job-state.jsonl",
        "sha256": "a0f143fa49f5b04458d122a68e33d5d449318cd6bddeb91a716ab24f63d714b8",
        "size_bytes": 21705,
    },
    "phase-state.jsonl": {
        "path": "phase-state.jsonl",
        "sha256": "b3f54b83468c9a5aa88ceb37b5bf67bea4c47528a29160bd3708a4ab8012ad5b",
        "size_bytes": 2149,
    },
    "resource_audit.jsonl": {
        "path": "resource_audit.jsonl",
        "sha256": "de03f80e349e03e626a2c69712cfcc28fdb572ebd5bf9ba047f7638018dab9a1",
        "size_bytes": 21738,
    },
    "training_metrics.jsonl": {
        "path": "training_metrics.jsonl",
        "sha256": "b2bb1f422ba2d27365e00cad8b22578ea23ebf9037368baa1d92ba9237a91f3b",
        "size_bytes": 678716,
    },
    "math_audit.jsonl": {
        "path": "math_audit.jsonl",
        "sha256": "ae477d88523b7f0095204a3fcfc73b763845f7933c768822e4c38442eb48a711",
        "size_bytes": 71384,
    },
    "checkpoint_audit.jsonl": {
        "path": "checkpoint_audit.jsonl",
        "sha256": "b54db9e9286629e385893a8b51c75c5a642cefd44ed1314234b8af888d9139d4",
        "size_bytes": 852,
    },
    "checkpoints/seed-20260716/update-00000009/checkpoint.pt": {
        "path": "checkpoints/seed-20260716/update-00000009/checkpoint.pt",
        "sha256": "9893e11ba4731707d629ef64bf9fe64a02b587ac0732545cc7805551aa88e4fd",
        "size_bytes": 30806315,
    },
    "checkpoints/seed-20260716/update-00000009/manifest.json": {
        "path": "checkpoints/seed-20260716/update-00000009/manifest.json",
        "sha256": "4d8994b549224e0a5a158fc119a479955089293630744001d178923892903c9b",
        "size_bytes": 964,
    },
    "checkpoints/seed-20260716/update-00000009/complete.json": {
        "path": "checkpoints/seed-20260716/update-00000009/complete.json",
        "sha256": "374d44204fe20a57007885823edb80bd4aa2472919ba8fcc2f962b4c3b79d528",
        "size_bytes": 256,
    },
}

_FIXED_EVIDENCE_BINDINGS: dict[str, dict[str, object]] = {
    "repair_diff": {
        "kind": "repair_diff",
        "path": "D:/xunce/review/s6-update10-fix-r1/review-package.diff",
        "sha256": "022b9eef839a5ebfd24bde25fa83d98d7c0450f897c270bb3ca5142f2a488919",
        "size_bytes": 15405,
    },
    "review_report": {
        "kind": "review_report",
        "path": "D:/xunce/review/s6-update10-fix-r1/review-report.md",
        "sha256": "cdb263d5903d0d532e5d7a3197b4d119d19896b357e216f70889f651a0853e71",
        "size_bytes": 3019,
    },
    "update10_replay": {
        "kind": "update10_replay",
        "path": "D:/xunce/review/s6-update10-debug-r1/unexpected-success.json",
        "sha256": "8c93cc6de0de396189c6336d4c0ad44b0336c127c7616409bb95db9f0129d098",
        "size_bytes": 189,
    },
}

_FIXED_PRIMARY_AMENDMENT_BINDING: dict[str, object] = {
    "path": SOURCE_REPAIR_AMENDMENT_NAME,
    "sha256": "803fd655881070b7ff424cebf17a6346300f510f7c5e86082e9fda97a194a54f",
    "size_bytes": 26058,
}

_FIXED_CONTINUATION_BINDING: dict[str, object] = {
    "path": SOURCE_REPAIR_CONTINUATION_NAME,
    "sha256": "53f9a81690050c68abe16d8b4583ae0587da313440fefe48f73151fe57de9fec",
    "size_bytes": 26190,
}
_FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256: Final = (
    "9af554d978c339a3d0ad176646dc74ca707748adffef94bc68645beddf6c17b0"
)

_CONTINUATION_PREFIX_BINDINGS: dict[str, dict[str, object]] = {
    **_FIXED_PREFIX_BINDINGS,
    "resource_audit.jsonl": {
        "path": "resource_audit.jsonl",
        "sha256": "698d6e7ee5f834b41e968f66d174cbfc3c6ddda04ccfb3d84bf210f85e64cfbd",
        "size_bytes": 22886,
    },
}

_CONTINUATION_EVIDENCE_BINDINGS: dict[str, dict[str, object]] = {
    "device_failure": {
        "kind": "device_failure",
        "path": "D:/xunce/review/s6-update10-device-debug-r1/evidence.json",
        "sha256": "c354a0afeec129825a8c2f00f1003395b63cf35492ad014e1eabd78c4a14ec3d",
        "size_bytes": 1436,
    },
    "stderr": {
        "kind": "stderr",
        "path": (
            "D:/xunce/out/ppo_frontier-logs/"
            "s6-standard-single-r1-20260718T220434Z.recovery-r13.stderr.log"
        ),
        "sha256": "ee8024c0d0b953cb558ab62bcd3c1eb949f6d54bab5d4e9e1b448ce999d9846f",
        "size_bytes": 4345,
    },
}

_SUPPLEMENT_PREFIX_BINDINGS: dict[str, dict[str, object]] = {
    **_CONTINUATION_PREFIX_BINDINGS,
    "checkpoints/index.jsonl": {
        "path": "checkpoints/index.jsonl",
        "sha256": "b9f9010cfc8474f072dbce1b8ed58651548d536caeb31057290660874d7a9387",
        "size_bytes": 16439,
    },
    "job-state.jsonl": {
        "path": "job-state.jsonl",
        "sha256": "e94e6f5c69417a2a9610eca926a35498f696665106635365cac7b10aa4582494",
        "size_bytes": 78183,
    },
    "phase-state.jsonl": {
        "path": "phase-state.jsonl",
        "sha256": "b3f54b83468c9a5aa88ceb37b5bf67bea4c47528a29160bd3708a4ab8012ad5b",
        "size_bytes": 2149,
    },
    "resource_audit.jsonl": {
        "path": "resource_audit.jsonl",
        "sha256": "355fc194cccd8dadce5ac7bbe655eb42ce51da6758f1d32c68f8d876df0acf2f",
        "size_bytes": 76753,
    },
    "training_metrics.jsonl": {
        "path": "training_metrics.jsonl",
        "sha256": "fcdd582da9934305894879619730c2847eec7ed9fc2d1a60ce85f21c0e82c47b",
        "size_bytes": 2422515,
    },
    "validation_metrics.jsonl": {
        "path": "validation_metrics.jsonl",
        "sha256": "b4fa8c572b5f8db829ee5ae1c73b3f3846fcdd85d4aab7ef364843044fdad7ea",
        "size_bytes": 4763,
    },
    "math_audit.jsonl": {
        "path": "math_audit.jsonl",
        "sha256": "aa8267d7039245fe4b7eff7d48235b66bc263a1735836e1fb345a7bf49d7b4f6",
        "size_bytes": 142772,
    },
    "checkpoint_audit.jsonl": {
        "path": "checkpoint_audit.jsonl",
        "sha256": "6c0fb6a132d86cb146bcf47c8fdcfbbd58bad6aea4a27e2b5904e447fe38da56",
        "size_bytes": 1705,
    },
    "checkpoints/seed-20260716/update-00000032/checkpoint.pt": {
        "path": "checkpoints/seed-20260716/update-00000032/checkpoint.pt",
        "sha256": "c79f1bf9af636c9a14174a367c7ec5fe22996093bb5ee67d7d82fc5f4a4da4d1",
        "size_bytes": 30808683,
    },
    "checkpoints/seed-20260716/update-00000032/manifest.json": {
        "path": "checkpoints/seed-20260716/update-00000032/manifest.json",
        "sha256": "62f4ecb7e198ee07196a5713939a1d6eb49c4363921636995756624e63c36f86",
        "size_bytes": 965,
    },
    "checkpoints/seed-20260716/update-00000032/complete.json": {
        "path": "checkpoints/seed-20260716/update-00000032/complete.json",
        "sha256": "56ceeacb31f846b32165da36ec8497b662002c2064498fbff6b96ef3be9dfa0d",
        "size_bytes": 257,
    },
}

_SUPPLEMENT_EVIDENCE_BINDINGS: dict[str, dict[str, object]] = {
    "stderr": {
        "kind": "stderr",
        "path": (
            "D:/xunce/out/ppo_frontier-logs/"
            "s6-standard-single-r1-20260718T220434Z.recovery-r14.stderr.log"
        ),
        "sha256": "cde50309f20e85be3b7154845d2a7e41189e3f1c78b8c647b9bf61c58d7c62e0",
        "size_bytes": 3266,
    }
}

_FIXED_SUPPLEMENT_BINDING: dict[str, object] = {
    "path": SOURCE_REPAIR_SUPPLEMENT_NAME,
    "sha256": "b8cd7c0e91e8d0f595d5b54989ef050ed263748716d28182cf3fa9fc3dc62cc4",
    "size_bytes": 27321,
}
_FIXED_SUPPLEMENT_EXECUTION_IDENTITY_SHA256: Final = (
    "ba5611952e9d6f7c45b3ba7fba8d8eb3802476157398af4771a90a1e6c47b65a"
)

_CLOSURE_PREFIX_BINDINGS: dict[str, dict[str, object]] = {
    **_SUPPLEMENT_PREFIX_BINDINGS,
    "resource_audit.jsonl": {
        "path": "resource_audit.jsonl",
        "sha256": "26034d2d2d14e930d1e73109a3206b1d8d4113ac2fe622d1b9fa187f13bcfa7f",
        "size_bytes": 77901,
    },
}

_CLOSURE_EVIDENCE_BINDINGS: dict[str, dict[str, object]] = {
    "finding_review": {
        "kind": "finding_review",
        "path": (
            "D:/xunce/review/s6-proxy-safety-r1-20260720T151730Z/"
            "finding-review.json"
        ),
        "sha256": "3ec619c06b01012f541b7facb90af79dcacc4c0c3758c77e390a4e94d6da943a",
        "size_bytes": 2519,
    },
    "safe_stop": {
        "kind": "safe_stop",
        "path": (
            "D:/xunce/review/s6-proxy-safety-r1-20260720T151730Z/"
            "safe-stop-audit.json"
        ),
        "sha256": "46d78e610d61411da3c703321aa612e9ed68c3700f6c1f7254d53c4bf8fb5032",
        "size_bytes": 1613,
    },
}

_FIXED_CLOSURE_BINDING: dict[str, object] = {
    "path": SOURCE_REPAIR_CLOSURE_NAME,
    "sha256": "03461ac4e882979c7cc0a6a7f442cfbc4f93d97f0e0cb3891ce59050da47abb7",
    "size_bytes": 28011,
}
_FIXED_CLOSURE_EXECUTION_IDENTITY_SHA256: Final = (
    "7d69ecefde4ab060d616ff54abf38b5a64bcec74168990b6f44c163f4347755f"
)

_FIXED_FRONTIER_RECOVERY_BINDING: dict[str, object] = {
    "path": SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
    "sha256": "705367a220455b3577b103e3277e0692aa99eaa47ada24e5fcd1e0dc155d732f",
    "size_bytes": 30690,
}
_FIXED_COVERAGE_CACHE_MANIFEST_BINDING: dict[str, object] = {
    "path": (
        "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
        "coverage-cache-manifest.json"
    ),
    "sha256": "675b923b64cee4cde90d5728550a7f939c41cfbc56e9c237268078b59641081c",
    "size_bytes": 519284,
    "cache_root": (
        "D:/xunce/cache/ppo_frontier/"
        "s6-standard-single-r1-20260718T220434Z/coverage-v1"
    ),
    "entry_set_sha256": (
        "c543a277154b41d14bd1a194d6c6e4f2b43075d5b93096c50d1952ee44c43558"
    ),
    "entry_count": 1064,
    "split_counts": {
        "train": 700,
        "validation": 150,
        "test": 150,
        "unseen": 64,
    },
}

_FRONTIER_RECOVERY_PREFIX_BINDINGS: dict[str, dict[str, object]] = {
    **_CLOSURE_PREFIX_BINDINGS,
    "checkpoints/index.jsonl": {
        "path": "checkpoints/index.jsonl",
        "sha256": "1e9facc5c364de8d8857116cdeaa35d95e9a0b16c3ec966a1b845f7e566ccdb8",
        "size_bytes": 23635,
    },
    "job-state.jsonl": {
        "path": "job-state.jsonl",
        "sha256": "347456c9b6826c071fb8779d364f97d6015f2867615890c856e9a53a0a813820",
        "size_bytes": 110765,
    },
    "phase-state.jsonl": {
        "path": "phase-state.jsonl",
        "sha256": "b3f54b83468c9a5aa88ceb37b5bf67bea4c47528a29160bd3708a4ab8012ad5b",
        "size_bytes": 2149,
    },
    "resource_audit.jsonl": {
        "path": "resource_audit.jsonl",
        "sha256": "e8520c6bb6d7dbf20dfeebbd0fcbbcd14e45241656dc02d55cb939e45af6ae1c",
        "size_bytes": 111162,
    },
    "training_metrics.jsonl": {
        "path": "training_metrics.jsonl",
        "sha256": "3f42d64567ff0aa86c69f75466e9c673f25feafc5bb7a59114366cabd76bd1c5",
        "size_bytes": 3489401,
    },
    "validation_metrics.jsonl": {
        "path": "validation_metrics.jsonl",
        "sha256": "e79bc23e771aea6e0f488bf129f37817a63b2f1fd851a33ee13b194bb813063b",
        "size_bytes": 6350,
    },
    "math_audit.jsonl": {
        "path": "math_audit.jsonl",
        "sha256": "aa8267d7039245fe4b7eff7d48235b66bc263a1735836e1fb345a7bf49d7b4f6",
        "size_bytes": 142772,
    },
    "checkpoint_audit.jsonl": {
        "path": "checkpoint_audit.jsonl",
        "sha256": "6c0fb6a132d86cb146bcf47c8fdcfbbd58bad6aea4a27e2b5904e447fe38da56",
        "size_bytes": 1705,
    },
    "checkpoints/seed-20260716/update-00000046/checkpoint.pt": {
        "path": "checkpoints/seed-20260716/update-00000046/checkpoint.pt",
        "sha256": "4c55325c3342d477354bd7e5a909bfd157a9fa6daee39b4f96fe926a7f647a96",
        "size_bytes": 30808555,
    },
    "checkpoints/seed-20260716/update-00000046/manifest.json": {
        "path": "checkpoints/seed-20260716/update-00000046/manifest.json",
        "sha256": "ed607a1168f5377c87e90a62c6651c09c0e9a580f0e0fbc7e91bb3add98434ff",
        "size_bytes": 965,
    },
    "checkpoints/seed-20260716/update-00000046/complete.json": {
        "path": "checkpoints/seed-20260716/update-00000046/complete.json",
        "sha256": "3fec78dc799b770d57beda63c2b6139c5277ba1bf0426fad60c9a4b4995978b0",
        "size_bytes": 257,
    },
}

_FRONTIER_RECOVERY_EVIDENCE_BINDINGS: dict[str, dict[str, object]] = {
    "stderr": {
        "kind": "stderr",
        "path": (
            "D:/xunce/out/ppo_frontier-logs/"
            "s6-standard-single-r1-20260718T220434Z.recovery-r16.stderr.log"
        ),
        "sha256": "06b78828149ba47cc8f1fd5f8ee5f5e753e3fc7d30e9d78b9c8ed14ba736689c",
        "size_bytes": 5221,
    },
    "replay_exception": {
        "kind": "replay_exception",
        "path": (
            "D:/xunce/review/s6-update47-debug-r1-20260721T1635-CST/"
            "replay-exception.json"
        ),
        "sha256": "aeb302256ef29b7ce01ae66adf4101c1c69c9de63522b01ea0389ee79dca62ba",
        "size_bytes": 1486,
    },
    "root_cause": {
        "kind": "root_cause",
        "path": (
            "D:/xunce/review/s6-update47-debug-r1-20260721T1635-CST/"
            "root-cause-analysis.json"
        ),
        "sha256": "7cfbb1f7c906677a1a0bbff96a36288d9a4e4487a78d5845b1a13863db426f48",
        "size_bytes": 2412,
    },
    "reviewed_patch": {
        "kind": "reviewed_patch",
        "path": "D:/xunce/review/s6-update47-fix-r2/review-package.diff",
        "sha256": "131120db74b0e4a6977dd545bbeb3f717ffa10fba4d565160e5d256e26e429e1",
        "size_bytes": 9994,
    },
    "benchmark": {
        "kind": "benchmark",
        "path": "D:/xunce/review/s6-update47-fix-r2/final-fix-benchmark.json",
        "sha256": "325a3e9e71ca147fa88d4b2ba078899d30106a34782ba1271471b01c018beed6",
        "size_bytes": 799,
    },
    "independent_review": {
        "kind": "independent_review",
        "path": "D:/xunce/review/s6-update47-fix-r2/independent-review.json",
        "sha256": "57cebaf1572d33bab51bc7d06cf74435bfb5f13c5d72fc4526f8e6e1559d04e3",
        "size_bytes": 1139,
    },
    "review_manifest": {
        "kind": "review_manifest",
        "path": "D:/xunce/review/s6-update47-fix-r2/review-manifest.json",
        "sha256": "9515183d748a2a993d679687f3dba9d97e8ebf67c0718730a5034b37f43f5add",
        "size_bytes": 1464,
    },
}

_SOURCE_REPAIR_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_FRONTIER_RECOVERY_SOURCE_BINDINGS: dict[str, dict[str, object]] = {
    "src/lunar_exploration_ppo/env/frontier.py": {
        "path": "src/lunar_exploration_ppo/env/frontier.py",
        "sha256": "7d94f48440d5421002c2d05aebf13a73f4062452911f0f8442315ace4069a62c",
    },
    "tests/ppo_highres_frontier/test_stage2_frontier.py": {
        "path": "tests/ppo_highres_frontier/test_stage2_frontier.py",
        "sha256": "1062d477bd62b5bb68f60f5781eb08f215134901dafa4f66c9ca1cf27ae7caa8",
    },
    "src/lunar_exploration_ppo/env/env.py": {
        "path": "src/lunar_exploration_ppo/env/env.py",
        "sha256": "feee2a000faa1c3beb3f308db8cdf28a77978c556bba46302ed977430f0f8b07",
    },
}

_SENSOR_ACCELERATION_PREFIX_BINDINGS: dict[str, dict[str, object]] = {
    **_FRONTIER_RECOVERY_PREFIX_BINDINGS,
    "checkpoints/index.jsonl": {
        "path": "checkpoints/index.jsonl",
        "sha256": "8809b8ff088de0748905f642ee02416a489af52631af017813b93d47fff67f23",
        "size_bytes": 25177,
    },
    "job-state.jsonl": {
        "path": "job-state.jsonl",
        "sha256": "764e31129766f4199d361852d15d4acb97f2e2029d0c2b0fa4bf4681ad3805b9",
        "size_bytes": 117281,
    },
    "resource_audit.jsonl": {
        "path": "resource_audit.jsonl",
        "sha256": "0980a5200ac8900e4055ca41b75083922df3f8a0f1b820e2a83eaa9e184a8607",
        "size_bytes": 119182,
    },
    "training_metrics.jsonl": {
        "path": "training_metrics.jsonl",
        "sha256": "10567615bebbb6362cd9ec3ed06bc38385095b0824de94437fc65af253e07cf0",
        "size_bytes": 3718010,
    },
    "checkpoints/seed-20260716/update-00000049/checkpoint.pt": {
        "path": "checkpoints/seed-20260716/update-00000049/checkpoint.pt",
        "sha256": "cc23ed757dd48b8ad3c98fe95bfbe044562f6f3ff22048ee29337e62aa773066",
        "size_bytes": 30809131,
    },
    "checkpoints/seed-20260716/update-00000049/manifest.json": {
        "path": "checkpoints/seed-20260716/update-00000049/manifest.json",
        "sha256": "a78c37fe3f96efb011df29c3f100cd9af9737ab1bb01a45c7de80cc97700991b",
        "size_bytes": 965,
    },
    "checkpoints/seed-20260716/update-00000049/complete.json": {
        "path": "checkpoints/seed-20260716/update-00000049/complete.json",
        "sha256": "fca211c5256f980e226f14db2817cefbf05ab4656a0420af9ad88cc44243c9de",
        "size_bytes": 257,
    },
}

_SENSOR_ACCELERATION_SOURCE_BINDINGS: dict[str, dict[str, object]] = {
    path: {"path": path, "sha256": sha256, "size_bytes": size_bytes}
    for path, sha256, size_bytes in (
        (
            "docs/superpowers/plans/2026-07-22-ppo-stage6-sensor-hotpath-acceleration.md",
            "0d3cfbae403bbfa8ea929a50022bcdd6ab304f5a88a5a34a8f6d62f37244f7ac",
            9768,
        ),
        (
            "docs/superpowers/plans/2026-07-22-ppo-stage6-coverable-mask-cache.md",
            "23918b98477ac0d87bd31561a7c4df93f745fb792e3fb441e8aea5113516e08b",
            8479,
        ),
        (
            "docs/superpowers/specs/2026-07-22-ppo-stage6-sensor-hotpath-acceleration-design-addendum.md",
            "d873423f2cd11694719c3e392c81a83c6ae1eebf1a409e753d435bb42b2511b8",
            6044,
        ),
        (
            "docs/superpowers/specs/2026-07-22-ppo-stage6-coverable-mask-cache-design-addendum.md",
            "520a4354a02144e4c7f39b5faaeb967ee3695c9b664ded7040cb53cd87636116",
            5675,
        ),
        (
            "scripts/benchmark_ppo_stage6_sensor_hotpath.py",
            "0f843f65d1bf7b8abf3765f976f3e7cd40d7debbf968aae1460cbdaa37d63578",
            13357,
        ),
        (
            "scripts/prewarm_ppo_stage6_coverage_cache.py",
            "4499994b653683adc4f77385c7cab234ec70dd6cc0f10af766b4d47dd2e25033",
            1109,
        ),
        (
            "src/lunar_exploration_ppo/env/action_execution.py",
            "bc673e22e235c3fb60f6d40797d4c2bd33de92428ea8e4f3b4ed3a6a358e3de5",
            2630,
        ),
        (
            "src/lunar_exploration_ppo/env/coverage_cache.py",
            "f8f203381d8ce3bf1797e88bacfbc382572935ebaa48dbd28062e5e14e9a581b",
            22467,
        ),
        (
            "src/lunar_exploration_ppo/env/env.py",
            "7bf8935aa1e305f31619432ee58d94e10a7ff2a46d1742c2aae134d5a9224689",
            44022,
        ),
        (
            "src/lunar_exploration_ppo/env/sensor_model.py",
            "75b360a9606c9c09612ca2a71ead78a5c4d3c70f515be14e4841fb22a58faf03",
            10237,
        ),
        (
            "src/lunar_exploration_ppo/env/standard_training.py",
            "005de2d3fd97c76363736835734ef83acddf9d23d22520c9be40536c8aed08c4",
            23241,
        ),
        (
            "src/lunar_exploration_ppo/workflows/stage6_coverage_cache.py",
            "1ac59fc458ec9f8d2f2eb30d33b3d1bb2edb1de729df71b08f438156a5e028d9",
            28086,
        ),
        (
            "tests/ppo_highres_frontier/test_stage6_coverage_cache.py",
            "f3533241ddb08759a25726a8277700628ab09fbac621c3983d9bebf234f753d4",
            41568,
        ),
        (
            "tests/ppo_highres_frontier/test_stage6_coverage_cache_workflow.py",
            "d3d904ee2bb5f114f2208a488410e528efc7cf7fdc0f4228409e878c936db130",
            27701,
        ),
        (
            "tests/ppo_highres_frontier/test_stage6_sensor_acceleration.py",
            "e83c2b9700e1cb1c33ae4aa94ba8f6628c1f2e1a0e7f3d783984b8931d8a9fe9",
            20309,
        ),
    )
}

_SENSOR_ACCELERATION_EVIDENCE_BINDINGS: dict[str, dict[str, object]] = {
    kind: {"kind": kind, "path": path, "sha256": sha256, "size_bytes": size_bytes}
    for kind, path, sha256, size_bytes in (
        (
            "sensor_implementation_report",
            ".superpowers/sdd/stage6-sensor-acceleration-task1-report.md",
            "7e2ee9f893c4919bec797e1308a511737aa3eaa67a09e369cbe04b938ac75f2c",
            7594,
        ),
        (
            "sensor_fresh_review",
            ".superpowers/sdd/stage6-sensor-acceleration-task1-review.md",
            "af4bb31b6e6924b569ee30b73b0262b0e6f9a8c58857dee27c5659745c3bc49d",
            1845,
        ),
        (
            "cache_b1_fresh_review",
            ".superpowers/sdd/stage6-coverable-cache-task1-final-rereview.md",
            "6665cad5b1e873dc4406186363c70c9581b7b578f2ffe0315dcd1c2921ebbe9d",
            2397,
        ),
        (
            "cache_b2_fresh_review",
            ".superpowers/sdd/stage6-coverable-cache-task2-r2-rereview.md",
            "5e5411ea673178f5c69d7a126909d83e371552db505cabc6f4a926043e38ed9c",
            1760,
        ),
        (
            "cache_formal_spec_review",
            ".superpowers/sdd/stage6-coverable-cache-formal-spec-review-r2.md",
            "719c495d8570f2b6587b0f0d4d8a7688457ebd31e4864d5284be2158c29e4b14",
            6968,
        ),
        (
            "cache_formal_quality_review",
            ".superpowers/sdd/stage6-coverable-cache-formal-quality-review-r2.md",
            "47a0183337622735a198d85e3d497f99c408d4c426b562443b9e386fe0d11855",
            7826,
        ),
        (
            "cache_path_safety_verification",
            ".superpowers/sdd/stage6-coverable-cache-path-safety-main-verification.md",
            "4c6c48612047768fab9e11f0a27f9214dbd4631b15edde8e6136980f7df298c0",
            2343,
        ),
        (
            "cache_path_safety_rereview",
            ".superpowers/sdd/stage6-coverable-cache-path-safety-rereview.md",
            "ca5e8cc2d4ede5789db618de5831a383a3ce9fbbacb4ab2c20c16e4980473cc8",
            2367,
        ),
        (
            "coverage_cache_manifest",
            "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/coverage-cache-manifest.json",
            "675b923b64cee4cde90d5728550a7f939c41cfbc56e9c237268078b59641081c",
            519284,
        ),
        (
            "coverage_cache_formal_audit",
            "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/formal-audit.json",
            "d7c12737da4c9a43353e02347b9c9d6abd527a78f7a3c6675e9164b6ae5b0068",
            4416,
        ),
    )
}
_FIXED_FRONTIER_RECOVERY_EXECUTION_IDENTITY_SHA256: Final = (
    "77cd6e0423b4f3942877fffe20176fbed842e404e0cf5798333b2bfe68e083e3"
)

_APPEND_ONLY_PREFIX_PATHS: Final = frozenset(
    {
        "checkpoints/index.jsonl",
        "job-state.jsonl",
        "phase-state.jsonl",
        "resource_audit.jsonl",
        "training_metrics.jsonl",
        "validation_metrics.jsonl",
        "math_audit.jsonl",
        "checkpoint_audit.jsonl",
    }
)


class Stage6SourceRepairError(RuntimeError):
    """The one authorized source-repair amendment failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(ArtifactStore.canonical_json_bytes(value))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(payload: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Stage6SourceRepairError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or ArtifactStore.canonical_json_bytes(value) != payload:
        raise Stage6SourceRepairError(f"{label} is not canonical JSON")
    return value


def _strict_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    if payload and not payload.endswith(b"\n"):
        raise Stage6SourceRepairError(f"{label} is not newline terminated")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage6SourceRepairError(f"{label} row {index} is invalid") from exc
        canonical = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if not isinstance(value, dict) or line != canonical:
            raise Stage6SourceRepairError(f"{label} row {index} is not canonical")
        rows.append(value)
    return tuple(rows)


def _stage_root(value: str | Path) -> Path:
    stage = lexical_absolute(value)
    try:
        stage = require_plain_path(
            stage,
            leaf_kind="directory",
            label="Stage 6 source-repair root",
        )
    except PathSecurityError as exc:
        raise Stage6SourceRepairError("source-repair root is not plain") from exc
    if stage.name != "s6" or stage.parent.name != FIXED_FORMAL_RUN_ID:
        raise Stage6SourceRepairError("source repair is restricted to the fixed formal run")
    return stage


def _secure_payload(path: Path, *, base: Path, label: str) -> bytes:
    try:
        return secure_read_bytes(path, base=base, label=label).payload
    except (OSError, PathSecurityError) as exc:
        raise Stage6SourceRepairError(f"{label} secure read failed") from exc


def _binding_identity(payload: bytes, path: str) -> dict[str, object]:
    return {"path": path, "sha256": _sha256(payload), "size_bytes": len(payload)}


def _bound_prefix_payloads(
    stage: Path,
    *,
    bindings: Mapping[str, Mapping[str, object]],
    allow_growth: bool,
    label: str,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for relative, binding in sorted(bindings.items()):
        if binding != {
            "path": relative,
            "sha256": binding.get("sha256"),
            "size_bytes": binding.get("size_bytes"),
        } or not _is_sha256(binding.get("sha256")) or type(
            binding.get("size_bytes")
        ) is not int:
            raise Stage6SourceRepairError(f"{label} binding schema drifted")
        payload = _secure_payload(
            stage / relative,
            base=stage,
            label=f"{label} {relative}",
        )
        size = int(binding["size_bytes"])
        expected = {
            "sha256": binding["sha256"],
            "size_bytes": size,
        }
        if allow_growth and relative in _APPEND_ONLY_PREFIX_PATHS:
            if len(payload) < size or {
                "sha256": _sha256(payload[:size]),
                "size_bytes": size,
            } != expected:
                raise Stage6SourceRepairError(
                    f"append-only {label} drifted: {relative}"
                )
            result[relative] = payload[:size]
        else:
            if {"sha256": _sha256(payload), "size_bytes": len(payload)} != expected:
                raise Stage6SourceRepairError(f"{label} drifted: {relative}")
            result[relative] = payload
    return result


def _fixed_prefix_payloads(stage: Path, *, allow_growth: bool) -> dict[str, bytes]:
    return _bound_prefix_payloads(
        stage,
        bindings=_FIXED_PREFIX_BINDINGS,
        allow_growth=allow_growth,
        label="source-repair prefix",
    )


def _continuation_prefix_payloads(
    stage: Path,
    *,
    allow_growth: bool,
) -> dict[str, bytes]:
    return _bound_prefix_payloads(
        stage,
        bindings=_CONTINUATION_PREFIX_BINDINGS,
        allow_growth=allow_growth,
        label="source-repair continuation prefix",
    )


def _supplement_prefix_payloads(
    stage: Path,
    *,
    allow_growth: bool,
) -> dict[str, bytes]:
    return _bound_prefix_payloads(
        stage,
        bindings=_SUPPLEMENT_PREFIX_BINDINGS,
        allow_growth=allow_growth,
        label="source-repair supplement prefix",
    )


def _closure_prefix_payloads(
    stage: Path,
    *,
    allow_growth: bool,
) -> dict[str, bytes]:
    return _bound_prefix_payloads(
        stage,
        bindings=_CLOSURE_PREFIX_BINDINGS,
        allow_growth=allow_growth,
        label="source-repair closure prefix",
    )


def _frontier_recovery_prefix_payloads(
    stage: Path,
    *,
    allow_growth: bool,
) -> dict[str, bytes]:
    return _bound_prefix_payloads(
        stage,
        bindings=_FRONTIER_RECOVERY_PREFIX_BINDINGS,
        allow_growth=allow_growth,
        label="source-repair frontier recovery prefix",
    )


def _sensor_acceleration_prefix_payloads(
    stage: Path,
    *,
    allow_growth: bool,
) -> dict[str, bytes]:
    return _bound_prefix_payloads(
        stage,
        bindings=_SENSOR_ACCELERATION_PREFIX_BINDINGS,
        allow_growth=allow_growth,
        label="source-repair sensor acceleration prefix",
    )


def _bound_evidence_payloads(
    *,
    bindings: Mapping[str, Mapping[str, object]],
    label: str,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for kind, binding in sorted(bindings.items()):
        if (
            binding.get("kind") != kind
            or not isinstance(binding.get("path"), str)
            or not _is_sha256(binding.get("sha256"))
            or type(binding.get("size_bytes")) is not int
        ):
            raise Stage6SourceRepairError(f"{label} binding drifted")
        path = lexical_absolute(str(binding["path"]))
        payload = _secure_payload(path, base=path.parent, label=f"{label} {kind}")
        if {
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        } != {
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        }:
            raise Stage6SourceRepairError(f"{label} drifted: {kind}")
        result[kind] = payload
    return result


def _frontier_recovery_source_payloads() -> dict[str, bytes]:
    root = lexical_absolute(_SOURCE_REPAIR_REPO_ROOT)
    result: dict[str, bytes] = {}
    for relative, binding in sorted(_FRONTIER_RECOVERY_SOURCE_BINDINGS.items()):
        expected_keys = {"path", "sha256"}
        if "size_bytes" in binding:
            expected_keys.add("size_bytes")
        if (
            set(binding) != expected_keys
            or binding.get("path") != relative
            or not _is_sha256(binding.get("sha256"))
            or (
                "size_bytes" in binding
                and type(binding.get("size_bytes")) is not int
            )
        ):
            raise Stage6SourceRepairError(
                "source-repair frontier recovery source binding drifted"
            )
        payload = _secure_payload(
            root / relative,
            base=root,
            label=f"source-repair frontier recovery source {relative}",
        )
        if _sha256(payload) != binding["sha256"] or (
            "size_bytes" in binding and len(payload) != binding["size_bytes"]
        ):
            raise Stage6SourceRepairError(
                f"source-repair frontier recovery source drifted: {relative}"
            )
        result[relative] = payload
    return result


def _sensor_acceleration_source_payloads() -> dict[str, bytes]:
    root = lexical_absolute(_SOURCE_REPAIR_REPO_ROOT)
    result: dict[str, bytes] = {}
    for relative, binding in sorted(_SENSOR_ACCELERATION_SOURCE_BINDINGS.items()):
        if (
            set(binding) != {"path", "sha256", "size_bytes"}
            or binding.get("path") != relative
            or not _is_sha256(binding.get("sha256"))
            or type(binding.get("size_bytes")) is not int
        ):
            raise Stage6SourceRepairError(
                "source-repair sensor acceleration source binding drifted"
            )
        payload = _secure_payload(
            root / relative,
            base=root,
            label=f"source-repair sensor acceleration source {relative}",
        )
        if {
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        } != {
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        }:
            raise Stage6SourceRepairError(
                f"source-repair sensor acceleration source drifted: {relative}"
            )
        result[relative] = payload
    return result


def _validate_coverage_cache_manifest_payload(payload: bytes) -> dict[str, object]:
    value = _strict_json(payload, label="coverage cache manifest")
    binding = _FIXED_COVERAGE_CACHE_MANIFEST_BINDING
    entries = value.get("entries")
    split_counts = value.get("split_counts")
    integrity = value.get("integrity")
    cache_root = value.get("cache_root")
    if (
        set(binding)
        != {
            "path",
            "sha256",
            "size_bytes",
            "cache_root",
            "entry_set_sha256",
            "entry_count",
            "split_counts",
        }
        or not isinstance(entries, list)
        or not isinstance(split_counts, Mapping)
        or not isinstance(integrity, Mapping)
        or not isinstance(cache_root, str)
        or value.get("schema_version")
        != "stage6_exact_coverable_cache_manifest/v1"
        or value.get("entry_set_sha256") != binding.get("entry_set_sha256")
        or dict(split_counts) != binding.get("split_counts")
        or len(entries) != binding.get("entry_count")
        or integrity.get("complete") is not True
        or integrity.get("entry_count") != binding.get("entry_count")
        or integrity.get("valid_entry_count") != binding.get("entry_count")
        or any(
            integrity.get(key) != 0
            for key in (
                "missing_entry_count",
                "duplicate_entry_count",
                "corrupt_entry_count",
            )
        )
        or _canonical_sha256(entries) != binding.get("entry_set_sha256")
    ):
        raise Stage6SourceRepairError("coverage cache manifest contract drifted")
    try:
        actual_cache_root = lexical_absolute(cache_root)
        expected_cache_root = lexical_absolute(str(binding["cache_root"]))
    except (OSError, ValueError) as exc:
        raise Stage6SourceRepairError("coverage cache root drifted") from exc
    if os.path.normcase(str(actual_cache_root)) != os.path.normcase(
        str(expected_cache_root)
    ):
        raise Stage6SourceRepairError("coverage cache root drifted")
    scenario_ids: list[str] = []
    scenario_hashes: list[str] = []
    key_hashes: list[str] = []
    paths: list[str] = []
    derived_split_counts = {key: 0 for key in ("train", "validation", "test", "unseen")}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "scenario_id",
            "scenario_hash",
            "split",
            "key_sha256",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise Stage6SourceRepairError("coverage cache manifest entry drifted")
        scenario_id = entry.get("scenario_id")
        scenario_hash = entry.get("scenario_hash")
        split = entry.get("split")
        key_sha256 = entry.get("key_sha256")
        relative = entry.get("path")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or not _is_sha256(scenario_hash)
            or split not in derived_split_counts
            or not _is_sha256(key_sha256)
            or not isinstance(relative, str)
            or relative != f"entries/{str(key_sha256)[:2]}/{key_sha256}.npz"
            or not _is_sha256(entry.get("sha256"))
            or type(entry.get("size_bytes")) is not int
            or int(entry["size_bytes"]) <= 0
        ):
            raise Stage6SourceRepairError("coverage cache manifest entry drifted")
        scenario_ids.append(scenario_id)
        scenario_hashes.append(str(scenario_hash))
        key_hashes.append(str(key_sha256))
        paths.append(relative)
        derived_split_counts[str(split)] += 1
    if (
        scenario_ids != sorted(scenario_ids)
        or any(
            len(set(values)) != len(entries)
            for values in (scenario_ids, scenario_hashes, key_hashes, paths)
        )
        or derived_split_counts != dict(split_counts)
    ):
        raise Stage6SourceRepairError(
            "coverage cache manifest scenario set drifted"
        )
    return value


def _validate_coverage_cache_formal_audit_payload(
    payload: bytes,
    *,
    manifest_payload: bytes,
    manifest_value: Mapping[str, object],
) -> None:
    audit = _strict_json(payload, label="coverage cache formal audit")
    boundaries = audit.get("boundaries")
    prewarm = audit.get("prewarm")
    strict_load = audit.get("strict_load")
    performance = audit.get("performance")
    equivalence = audit.get("representative_equivalence")
    leakage = audit.get("leakage")
    regressions = audit.get("focused_regressions")
    if (
        audit.get("schema_version") != "stage6_coverable_cache_formal_audit/v1"
        or audit.get("status") != "pass"
        or audit.get("formal_run_id") != FIXED_FORMAL_RUN_ID
        or not isinstance(boundaries, Mapping)
        or boundaries.get("formal_training_checkpoint_update") != 49
        or boundaries.get("formal_training_started") is not False
        or boundaries.get("checkpoint_published") is not False
        or boundaries.get("update50_attempt1_discarded") is not True
        or not isinstance(prewarm, Mapping)
        or prewarm.get("completed_count") != 1064
        or prewarm.get("entry_set_sha256") != manifest_value.get("entry_set_sha256")
        or prewarm.get("manifest_sha256") != _sha256(manifest_payload)
        or prewarm.get("manifest_size_bytes") != len(manifest_payload)
        or prewarm.get("split_counts") != manifest_value.get("split_counts")
        or not isinstance(strict_load, Mapping)
        or strict_load.get("loaded_count") != 1064
        or strict_load.get("canonical_manifest") is not True
        or not isinstance(performance, Mapping)
        or performance.get("passed") is not True
        or not isinstance(performance.get("cached_median_seconds"), (int, float))
        or float(performance["cached_median_seconds"]) > 5.0
        or not isinstance(performance.get("measured_speedup"), (int, float))
        or float(performance["measured_speedup"]) < 20.0
        or not isinstance(equivalence, Mapping)
        or any(
            equivalence.get(key) is not True
            for key in (
                "mask_bitwise_equal",
                "reset_observation_equal",
                "candidate_equal",
                "deterministic_action_equal",
                "first_step_equal",
            )
        )
        or not isinstance(leakage, Mapping)
        or leakage.get("passed") is not True
        or leakage.get("policy_frontier_cache_parameter_absent") is not True
        or not isinstance(regressions, Mapping)
        or regressions.get("failed") != 0
    ):
        raise Stage6SourceRepairError("coverage cache formal audit drifted")


def _sensor_acceleration_evidence_payloads() -> dict[str, bytes]:
    root = lexical_absolute(_SOURCE_REPAIR_REPO_ROOT)
    result: dict[str, bytes] = {}
    for kind, binding in sorted(_SENSOR_ACCELERATION_EVIDENCE_BINDINGS.items()):
        if (
            set(binding) != {"kind", "path", "sha256", "size_bytes"}
            or binding.get("kind") != kind
            or not isinstance(binding.get("path"), str)
            or not _is_sha256(binding.get("sha256"))
            or type(binding.get("size_bytes")) is not int
        ):
            raise Stage6SourceRepairError(
                "source-repair sensor acceleration evidence binding drifted"
            )
        raw_path = Path(str(binding["path"]))
        path = lexical_absolute(raw_path if raw_path.is_absolute() else root / raw_path)
        payload = _secure_payload(
            path,
            base=(path.parent if raw_path.is_absolute() else root),
            label=f"source-repair sensor acceleration evidence {kind}",
        )
        if {
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        } != {
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        }:
            raise Stage6SourceRepairError(
                f"source-repair sensor acceleration evidence drifted: {kind}"
            )
        result[kind] = payload
    manifest_payload = result.get("coverage_cache_manifest")
    audit_payload = result.get("coverage_cache_formal_audit")
    if manifest_payload is None or audit_payload is None:
        raise Stage6SourceRepairError("coverage cache evidence is incomplete")
    manifest_binding = _FIXED_COVERAGE_CACHE_MANIFEST_BINDING
    evidence_binding = _SENSOR_ACCELERATION_EVIDENCE_BINDINGS[
        "coverage_cache_manifest"
    ]
    if (
        evidence_binding.get("path") != manifest_binding.get("path")
        or evidence_binding.get("sha256") != manifest_binding.get("sha256")
        or evidence_binding.get("size_bytes") != manifest_binding.get("size_bytes")
    ):
        raise Stage6SourceRepairError("coverage cache manifest binding drifted")
    manifest_value = _validate_coverage_cache_manifest_payload(manifest_payload)
    _validate_coverage_cache_formal_audit_payload(
        audit_payload,
        manifest_payload=manifest_payload,
        manifest_value=manifest_value,
    )
    return result


def _fixed_evidence_payloads() -> dict[str, bytes]:
    result = _bound_evidence_payloads(
        bindings=_FIXED_EVIDENCE_BINDINGS,
        label="repair evidence",
    )
    replay = _strict_json(result["update10_replay"], label="Update 10 exact replay")
    if replay.get("trainable_transition_count") != 1024:
        raise Stage6SourceRepairError("Update 10 replay did not collect 1024 transitions")
    return result


def _read_cutover_checkpoint_payload(payload: bytes) -> dict[str, object]:
    from lunar_exploration_ppo.ppo.checkpoint import safe_load_checkpoint_payload

    try:
        value = safe_load_checkpoint_payload(payload)
    except Exception as exc:
        raise Stage6SourceRepairError("Update 9 checkpoint payload is invalid") from exc
    if not isinstance(value, dict):
        raise Stage6SourceRepairError("Update 9 checkpoint payload schema drifted")
    return value


def _validate_checkpoint_prefix(
    prefix: Mapping[str, bytes],
    *,
    config_sha256: str,
    receipts: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], dict[str, object]]:
    root = f"checkpoints/seed-{FIXED_SEED}/update-{LAST_ORIGIN_UPDATE:08d}"
    checkpoint = prefix[f"{root}/checkpoint.pt"]
    manifest_payload = prefix[f"{root}/manifest.json"]
    complete_payload = prefix[f"{root}/complete.json"]
    manifest = _strict_json(manifest_payload, label="Update 9 checkpoint manifest")
    complete = _strict_json(complete_payload, label="Update 9 complete marker")
    payload = _read_cutover_checkpoint_payload(checkpoint)
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise Stage6SourceRepairError("Update 9 checkpoint lineage is missing")
    lineage_value = dict(lineage)
    if (
        payload.get("update_step") != LAST_ORIGIN_UPDATE
        or payload.get("config_sha256") != config_sha256
        or not _is_sha256(payload.get("policy_state_sha256"))
        or _canonical_sha256(lineage_value)
        != _FIXED_ORIGIN_CHECKPOINT_LINEAGE_SHA256
        or manifest.get("update_step") != LAST_ORIGIN_UPDATE
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("policy_state_sha256") != payload.get("policy_state_sha256")
        or manifest.get("lineage_sha256") != _canonical_sha256(lineage_value)
        or manifest.get("checkpoint")
        != {
            "path": "checkpoint.pt",
            "sha256": _sha256(checkpoint),
            "size_bytes": len(checkpoint),
        }
        or complete.get("update_step") != LAST_ORIGIN_UPDATE
        or complete.get("checkpoint_sha256") != _sha256(checkpoint)
        or complete.get("manifest_sha256") != _sha256(manifest_payload)
        or not receipts
        or receipts[-1].get("update") != LAST_ORIGIN_UPDATE
        or receipts[-1].get("checkpoint_sha256") != _sha256(checkpoint)
        or receipts[-1].get("complete_marker_sha256") != _sha256(complete_payload)
        or receipts[-1].get("policy_state_sha256")
        != payload.get("policy_state_sha256")
    ):
        raise Stage6SourceRepairError("Update 9 checkpoint identity drifted")
    return lineage_value, {
        "transaction_key": receipts[-1]["transaction_key"],
        "seed": FIXED_SEED,
        "update": LAST_ORIGIN_UPDATE,
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_sha256": _sha256(manifest_payload),
        "complete_marker_sha256": _sha256(complete_payload),
        "policy_state_sha256": payload["policy_state_sha256"],
    }


def _validate_prefix_semantics(
    prefix: Mapping[str, bytes],
    *,
    origin_immutable: Mapping[str, object],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    from lunar_exploration_ppo.configs.stage6 import parse_stage6_config_bytes
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        build_standard_training_transactions,
        verify_journal_checkpoint_bindings,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    config = parse_stage6_config_bytes(prefix["config.json"])
    transactions = build_standard_training_transactions(config)
    receipts = CheckpointReceiptIndex.verify_snapshot_bytes(
        prefix["checkpoints/index.jsonl"]
    )
    job_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["job-state.jsonl"])
    completed = verify_journal_checkpoint_bindings(
        job_rows,
        transactions,
        receipts,
        origin_immutable,
    )
    phase_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["phase-state.jsonl"])
    expected_keys = tuple(transaction.key for transaction in transactions[:9])
    if (
        len(receipts) != 9
        or tuple(row["transaction_key"] for row in receipts) != expected_keys
        or completed != expected_keys
        or len(job_rows) != 10
        or len(phase_rows) != 1
        or phase_rows[0].get("state") != "preflight"
        or any(
            phase_rows[0].get("bindings", {}).get(key) != value
            for key, value in origin_immutable.items()
        )
    ):
        raise Stage6SourceRepairError("accepted Update 1-9 prefix drifted")
    resource_rows = _strict_jsonl(
        prefix["resource_audit.jsonl"], label="resource audit prefix"
    )
    accepted = tuple(
        row
        for row in resource_rows
        if row.get("phase") == "accepted" and row.get("accepted") is True
    )
    update10 = tuple(
        row
        for row in resource_rows
        if row.get("transaction_key") == NEXT_TRANSACTION_KEY
    )
    if (
        len(resource_rows) != 29
        or len(accepted) != 9
        or tuple(row.get("update") for row in accepted) != tuple(range(1, 10))
        or len(update10) != 1
        or update10[0].get("attempt") != 1
        or update10[0].get("phase") != "pre"
        or update10[0].get("accepted") is not False
        or not isinstance(update10[0].get("resource"), Mapping)
        or update10[0]["resource"].get("passed") is not True  # type: ignore[index]
    ):
        raise Stage6SourceRepairError("Update 10 resource retry prefix drifted")
    return receipts, {
        "receipt_count": 9,
        "completed_transaction_count": 9,
        "job_row_count": 10,
        "phase_states": ["preflight"],
        "resource_row_count": 29,
        "next_resource_attempt": 2,
    }


def _validate_continuation_prefix_semantics(
    stage: Path,
    prefix: Mapping[str, bytes],
    *,
    origin_immutable: Mapping[str, object],
    at_creation: bool,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    from lunar_exploration_ppo.configs.stage6 import parse_stage6_config_bytes
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        build_standard_training_transactions,
        verify_journal_checkpoint_bindings,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    config = parse_stage6_config_bytes(prefix["config.json"])
    transactions = build_standard_training_transactions(config)
    receipts = CheckpointReceiptIndex.verify_snapshot_bytes(
        prefix["checkpoints/index.jsonl"]
    )
    job_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["job-state.jsonl"])
    completed = verify_journal_checkpoint_bindings(
        job_rows,
        transactions,
        receipts,
        origin_immutable,
    )
    phase_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["phase-state.jsonl"])
    expected_keys = tuple(transaction.key for transaction in transactions[:9])
    training_rows = _strict_jsonl(
        prefix["training_metrics.jsonl"],
        label="continuation training metric prefix",
    )
    math_rows = _strict_jsonl(
        prefix["math_audit.jsonl"],
        label="continuation math audit prefix",
    )
    checkpoint_rows = _strict_jsonl(
        prefix["checkpoint_audit.jsonl"],
        label="continuation checkpoint audit prefix",
    )
    resource_rows = _strict_jsonl(
        prefix["resource_audit.jsonl"],
        label="continuation resource audit prefix",
    )
    accepted = tuple(
        row
        for row in resource_rows
        if row.get("phase") == "accepted" and row.get("accepted") is True
    )
    update10 = tuple(
        row
        for row in resource_rows
        if row.get("transaction_key") == NEXT_TRANSACTION_KEY
    )
    segment_start = resource_rows[-2] if len(resource_rows) >= 2 else {}
    attempt2 = resource_rows[-1] if resource_rows else {}
    primary_resource = _FIXED_PREFIX_BINDINGS["resource_audit.jsonl"]
    if (
        len(receipts) != 9
        or tuple(row["transaction_key"] for row in receipts) != expected_keys
        or completed != expected_keys
        or len(job_rows) != 10
        or len(phase_rows) != 1
        or phase_rows[0].get("state") != "preflight"
        or any(
            phase_rows[0].get("bindings", {}).get(key) != value
            for key, value in origin_immutable.items()
        )
        or len(training_rows) != 9
        or len(math_rows) != 1
        or len(checkpoint_rows) != 1
        or len(resource_rows) != 31
        or len(accepted) != 9
        or tuple(row.get("update") for row in accepted) != tuple(range(1, 10))
        or len(update10) != 2
        or tuple(row.get("attempt") for row in update10) != (1, 2)
        or any(row.get("phase") != "pre" for row in update10)
        or any(row.get("accepted") is not False for row in update10)
        or any(
            not isinstance(row.get("resource"), Mapping)
            or row["resource"].get("passed") is not True  # type: ignore[index]
            for row in update10
        )
        or segment_start.get("schema_version")
        != "stage6_resource_lifecycle_segment/v1"
        or segment_start.get("phase") != "segment_start"
        or segment_start.get("segment_index") != 2
        or segment_start.get("prior_resource_log")
        != {
            "sha256": primary_resource["sha256"],
            "size_bytes": primary_resource["size_bytes"],
        }
        or attempt2 != update10[-1]
        or attempt2.get("attempt") != 2
        or attempt2.get("segment_index") != 2
    ):
        raise Stage6SourceRepairError("Update 10 second-cutover prefix drifted")
    update10_checkpoint = (
        stage / f"checkpoints/seed-{FIXED_SEED}/update-{FIRST_REPAIRED_UPDATE:08d}"
    )
    if at_creation and os.path.lexists(update10_checkpoint):
        raise Stage6SourceRepairError(
            "Update 10 checkpoint exists before continuation publication"
        )
    checkpoint_lineage, checkpoint_identity = _validate_checkpoint_prefix(
        prefix,
        config_sha256=str(origin_immutable["config_sha256"]),
        receipts=receipts,
    )
    return checkpoint_lineage, checkpoint_identity, {
        "receipt_count": 9,
        "completed_transaction_count": 9,
        "job_row_count": 10,
        "phase_states": ["preflight"],
        "training_row_count": 9,
        "math_row_count": 1,
        "checkpoint_audit_row_count": 1,
        "resource_row_count": 31,
        "failed_update": 10,
        "failed_resource_attempt": 2,
        "next_resource_attempt": 3,
    }


def _continuation_evidence_payloads(
    *,
    stage: Path,
    primary_payload: bytes,
) -> dict[str, bytes]:
    result = _bound_evidence_payloads(
        bindings=_CONTINUATION_EVIDENCE_BINDINGS,
        label="source-repair continuation evidence",
    )
    evidence = _strict_json(
        result["device_failure"],
        label="Update 10 device failure evidence",
    )
    expected_fields = {
        "schema_version",
        "formal_run_id",
        "accepted_prefix_updates",
        "failed_update",
        "failed_attempt",
        "failure_phase",
        "requested_policy_device",
        "actual_parameter_device",
        "cuda_current_device",
        "strict_torch_device_equality",
        "root_cause",
        "exception",
        "formal_side_effects",
        "parent_amendment",
        "resource_prefix",
        "stderr",
    }
    parent = evidence.get("parent_amendment")
    resource = evidence.get("resource_prefix")
    stderr = evidence.get("stderr")
    primary_binding = {
        "sha256": _sha256(primary_payload),
        "size_bytes": len(primary_payload),
    }
    continuation_resource = _CONTINUATION_PREFIX_BINDINGS[
        "resource_audit.jsonl"
    ]
    stderr_binding = _CONTINUATION_EVIDENCE_BINDINGS["stderr"]
    try:
        parent_path = lexical_absolute(str(parent["path"]))  # type: ignore[index]
        resource_path = lexical_absolute(str(resource["path"]))  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage6SourceRepairError(
            "Update 10 device failure path binding drifted"
        ) from exc
    if (
        set(evidence) != expected_fields
        or evidence.get("schema_version")
        != "stage6_update10_validation_device_failure/v1"
        or evidence.get("formal_run_id") != FIXED_FORMAL_RUN_ID
        or evidence.get("accepted_prefix_updates") != LAST_ORIGIN_UPDATE
        or evidence.get("failed_update") != FIRST_REPAIRED_UPDATE
        or evidence.get("failed_attempt") != 2
        or evidence.get("failure_phase") != "validation"
        or evidence.get("requested_policy_device") != "cuda"
        or evidence.get("actual_parameter_device") != "cuda:0"
        or evidence.get("cuda_current_device") != 0
        or evidence.get("strict_torch_device_equality") is not False
        or evidence.get("root_cause")
        != "bare_cuda_alias_compared_by_strict_torch_device_equality/v1"
        or evidence.get("exception")
        != "StandardEvaluationError: PPO evaluation policy device drifted"
        or evidence.get("formal_side_effects")
        != {
            "resource_pre_attempt2_written": True,
            "update10_checkpoint_receipt_written": False,
            "update10_training_metric_written": False,
        }
        or not isinstance(parent, Mapping)
        or parent_path != stage / SOURCE_REPAIR_AMENDMENT_NAME
        or {"sha256": parent.get("sha256"), "size_bytes": parent.get("size_bytes")}
        != primary_binding
        or not isinstance(resource, Mapping)
        or resource_path != stage / "resource_audit.jsonl"
        or {
            "sha256": resource.get("sha256"),
            "size_bytes": resource.get("size_bytes"),
        }
        != {
            "sha256": continuation_resource["sha256"],
            "size_bytes": continuation_resource["size_bytes"],
        }
        or not isinstance(stderr, Mapping)
        or stderr
        != {
            "path": stderr_binding["path"],
            "sha256": stderr_binding["sha256"],
            "size_bytes": stderr_binding["size_bytes"],
        }
    ):
        raise Stage6SourceRepairError("Update 10 device failure evidence drifted")
    return result


def _supplement_evidence_payloads() -> dict[str, bytes]:
    result = _bound_evidence_payloads(
        bindings=_SUPPLEMENT_EVIDENCE_BINDINGS,
        label="source-repair supplement evidence",
    )
    marker = (
        b"lunar_exploration_ppo.ppo.trainer.PPOTrainingError: initial ratio "
        b"mismatch: max_abs_error=1.26361847e-05"
    )
    if marker not in result["stderr"]:
        raise Stage6SourceRepairError(
            "Update 33 initial-ratio failure evidence drifted"
        )
    return result


def _external_json(payload: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Stage6SourceRepairError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise Stage6SourceRepairError(f"{label} schema drifted")
    return value


def _closure_evidence_payloads(
    *,
    supplement_payload: bytes,
) -> dict[str, bytes]:
    result = _bound_evidence_payloads(
        bindings=_CLOSURE_EVIDENCE_BINDINGS,
        label="source-repair closure evidence",
    )
    finding = _external_json(
        result["finding_review"],
        label="proxy-safety finding review",
    )
    required = finding.get("required_repair_semantics")
    required_semantics = {
        "classify_as_adverse_synthetic_proxy_finding": True,
        "machine_blocking_only_if_evidence_contract_fails": True,
        "physical_obstacle_cells_written": False,
        "physical_safety_claim_allowed": False,
        "preserve_per_split_method_and_total_counts": True,
        "preserve_safety_done_termination_classification": True,
        "preserve_true_safety_violation_count": True,
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
    }
    if (
        finding.get("schema_version")
        != "stage6_proxy_safety_gate_finding_review/v1"
        or finding.get("finding_id")
        != "stage6_terminal_proxy_safety_gate_contract_conflict/v1"
        or finding.get("review_kind")
        != "fresh_independent_contract_review/v1"
        or finding.get("severity") != "Critical"
        or finding.get("verdict") != "REJECT"
        or not isinstance(required, Mapping)
        or any(required.get(key) != value for key, value in required_semantics.items())
    ):
        raise Stage6SourceRepairError("proxy-safety finding review drifted")

    safe_stop = _external_json(
        result["safe_stop"],
        label="R15 safe-stop audit",
    )
    formal_prefix = safe_stop.get("formal_prefix")
    process_exit = safe_stop.get("process_exit")
    resource_binding = _CLOSURE_PREFIX_BINDINGS["resource_audit.jsonl"]
    last_resource_row = {
        "accepted": False,
        "attempt": 2,
        "phase": "pre",
        "segment_index": 4,
        "transaction_key": CLOSURE_NEXT_TRANSACTION_KEY,
        "update": FIRST_CLOSURE_UPDATE,
    }
    if (
        safe_stop.get("schema_version") != "stage6_r15_safe_stop_audit/v1"
        or safe_stop.get("formal_run_id") != FIXED_FORMAL_RUN_ID
        or safe_stop.get("recovery") != "r15"
        or safe_stop.get("reason")
        != "fresh_independent_review_critical_terminal_contract_conflict/v1"
        or safe_stop.get("source_repair_supplement_sha256")
        != _sha256(supplement_payload)
        or not isinstance(formal_prefix, Mapping)
        or formal_prefix.get("accepted_update_count") != LAST_CONTINUATION_UPDATE
        or formal_prefix.get("last_accepted_update") != LAST_CONTINUATION_UPDATE
        or formal_prefix.get("job_row_count") != 36
        or formal_prefix.get("training_row_count") != LAST_CONTINUATION_UPDATE
        or formal_prefix.get("resource_row_count") != 104
        or formal_prefix.get("resource_audit") != dict(resource_binding)
        or formal_prefix.get("last_resource_row") != last_resource_row
        or formal_prefix.get("stage6_lease_exists") is not False
        or formal_prefix.get("update33_checkpoint_exists") is not False
        or not isinstance(process_exit, Mapping)
        or process_exit.get("duplicate_runner_started") is not False
        or process_exit.get("remaining_child_count_after_cleanup") != 0
        or process_exit.get("runner_count_after_stop") != 0
    ):
        raise Stage6SourceRepairError("R15 safe-stop audit drifted")
    return result


def _frontier_recovery_evidence_payloads() -> dict[str, bytes]:
    result = _bound_evidence_payloads(
        bindings=_FRONTIER_RECOVERY_EVIDENCE_BINDINGS,
        label="source-repair frontier recovery evidence",
    )
    failure_marker = "production frontier is empty while observed opportunities remain"
    try:
        stderr = result["stderr"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage6SourceRepairError("Update 47 stderr is not UTF-8") from exc
    replay = _external_json(
        result["replay_exception"],
        label="Update 47 replay exception",
    )
    root_cause = _external_json(
        result["root_cause"],
        label="Update 47 root-cause analysis",
    )
    benchmark = _external_json(
        result["benchmark"],
        label="Update 47 final benchmark",
    )
    independent = _external_json(
        result["independent_review"],
        label="Update 47 independent review",
    )
    review_manifest = _external_json(
        result["review_manifest"],
        label="Update 47 review manifest",
    )
    source = _FRONTIER_RECOVERY_SOURCE_BINDINGS
    patch_binding = _FRONTIER_RECOVERY_EVIDENCE_BINDINGS["reviewed_patch"]
    benchmark_binding = _FRONTIER_RECOVERY_EVIDENCE_BINDINGS["benchmark"]
    files = review_manifest.get("files")
    if (
        failure_marker not in stderr
        or replay.get("schema_version") != "stage6_update47_debug_replay/v1"
        or replay.get("error_type") != "WorkerProcessError"
        or failure_marker not in str(replay.get("message"))
        or root_cause.get("schema_version")
        != "stage6_update47_frontier_root_cause/v1"
        or root_cause.get("formal_run_id") != FIXED_FORMAL_RUN_ID
        or root_cause.get("failed_update") != FIRST_FRONTIER_RECOVERY_UPDATE
        or root_cause.get("last_complete_update")
        != LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        or root_cause.get("formal_resume_authorized") is not False
        or root_cause.get("replay_result")
        != "exact_failure_reproduced_and_observed_state_captured"
        or benchmark.get("schema_version")
        != "stage6_update47_final_fix_benchmark/v1"
        or benchmark.get("frontier_source_sha256")
        != source["src/lunar_exploration_ppo/env/frontier.py"]["sha256"]
        or benchmark.get("public_candidate_count") != 3
        or benchmark.get("public_fallback_activated") is not True
        or type(benchmark.get("positive_scored")) is not int
        or int(benchmark["positive_scored"]) <= 0
        or independent.get("schema_version")
        != "stage6_update47_independent_review/v2"
        or independent.get("review_mode") != "read_only"
        or independent.get("spec_compliance") != "PASS"
        or independent.get("code_quality") != "PASS"
        or independent.get("finding_counts")
        != {"critical": 0, "important": 0, "minor": 0}
        or independent.get("review_package_sha256") != patch_binding["sha256"]
        or independent.get("review_package_size_bytes")
        != patch_binding["size_bytes"]
        or independent.get("frontier_source_sha256")
        != source["src/lunar_exploration_ppo/env/frontier.py"]["sha256"]
        or independent.get("stage2_test_source_sha256")
        != source["tests/ppo_highres_frontier/test_stage2_frontier.py"]["sha256"]
        or independent.get("benchmark_sha256") != benchmark_binding["sha256"]
        or independent.get("formal_recovery_authorization_may_be_prepared")
        is not True
        or independent.get("formal_runner_launch_authorized_by_this_record")
        is not False
        or review_manifest.get("schema_version")
        != "stage6_update47_fix_review_manifest/v2"
        or review_manifest.get("task_scope")
        != "bounded_irregular_frontier_sampling_update47_repair"
        or review_manifest.get("formal_replay_or_training_launched") is not False
        or not isinstance(files, Mapping)
        or files.get("review-package.diff")
        != {
            "sha256": patch_binding["sha256"],
            "size_bytes": patch_binding["size_bytes"],
        }
        or files.get("final-fix-benchmark.json")
        != {
            "sha256": benchmark_binding["sha256"],
            "size_bytes": benchmark_binding["size_bytes"],
        }
        or files.get("post-fix/frontier.py")
        != {
            "sha256": source["src/lunar_exploration_ppo/env/frontier.py"][
                "sha256"
            ]
        }
        or files.get("post-fix/env.py")
        != {"sha256": source["src/lunar_exploration_ppo/env/env.py"]["sha256"]}
        or files.get("post-fix/test_stage2_frontier.py")
        != {
            "sha256": source[
                "tests/ppo_highres_frontier/test_stage2_frontier.py"
            ]["sha256"]
        }
        or not isinstance(files.get("root-cause-analysis.json"), Mapping)
        or files["root-cause-analysis.json"].get("sha256")
        != _FRONTIER_RECOVERY_EVIDENCE_BINDINGS["root_cause"]["sha256"]
    ):
        raise Stage6SourceRepairError("Update 47 reviewed repair evidence drifted")
    return result


def _validate_supplement_checkpoint_prefix(
    prefix: Mapping[str, bytes],
    *,
    parent_context: "Stage6SourceRepairContext",
    receipts: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], dict[str, object]]:
    root = (
        f"checkpoints/seed-{FIXED_SEED}/"
        f"update-{LAST_CONTINUATION_UPDATE:08d}"
    )
    checkpoint = prefix[f"{root}/checkpoint.pt"]
    manifest_payload = prefix[f"{root}/manifest.json"]
    complete_payload = prefix[f"{root}/complete.json"]
    manifest = _strict_json(
        manifest_payload,
        label="Update 32 checkpoint manifest",
    )
    complete = _strict_json(
        complete_payload,
        label="Update 32 complete marker",
    )
    payload = _read_cutover_checkpoint_payload(checkpoint)
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise Stage6SourceRepairError("Update 32 checkpoint lineage is missing")
    lineage_value = dict(lineage)
    expected_lineage = parent_context.checkpoint_lineage_for_update(
        LAST_CONTINUATION_UPDATE
    )
    if (
        payload.get("update_step") != LAST_CONTINUATION_UPDATE
        or payload.get("config_sha256")
        != parent_context.origin_immutable_bindings.get("config_sha256")
        or not _is_sha256(payload.get("policy_state_sha256"))
        or lineage_value != expected_lineage
        or manifest.get("update_step") != LAST_CONTINUATION_UPDATE
        or manifest.get("config_sha256") != payload.get("config_sha256")
        or manifest.get("policy_state_sha256")
        != payload.get("policy_state_sha256")
        or manifest.get("lineage_sha256") != _canonical_sha256(expected_lineage)
        or manifest.get("checkpoint")
        != {
            "path": "checkpoint.pt",
            "sha256": _sha256(checkpoint),
            "size_bytes": len(checkpoint),
        }
        or complete.get("update_step") != LAST_CONTINUATION_UPDATE
        or complete.get("checkpoint_sha256") != _sha256(checkpoint)
        or complete.get("manifest_sha256") != _sha256(manifest_payload)
        or not receipts
        or receipts[-1].get("update") != LAST_CONTINUATION_UPDATE
        or receipts[-1].get("checkpoint_sha256") != _sha256(checkpoint)
        or receipts[-1].get("complete_marker_sha256")
        != _sha256(complete_payload)
        or receipts[-1].get("policy_state_sha256")
        != payload.get("policy_state_sha256")
    ):
        raise Stage6SourceRepairError("Update 32 checkpoint identity drifted")
    return lineage_value, {
        "transaction_key": receipts[-1]["transaction_key"],
        "seed": FIXED_SEED,
        "update": LAST_CONTINUATION_UPDATE,
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_sha256": _sha256(manifest_payload),
        "complete_marker_sha256": _sha256(complete_payload),
        "policy_state_sha256": payload["policy_state_sha256"],
        "lineage_sha256": _canonical_sha256(lineage_value),
    }


def _validate_supplement_prefix_semantics(
    stage: Path,
    prefix: Mapping[str, bytes],
    *,
    parent_context: "Stage6SourceRepairContext",
    at_creation: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    from lunar_exploration_ppo.configs.stage6 import parse_stage6_config_bytes
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        build_standard_training_transactions,
        verify_journal_checkpoint_bindings,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    config = parse_stage6_config_bytes(prefix["config.json"])
    transactions = build_standard_training_transactions(config)
    receipts = CheckpointReceiptIndex.verify_snapshot_bytes(
        prefix["checkpoints/index.jsonl"]
    )
    job_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["job-state.jsonl"])
    completed = verify_journal_checkpoint_bindings(
        job_rows,
        transactions,
        receipts,
        parent_context.origin_immutable_bindings,
    )
    phase_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["phase-state.jsonl"])
    training_rows = _strict_jsonl(
        prefix["training_metrics.jsonl"],
        label="supplement training metric prefix",
    )
    validation_rows = _strict_jsonl(
        prefix["validation_metrics.jsonl"],
        label="supplement validation metric prefix",
    )
    math_rows = _strict_jsonl(
        prefix["math_audit.jsonl"],
        label="supplement math audit prefix",
    )
    checkpoint_rows = _strict_jsonl(
        prefix["checkpoint_audit.jsonl"],
        label="supplement checkpoint audit prefix",
    )
    resource_rows = _strict_jsonl(
        prefix["resource_audit.jsonl"],
        label="supplement resource audit prefix",
    )
    accepted = tuple(
        row
        for row in resource_rows
        if row.get("phase") == "accepted" and row.get("accepted") is True
    )
    segment_rows = tuple(
        row for row in resource_rows if row.get("phase") == "segment_start"
    )
    update33 = tuple(
        row
        for row in resource_rows
        if row.get("transaction_key") == SUPPLEMENT_NEXT_TRANSACTION_KEY
    )
    phase_counts = {
        phase: sum(row.get("phase") == phase for row in resource_rows)
        for phase in ("segment_start", "pre", "post", "accepted")
    }
    expected_keys = tuple(
        transaction.key for transaction in transactions[:LAST_CONTINUATION_UPDATE]
    )
    if (
        len(receipts) != LAST_CONTINUATION_UPDATE
        or tuple(row["transaction_key"] for row in receipts) != expected_keys
        or completed != expected_keys
        or len(job_rows) != 36
        or len(phase_rows) != 1
        or phase_rows[0].get("state") != "preflight"
        or any(
            phase_rows[0].get("bindings", {}).get(key) != value
            for key, value in parent_context.origin_immutable_bindings.items()
        )
        or len(training_rows) != LAST_CONTINUATION_UPDATE
        or tuple(row.get("update") for row in training_rows)
        != tuple(range(1, LAST_CONTINUATION_UPDATE + 1))
        or len(validation_rows) != 3
        or tuple(row.get("update") for row in validation_rows) != (10, 20, 30)
        or len(math_rows) != 2
        or len(checkpoint_rows) != 2
        or len(resource_rows) != 102
        or phase_counts
        != {"segment_start": 3, "pre": 35, "post": 32, "accepted": 32}
        or tuple(row.get("segment_index") for row in segment_rows) != (1, 2, 3)
        or len(accepted) != LAST_CONTINUATION_UPDATE
        or tuple(row.get("update") for row in accepted)
        != tuple(range(1, LAST_CONTINUATION_UPDATE + 1))
        or len(update33) != 1
        or update33[0].get("update") != FIRST_SUPPLEMENT_UPDATE
        or update33[0].get("attempt") != 1
        or update33[0].get("phase") != "pre"
        or update33[0].get("accepted") is not False
        or not isinstance(update33[0].get("resource"), Mapping)
        or update33[0]["resource"].get("passed") is not True  # type: ignore[index]
    ):
        raise Stage6SourceRepairError("accepted Update 1-32 prefix drifted")
    update33_checkpoint = (
        stage
        / f"checkpoints/seed-{FIXED_SEED}/update-{FIRST_SUPPLEMENT_UPDATE:08d}"
    )
    if at_creation and os.path.lexists(update33_checkpoint):
        raise Stage6SourceRepairError(
            "Update 33 checkpoint exists before supplement publication"
        )
    checkpoint_lineage, checkpoint_identity = (
        _validate_supplement_checkpoint_prefix(
            prefix,
            parent_context=parent_context,
            receipts=receipts,
        )
    )
    if checkpoint_lineage != parent_context.checkpoint_lineage_for_update(
        LAST_CONTINUATION_UPDATE
    ):
        raise Stage6SourceRepairError("Update 32 historical lineage drifted")
    return checkpoint_identity, {
        "receipt_count": LAST_CONTINUATION_UPDATE,
        "completed_transaction_count": LAST_CONTINUATION_UPDATE,
        "job_row_count": 36,
        "phase_states": ["preflight"],
        "training_row_count": LAST_CONTINUATION_UPDATE,
        "validation_row_count": 3,
        "math_row_count": 2,
        "checkpoint_audit_row_count": 2,
        "resource_row_count": 102,
        "resource_phase_counts": phase_counts,
        "accepted_prefix_updates": LAST_CONTINUATION_UPDATE,
        "failed_update": FIRST_SUPPLEMENT_UPDATE,
        "failed_resource_attempt": 1,
        "next_resource_attempt": 2,
        "next_transaction_key": SUPPLEMENT_NEXT_TRANSACTION_KEY,
    }


def _validate_closure_prefix_semantics(
    stage: Path,
    prefix: Mapping[str, bytes],
    *,
    continuation_context: "Stage6SourceRepairContext",
    supplement_context: "Stage6SourceRepairContext",
    at_creation: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    supplement_prefix = _supplement_prefix_payloads(stage, allow_growth=True)
    checkpoint_identity, base_semantics = _validate_supplement_prefix_semantics(
        stage,
        supplement_prefix,
        parent_context=continuation_context,
        at_creation=False,
    )
    if supplement_context.checkpoint_lineage_for_update(
        LAST_CONTINUATION_UPDATE
    ) != continuation_context.checkpoint_lineage_for_update(LAST_CONTINUATION_UPDATE):
        raise Stage6SourceRepairError("closure historical lineage drifted")
    for relative, payload in prefix.items():
        if relative != "resource_audit.jsonl" and payload != supplement_prefix[relative]:
            raise Stage6SourceRepairError(
                f"source-repair closure prefix drifted: {relative}"
            )
    prior_rows = _strict_jsonl(
        supplement_prefix["resource_audit.jsonl"],
        label="supplement resource audit prefix",
    )
    resource_rows = _strict_jsonl(
        prefix["resource_audit.jsonl"],
        label="closure resource audit prefix",
    )
    if resource_rows[: len(prior_rows)] != prior_rows:
        raise Stage6SourceRepairError("closure resource predecessor drifted")
    delta = resource_rows[len(prior_rows) :]
    segment = delta[0] if len(delta) == 2 else {}
    attempt = delta[1] if len(delta) == 2 else {}
    phase_counts = {
        phase: sum(row.get("phase") == phase for row in resource_rows)
        for phase in ("segment_start", "pre", "post", "accepted")
    }
    prior_resource = _SUPPLEMENT_PREFIX_BINDINGS["resource_audit.jsonl"]
    if (
        len(prior_rows) != 102
        or len(resource_rows) != 104
        or phase_counts
        != {"segment_start": 4, "pre": 36, "post": 32, "accepted": 32}
        or segment.get("schema_version")
        != "stage6_resource_lifecycle_segment/v1"
        or segment.get("phase") != "segment_start"
        or segment.get("segment_index") != 4
        or segment.get("prior_resource_log")
        != {
            "sha256": prior_resource["sha256"],
            "size_bytes": prior_resource["size_bytes"],
        }
        or attempt.get("schema_version") != "stage6_resource_attempt/v1"
        or attempt.get("kind") != "update"
        or attempt.get("transaction_key") != CLOSURE_NEXT_TRANSACTION_KEY
        or attempt.get("seed") != FIXED_SEED
        or attempt.get("update") != FIRST_CLOSURE_UPDATE
        or attempt.get("attempt") != 2
        or attempt.get("phase") != "pre"
        or attempt.get("accepted") is not False
        or attempt.get("segment_index") != 4
        or not isinstance(attempt.get("resource"), Mapping)
        or attempt["resource"].get("passed") is not True  # type: ignore[index]
    ):
        raise Stage6SourceRepairError("Update 33 closure prefix drifted")
    update33_checkpoint = (
        stage / f"checkpoints/seed-{FIXED_SEED}/update-{FIRST_CLOSURE_UPDATE:08d}"
    )
    if at_creation and os.path.lexists(update33_checkpoint):
        raise Stage6SourceRepairError(
            "closure requires a stopped, Update 33 checkpoint-free boundary"
        )
    return checkpoint_identity, {
        **base_semantics,
        "resource_row_count": 104,
        "resource_phase_counts": phase_counts,
        "failed_resource_attempt": 2,
        "next_resource_attempt": 3,
        "next_segment_index": 5,
        "next_transaction_key": CLOSURE_NEXT_TRANSACTION_KEY,
    }


def _validate_frontier_recovery_checkpoint_prefix(
    prefix: Mapping[str, bytes],
    *,
    parent_context: "Stage6SourceRepairContext",
    receipts: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], dict[str, object]]:
    root = (
        f"checkpoints/seed-{FIXED_SEED}/"
        f"update-{LAST_FRONTIER_RECOVERY_PARENT_UPDATE:08d}"
    )
    checkpoint = prefix[f"{root}/checkpoint.pt"]
    manifest_payload = prefix[f"{root}/manifest.json"]
    complete_payload = prefix[f"{root}/complete.json"]
    manifest = _strict_json(
        manifest_payload,
        label="Update 46 checkpoint manifest",
    )
    complete = _strict_json(
        complete_payload,
        label="Update 46 complete marker",
    )
    payload = _read_cutover_checkpoint_payload(checkpoint)
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise Stage6SourceRepairError("Update 46 checkpoint lineage is missing")
    lineage_value = dict(lineage)
    expected_lineage = parent_context.checkpoint_lineage_for_update(
        LAST_FRONTIER_RECOVERY_PARENT_UPDATE
    )
    if (
        payload.get("update_step") != LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        or payload.get("config_sha256")
        != parent_context.current_immutable_bindings.get("config_sha256")
        or not _is_sha256(payload.get("policy_state_sha256"))
        or lineage_value != expected_lineage
        or manifest.get("update_step") != LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        or manifest.get("config_sha256") != payload.get("config_sha256")
        or manifest.get("policy_state_sha256")
        != payload.get("policy_state_sha256")
        or manifest.get("lineage_sha256") != _canonical_sha256(expected_lineage)
        or manifest.get("checkpoint")
        != {
            "path": "checkpoint.pt",
            "sha256": _sha256(checkpoint),
            "size_bytes": len(checkpoint),
        }
        or complete.get("update_step") != LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        or complete.get("checkpoint_sha256") != _sha256(checkpoint)
        or complete.get("manifest_sha256") != _sha256(manifest_payload)
        or not receipts
        or receipts[-1].get("update") != LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        or receipts[-1].get("checkpoint_sha256") != _sha256(checkpoint)
        or receipts[-1].get("complete_marker_sha256")
        != _sha256(complete_payload)
        or receipts[-1].get("policy_state_sha256")
        != payload.get("policy_state_sha256")
    ):
        raise Stage6SourceRepairError("Update 46 checkpoint identity drifted")
    return lineage_value, {
        "transaction_key": receipts[-1]["transaction_key"],
        "seed": FIXED_SEED,
        "update": LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_sha256": _sha256(manifest_payload),
        "complete_marker_sha256": _sha256(complete_payload),
        "policy_state_sha256": payload["policy_state_sha256"],
        "lineage_sha256": _canonical_sha256(lineage_value),
    }


def _validate_frontier_recovery_prefix_semantics(
    stage: Path,
    prefix: Mapping[str, bytes],
    *,
    parent_context: "Stage6SourceRepairContext",
    at_creation: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    from lunar_exploration_ppo.configs.stage6 import parse_stage6_config_bytes
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        build_standard_training_transactions,
        verify_journal_checkpoint_bindings,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    config = parse_stage6_config_bytes(prefix["config.json"])
    transactions = build_standard_training_transactions(config)
    receipts = CheckpointReceiptIndex.verify_snapshot_bytes(
        prefix["checkpoints/index.jsonl"]
    )
    job_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["job-state.jsonl"])
    completed = verify_journal_checkpoint_bindings(
        job_rows,
        transactions,
        receipts,
        parent_context.origin_immutable_bindings,
    )
    phase_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["phase-state.jsonl"])
    training_rows = _strict_jsonl(
        prefix["training_metrics.jsonl"],
        label="frontier recovery training metric prefix",
    )
    validation_rows = _strict_jsonl(
        prefix["validation_metrics.jsonl"],
        label="frontier recovery validation metric prefix",
    )
    math_rows = _strict_jsonl(
        prefix["math_audit.jsonl"],
        label="frontier recovery math audit prefix",
    )
    checkpoint_rows = _strict_jsonl(
        prefix["checkpoint_audit.jsonl"],
        label="frontier recovery checkpoint audit prefix",
    )
    resource_rows = _strict_jsonl(
        prefix["resource_audit.jsonl"],
        label="frontier recovery resource audit prefix",
    )
    accepted = tuple(
        row
        for row in resource_rows
        if row.get("phase") == "accepted" and row.get("accepted") is True
    )
    segment_rows = tuple(
        row for row in resource_rows if row.get("phase") == "segment_start"
    )
    update47 = tuple(
        row
        for row in resource_rows
        if row.get("transaction_key") == FRONTIER_RECOVERY_NEXT_TRANSACTION_KEY
    )
    phase_counts = {
        phase: sum(row.get("phase") == phase for row in resource_rows)
        for phase in ("segment_start", "pre", "post", "accepted")
    }
    expected_transactions = transactions[:LAST_FRONTIER_RECOVERY_PARENT_UPDATE]
    expected_keys = tuple(transaction.key for transaction in expected_transactions)
    expected_job_row_count = sum(
        len(transaction.commit_states) for transaction in expected_transactions
    )
    segment = segment_rows[-1] if segment_rows else {}
    attempt = update47[0] if len(update47) == 1 else {}
    prior_resource = _CLOSURE_PREFIX_BINDINGS["resource_audit.jsonl"]
    if (
        len(receipts) != LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        or tuple(row["transaction_key"] for row in receipts) != expected_keys
        or completed != expected_keys
        or len(job_rows) != expected_job_row_count
        or len(phase_rows) != 1
        or phase_rows[0].get("state") != "preflight"
        or any(
            phase_rows[0].get("bindings", {}).get(key) != value
            for key, value in parent_context.origin_immutable_bindings.items()
        )
        or len(training_rows) != LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        or tuple(row.get("update") for row in training_rows)
        != tuple(range(1, LAST_FRONTIER_RECOVERY_PARENT_UPDATE + 1))
        or len(validation_rows) != 4
        or tuple(row.get("update") for row in validation_rows) != (10, 20, 30, 40)
        or len(math_rows) != 2
        or len(checkpoint_rows) != 2
        or len(resource_rows) != 148
        or phase_counts
        != {"segment_start": 5, "pre": 51, "post": 46, "accepted": 46}
        or tuple(row.get("segment_index") for row in segment_rows)
        != (1, 2, 3, 4, 5)
        or len(accepted) != LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        or tuple(row.get("update") for row in accepted)
        != tuple(range(1, LAST_FRONTIER_RECOVERY_PARENT_UPDATE + 1))
        or segment.get("schema_version")
        != "stage6_resource_lifecycle_segment/v1"
        or segment.get("phase") != "segment_start"
        or segment.get("segment_index") != 5
        or segment.get("prior_resource_log")
        != {
            "sha256": prior_resource["sha256"],
            "size_bytes": prior_resource["size_bytes"],
        }
        or attempt.get("schema_version") != "stage6_resource_attempt/v1"
        or attempt.get("kind") != "update"
        or attempt.get("seed") != FIXED_SEED
        or attempt.get("update") != FIRST_FRONTIER_RECOVERY_UPDATE
        or attempt.get("attempt") != 1
        or attempt.get("phase") != "pre"
        or attempt.get("accepted") is not False
        or attempt.get("segment_index") != 5
        or not isinstance(attempt.get("resource"), Mapping)
        or attempt["resource"].get("passed") is not True  # type: ignore[index]
    ):
        raise Stage6SourceRepairError("accepted Update 1-46 prefix drifted")
    update47_checkpoint = (
        stage
        / f"checkpoints/seed-{FIXED_SEED}/"
        f"update-{FIRST_FRONTIER_RECOVERY_UPDATE:08d}"
    )
    if at_creation and os.path.lexists(update47_checkpoint):
        raise Stage6SourceRepairError(
            "frontier recovery requires an Update 47 checkpoint-free boundary"
        )
    checkpoint_lineage, checkpoint_identity = (
        _validate_frontier_recovery_checkpoint_prefix(
            prefix,
            parent_context=parent_context,
            receipts=receipts,
        )
    )
    if checkpoint_lineage != parent_context.checkpoint_lineage_for_update(
        LAST_FRONTIER_RECOVERY_PARENT_UPDATE
    ):
        raise Stage6SourceRepairError("Update 46 historical lineage drifted")
    return checkpoint_identity, {
        "receipt_count": LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
        "completed_transaction_count": LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
        "job_row_count": expected_job_row_count,
        "phase_states": ["preflight"],
        "training_row_count": LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
        "validation_row_count": 4,
        "math_row_count": 2,
        "checkpoint_audit_row_count": 2,
        "resource_row_count": 148,
        "resource_phase_counts": phase_counts,
        "accepted_prefix_updates": LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
        "failed_update": FIRST_FRONTIER_RECOVERY_UPDATE,
        "failed_resource_attempt": 1,
        "next_resource_attempt": 2,
        "next_segment_index": 6,
        "next_transaction_key": FRONTIER_RECOVERY_NEXT_TRANSACTION_KEY,
    }


def _validate_sensor_acceleration_checkpoint_prefix(
    prefix: Mapping[str, bytes],
    *,
    parent_context: "Stage6SourceRepairContext",
    receipts: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], dict[str, object]]:
    root = (
        f"checkpoints/seed-{FIXED_SEED}/"
        f"update-{LAST_SENSOR_ACCELERATION_PARENT_UPDATE:08d}"
    )
    checkpoint = prefix[f"{root}/checkpoint.pt"]
    manifest_payload = prefix[f"{root}/manifest.json"]
    complete_payload = prefix[f"{root}/complete.json"]
    manifest = _strict_json(manifest_payload, label="Update 49 checkpoint manifest")
    complete = _strict_json(complete_payload, label="Update 49 complete marker")
    payload = _read_cutover_checkpoint_payload(checkpoint)
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise Stage6SourceRepairError("Update 49 checkpoint lineage is missing")
    lineage_value = dict(lineage)
    expected_lineage = parent_context.checkpoint_lineage_for_update(
        LAST_SENSOR_ACCELERATION_PARENT_UPDATE
    )
    if (
        payload.get("update_step") != LAST_SENSOR_ACCELERATION_PARENT_UPDATE
        or payload.get("config_sha256")
        != parent_context.current_immutable_bindings.get("config_sha256")
        or not _is_sha256(payload.get("policy_state_sha256"))
        or lineage_value != expected_lineage
        or manifest.get("update_step") != LAST_SENSOR_ACCELERATION_PARENT_UPDATE
        or manifest.get("config_sha256") != payload.get("config_sha256")
        or manifest.get("policy_state_sha256") != payload.get("policy_state_sha256")
        or manifest.get("lineage_sha256") != _canonical_sha256(expected_lineage)
        or manifest.get("checkpoint")
        != {
            "path": "checkpoint.pt",
            "sha256": _sha256(checkpoint),
            "size_bytes": len(checkpoint),
        }
        or complete.get("update_step") != LAST_SENSOR_ACCELERATION_PARENT_UPDATE
        or complete.get("checkpoint_sha256") != _sha256(checkpoint)
        or complete.get("manifest_sha256") != _sha256(manifest_payload)
        or not receipts
        or receipts[-1].get("update") != LAST_SENSOR_ACCELERATION_PARENT_UPDATE
        or receipts[-1].get("checkpoint_sha256") != _sha256(checkpoint)
        or receipts[-1].get("complete_marker_sha256")
        != _sha256(complete_payload)
        or receipts[-1].get("policy_state_sha256")
        != payload.get("policy_state_sha256")
    ):
        raise Stage6SourceRepairError("Update 49 checkpoint identity drifted")
    return lineage_value, {
        "transaction_key": receipts[-1]["transaction_key"],
        "seed": FIXED_SEED,
        "update": LAST_SENSOR_ACCELERATION_PARENT_UPDATE,
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_sha256": _sha256(manifest_payload),
        "complete_marker_sha256": _sha256(complete_payload),
        "policy_state_sha256": payload["policy_state_sha256"],
        "lineage_sha256": _canonical_sha256(lineage_value),
    }


def _sensor_acceleration_process_parent_map() -> Mapping[int, int]:
    from lunar_exploration_ppo.utils.resources import snapshot_process_parent_map

    return snapshot_process_parent_map()


def _require_sensor_acceleration_process_tree_stopped(
    prior_root_pid: int,
    *,
    process_parent_map_provider: Callable[[], Mapping[int, int]] | None = None,
    snapshot_identity_pid: int | None = None,
    platform_system_provider: Callable[[], str] | None = None,
) -> None:
    if type(prior_root_pid) is not int or prior_root_pid <= 0:
        raise Stage6SourceRepairError(
            "sensor acceleration process snapshot is uncertain"
        )
    identity_pid = os.getpid() if snapshot_identity_pid is None else snapshot_identity_pid
    if type(identity_pid) is not int or identity_pid <= 0:
        raise Stage6SourceRepairError(
            "sensor acceleration process snapshot is uncertain"
        )
    provider = (
        _sensor_acceleration_process_parent_map
        if process_parent_map_provider is None
        else process_parent_map_provider
    )
    system_provider = (
        platform.system if platform_system_provider is None else platform_system_provider
    )
    try:
        system_name = system_provider()
        if not isinstance(system_name, str) or not system_name:
            raise ValueError("platform identity is invalid")
        snapshot = provider()
        if not isinstance(snapshot, Mapping):
            raise TypeError("process parent map snapshot must be a mapping")
        parents: dict[int, int] = {}
        for pid, parent_pid in snapshot.items():
            if type(pid) is not int or pid <= 0:
                raise ValueError("invalid process snapshot PID")
            if type(parent_pid) is not int or parent_pid < 0:
                raise ValueError("invalid process snapshot parent PID")
            parents[pid] = parent_pid
        if identity_pid not in parents:
            raise ValueError("snapshot does not contain the probing process")

        old_tree_members: set[int] = set()
        for pid in parents:
            if pid == prior_root_pid:
                old_tree_members.add(pid)
                continue
            current = pid
            visited: set[int] = set()
            while current in parents:
                if current in visited:
                    raise ValueError("process parent map contains a cycle")
                visited.add(current)
                parent_pid = parents[current]
                if parent_pid == prior_root_pid:
                    old_tree_members.add(pid)
                    break
                current = parent_pid
    except Stage6SourceRepairError:
        raise
    except Exception as exc:
        raise Stage6SourceRepairError(
            "sensor acceleration process snapshot is uncertain"
        ) from exc
    if old_tree_members:
        raise Stage6SourceRepairError(
            "sensor acceleration previous process tree is still alive"
        )
    if system_name.casefold() != "windows":
        raise Stage6SourceRepairError(
            "sensor acceleration process identity/reparent unverifiable"
        )


def _validate_sensor_acceleration_prefix_semantics(
    stage: Path,
    prefix: Mapping[str, bytes],
    *,
    parent_context: "Stage6SourceRepairContext",
    at_creation: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    from lunar_exploration_ppo.configs.stage6 import parse_stage6_config_bytes
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        build_standard_training_transactions,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    config = parse_stage6_config_bytes(prefix["config.json"])
    transactions = build_standard_training_transactions(config)
    receipts = CheckpointReceiptIndex.verify_snapshot_bytes(
        prefix["checkpoints/index.jsonl"]
    )
    job_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["job-state.jsonl"])
    phase_rows = Stage6StateJournal.verify_snapshot_bytes(prefix["phase-state.jsonl"])
    training_rows = _strict_jsonl(
        prefix["training_metrics.jsonl"],
        label="sensor acceleration training metric prefix",
    )
    validation_rows = _strict_jsonl(
        prefix["validation_metrics.jsonl"],
        label="sensor acceleration validation metric prefix",
    )
    math_rows = _strict_jsonl(
        prefix["math_audit.jsonl"],
        label="sensor acceleration math audit prefix",
    )
    checkpoint_rows = _strict_jsonl(
        prefix["checkpoint_audit.jsonl"],
        label="sensor acceleration checkpoint audit prefix",
    )
    resource_rows = _strict_jsonl(
        prefix["resource_audit.jsonl"],
        label="sensor acceleration resource audit prefix",
    )
    accepted = tuple(
        row
        for row in resource_rows
        if row.get("phase") == "accepted" and row.get("accepted") is True
    )
    segment_rows = tuple(
        row for row in resource_rows if row.get("phase") == "segment_start"
    )
    update50_rows = tuple(
        row
        for row in resource_rows
        if row.get("transaction_key") == SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY
    )
    phase_counts = {
        phase: sum(row.get("phase") == phase for row in resource_rows)
        for phase in ("segment_start", "pre", "post", "accepted")
    }
    expected_transactions = transactions[:LAST_SENSOR_ACCELERATION_PARENT_UPDATE]
    expected_keys = tuple(transaction.key for transaction in expected_transactions)
    expected_job_row_count = sum(
        len(transaction.commit_states) for transaction in expected_transactions
    )
    attempt = update50_rows[0] if len(update50_rows) == 1 else {}
    sixth_segment_starts = tuple(
        (index, row)
        for index, row in enumerate(resource_rows)
        if row.get("phase") == "segment_start" and row.get("segment_index") == 6
    )
    sixth_segment_offset, sixth_segment = (
        sixth_segment_starts[0] if len(sixth_segment_starts) == 1 else (-1, {})
    )
    sixth_segment_id = sixth_segment.get("segment_id")
    sixth_segment_binding_is_valid = (
        isinstance(sixth_segment_id, str)
        and bool(sixth_segment_id)
        and sixth_segment_offset >= 0
        and all(
            row.get("segment_index") == 6
            and row.get("segment_id") == sixth_segment_id
            for row in resource_rows[sixth_segment_offset:]
        )
    )
    prior_root_pid = sixth_segment.get("root_pid")
    resource_snapshots: list[object] = []
    if sixth_segment_offset >= 0:
        for row in resource_rows[sixth_segment_offset:]:
            if row.get("phase") == "segment_start":
                resource_snapshots.append(row.get("first_sample"))
            elif row.get("phase") == "accepted":
                resource_snapshots.extend((row.get("pre"), row.get("post")))
            else:
                resource_snapshots.append(row.get("resource"))
    prior_root_binding_is_valid = (
        type(prior_root_pid) is int
        and prior_root_pid > 0
        and bool(resource_snapshots)
        and all(
            isinstance(snapshot, Mapping)
            and snapshot.get("rss_root_pid") == prior_root_pid
            for snapshot in resource_snapshots
        )
    )
    if (
        len(receipts) != LAST_SENSOR_ACCELERATION_PARENT_UPDATE
        or tuple(row["transaction_key"] for row in receipts) != expected_keys
        or len(job_rows) != expected_job_row_count
        or not job_rows
        or job_rows[-1].get("state")
        != f"seed_{FIXED_SEED}_training_update_49"
        or len(phase_rows) != 1
        or phase_rows[0].get("state") != "preflight"
        or len(training_rows) != LAST_SENSOR_ACCELERATION_PARENT_UPDATE
        or tuple(row.get("update") for row in training_rows)
        != tuple(range(1, LAST_SENSOR_ACCELERATION_PARENT_UPDATE + 1))
        or len(validation_rows) != 4
        or tuple(row.get("update") for row in validation_rows) != (10, 20, 30, 40)
        or len(math_rows) != 2
        or len(checkpoint_rows) != 2
        or len(resource_rows) != 159
        or phase_counts
        != {"segment_start": 6, "pre": 55, "post": 49, "accepted": 49}
        or tuple(row.get("segment_index") for row in segment_rows)
        != (1, 2, 3, 4, 5, 6)
        or not sixth_segment_binding_is_valid
        or not prior_root_binding_is_valid
        or len(accepted) != LAST_SENSOR_ACCELERATION_PARENT_UPDATE
        or tuple(row.get("update") for row in accepted)
        != tuple(range(1, LAST_SENSOR_ACCELERATION_PARENT_UPDATE + 1))
        or attempt.get("schema_version") != "stage6_resource_attempt/v1"
        or attempt.get("kind") != "update"
        or attempt.get("seed") != FIXED_SEED
        or attempt.get("update") != FIRST_SENSOR_ACCELERATION_UPDATE
        or attempt.get("attempt") != SENSOR_ACCELERATION_FAILED_RESOURCE_ATTEMPT
        or attempt.get("phase") != "pre"
        or attempt.get("accepted") is not False
        or attempt.get("segment_id") != sixth_segment_id
        or attempt.get("segment_index") != 6
        or not isinstance(attempt.get("resource"), Mapping)
        or attempt["resource"].get("passed") is not True  # type: ignore[index]
    ):
        raise Stage6SourceRepairError(
            "accepted Update 1-49 sensor acceleration prefix drifted"
        )
    if at_creation:
        if _stage6_run_lease_is_active(stage) or os.path.lexists(
            stage
            / f"checkpoints/seed-{FIXED_SEED}/"
            f"update-{FIRST_SENSOR_ACCELERATION_UPDATE:08d}"
        ):
            raise Stage6SourceRepairError(
                "sensor acceleration requires a stopped Update 50 checkpoint-free boundary"
            )
        _require_sensor_acceleration_process_tree_stopped(prior_root_pid)
    checkpoint_lineage, checkpoint_identity = (
        _validate_sensor_acceleration_checkpoint_prefix(
            prefix,
            parent_context=parent_context,
            receipts=receipts,
        )
    )
    if checkpoint_lineage != parent_context.checkpoint_lineage_for_update(49):
        raise Stage6SourceRepairError("Update 49 historical lineage drifted")
    return checkpoint_identity, {
        "receipt_count": 49,
        "completed_transaction_count": 49,
        "job_row_count": expected_job_row_count,
        "phase_states": ["preflight"],
        "training_row_count": 49,
        "validation_row_count": 4,
        "math_row_count": 2,
        "checkpoint_audit_row_count": 2,
        "resource_row_count": 159,
        "resource_phase_counts": phase_counts,
        "accepted_prefix_updates": 49,
        "failed_update": 50,
        "failed_resource_attempt": SENSOR_ACCELERATION_FAILED_RESOURCE_ATTEMPT,
        "previous_resource_root_pid": prior_root_pid,
        "next_resource_attempt": SENSOR_ACCELERATION_NEXT_RESOURCE_ATTEMPT,
        "next_segment_index": SENSOR_ACCELERATION_NEXT_SEGMENT_INDEX,
        "next_transaction_key": SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY,
    }


def _stage6_run_lease_is_active(stage: Path) -> bool:
    lease_path = stage.parent / ".stage6.lease"
    if not os.path.lexists(lease_path):
        return False
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    lease = RunLease(lease_path)
    try:
        lease.acquire()
    except RunLeaseError:
        return True
    try:
        lease.require_current()
    except RunLeaseError as exc:
        raise Stage6SourceRepairError(
            "Stage 6 run lease probe lost ownership"
        ) from exc
    finally:
        try:
            lease.release()
        except RunLeaseError as exc:
            raise Stage6SourceRepairError(
                "Stage 6 run lease probe release failed"
            ) from exc
    return False


def _validated_identities(
    *,
    stage: Path,
    prefix: Mapping[str, bytes],
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    from lunar_exploration_ppo.ppo.standard_training import (
        _immutable_bindings_valid,
        _review_authorization_immutable_bindings,
    )
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        validate_stage6_verified_review_authorization,
    )

    lineage_payload = prefix["lineage_audit.json"]
    lineage = _strict_json(lineage_payload, label="origin lineage audit")
    if (
        _sha256(lineage_payload) != _FIXED_LINEAGE_AUDIT_SHA256
        or set(lineage)
        != {
            "schema_version",
            "execution_identity",
            "immutable_bindings",
            "verified_review_authorization",
        }
        or lineage.get("schema_version") != "stage6_lineage_audit/v2"
        or not isinstance(lineage.get("execution_identity"), Mapping)
        or not isinstance(lineage.get("immutable_bindings"), Mapping)
        or not isinstance(lineage.get("verified_review_authorization"), Mapping)
    ):
        raise Stage6SourceRepairError("origin lineage audit drifted")
    origin_identity = dict(lineage["execution_identity"])
    origin_immutable = dict(lineage["immutable_bindings"])
    origin_review = dict(lineage["verified_review_authorization"])
    current_identity = dict(current_execution_identity)
    current_review = dict(current_verified_review_authorization)
    current_immutable = dict(current_immutable_bindings)
    try:
        validate_stage6_verified_review_authorization(
            origin_review,
            execution_identity=origin_identity,
            formal_run_id=stage.parent.name,
        )
        validate_stage6_verified_review_authorization(
            current_review,
            execution_identity=current_identity,
            formal_run_id=stage.parent.name,
        )
    except (Stage6WorkflowError, TypeError, ValueError) as exc:
        raise Stage6SourceRepairError("source-repair review identity drifted") from exc
    if (
        not _immutable_bindings_valid(origin_immutable)
        or not _immutable_bindings_valid(current_immutable)
        or _canonical_sha256(origin_immutable)
        != _FIXED_ORIGIN_IMMUTABLE_SHA256
    ):
        raise Stage6SourceRepairError("source-repair immutable identity drifted")
    expected_current = {
        key: current_identity.get(key)
        for key in (
            "config_sha256",
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
        )
    }
    expected_current["stage5_gate_sha256"] = origin_immutable["stage5_gate_sha256"]
    expected_current.update(_review_authorization_immutable_bindings(current_review))
    stable = (
        "config_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
        "stage5_gate_sha256",
        "formal_run_id",
    )
    if (
        current_immutable != expected_current
        or any(origin_immutable.get(key) != current_immutable.get(key) for key in stable)
        or origin_identity.get("source_set_sha256")
        == current_identity.get("source_set_sha256")
        or origin_review == current_review
    ):
        raise Stage6SourceRepairError("source repair is not an exact reviewed source cutover")
    return origin_identity, origin_immutable, origin_review, current_identity, current_immutable


def _validated_successor_identity(
    *,
    stage: Path,
    parent_execution_identity: Mapping[str, object],
    parent_verified_review_authorization: Mapping[str, object],
    parent_immutable_bindings: Mapping[str, object],
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    from lunar_exploration_ppo.ppo.standard_training import (
        _immutable_bindings_valid,
        _review_authorization_immutable_bindings,
    )
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6WorkflowError,
        validate_stage6_verified_review_authorization,
    )

    parent_identity = dict(parent_execution_identity)
    parent_review = dict(parent_verified_review_authorization)
    parent_immutable = dict(parent_immutable_bindings)
    current_identity = dict(current_execution_identity)
    current_review = dict(current_verified_review_authorization)
    current_immutable = dict(current_immutable_bindings)
    try:
        validate_stage6_verified_review_authorization(
            current_review,
            execution_identity=current_identity,
            formal_run_id=stage.parent.name,
        )
    except (Stage6WorkflowError, TypeError, ValueError) as exc:
        raise Stage6SourceRepairError(
            "source-repair continuation review identity drifted"
        ) from exc
    if not _immutable_bindings_valid(parent_immutable) or not _immutable_bindings_valid(
        current_immutable
    ):
        raise Stage6SourceRepairError(
            "source-repair continuation immutable identity drifted"
        )
    expected_current = {
        key: current_identity.get(key)
        for key in (
            "config_sha256",
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
        )
    }
    expected_current["stage5_gate_sha256"] = parent_immutable[
        "stage5_gate_sha256"
    ]
    expected_current.update(_review_authorization_immutable_bindings(current_review))
    stable = (
        "config_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
        "stage5_gate_sha256",
        "formal_run_id",
    )
    if (
        current_immutable != expected_current
        or any(
            parent_immutable.get(key) != current_immutable.get(key) for key in stable
        )
        or parent_identity.get("source_set_sha256")
        == current_identity.get("source_set_sha256")
        or parent_review == current_review
    ):
        raise Stage6SourceRepairError(
            "source repair continuation is not an exact reviewed source cutover"
        )
    return current_identity, current_immutable, current_review


_SENSOR_ACCELERATION_IMMUTABLE_KEYS: Final = frozenset(
    {
        "coverage_cache_manifest_path",
        "coverage_cache_manifest_sha256",
        "coverage_cache_manifest_size_bytes",
        "coverage_cache_root",
        "coverage_cache_entry_set_sha256",
        "coverage_cache_runtime_mode",
        "coverage_cache_formal_audit_sha256",
        "coverage_cache_formal_audit_size_bytes",
    }
)


def _validated_sensor_acceleration_identity(
    *,
    stage: Path,
    parent_context: "Stage6SourceRepairContext",
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    parent_keys = set(parent_context.current_immutable_bindings)
    provided = dict(current_immutable_bindings)
    if set(provided) != parent_keys | set(_SENSOR_ACCELERATION_IMMUTABLE_KEYS):
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration immutable identity drifted"
        )
    current_identity, _base_immutable, current_review = _validated_successor_identity(
        stage=stage,
        parent_execution_identity=parent_context.current_execution_identity,
        parent_verified_review_authorization=(
            parent_context.current_verified_review_authorization
        ),
        parent_immutable_bindings=parent_context.current_immutable_bindings,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=current_verified_review_authorization,
        current_immutable_bindings={key: provided[key] for key in parent_keys},
    )
    manifest = _FIXED_COVERAGE_CACHE_MANIFEST_BINDING
    audit = _SENSOR_ACCELERATION_EVIDENCE_BINDINGS.get(
        "coverage_cache_formal_audit",
        {},
    )
    expected_cache = {
        "coverage_cache_manifest_path": manifest.get("path"),
        "coverage_cache_manifest_sha256": manifest.get("sha256"),
        "coverage_cache_manifest_size_bytes": manifest.get("size_bytes"),
        "coverage_cache_root": manifest.get("cache_root"),
        "coverage_cache_entry_set_sha256": manifest.get("entry_set_sha256"),
        "coverage_cache_runtime_mode": "persistent_exact_manifest_read_only/v1",
        "coverage_cache_formal_audit_sha256": audit.get("sha256"),
        "coverage_cache_formal_audit_size_bytes": audit.get("size_bytes"),
    }
    if any(provided.get(key) != value for key, value in expected_cache.items()):
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration cache identity drifted"
        )
    return current_identity, provided, current_review


def _require_fixed_primary_amendment(
    primary_payload: bytes,
) -> dict[str, object]:
    expected = {
        "path": SOURCE_REPAIR_AMENDMENT_NAME,
        "sha256": _FIXED_PRIMARY_AMENDMENT_BINDING.get("sha256"),
        "size_bytes": _FIXED_PRIMARY_AMENDMENT_BINDING.get("size_bytes"),
    }
    if (
        _FIXED_PRIMARY_AMENDMENT_BINDING != expected
        or not _is_sha256(expected["sha256"])
        or type(expected["size_bytes"]) is not int
        or {
            "path": SOURCE_REPAIR_AMENDMENT_NAME,
            "sha256": _sha256(primary_payload),
            "size_bytes": len(primary_payload),
        }
        != expected
    ):
        raise Stage6SourceRepairError("primary source-repair amendment drifted")
    return expected


def _require_fixed_continuation(
    continuation_payload: bytes,
) -> dict[str, object]:
    expected = {
        "path": SOURCE_REPAIR_CONTINUATION_NAME,
        "sha256": _FIXED_CONTINUATION_BINDING.get("sha256"),
        "size_bytes": _FIXED_CONTINUATION_BINDING.get("size_bytes"),
    }
    value = _strict_json(
        continuation_payload,
        label="source-repair continuation",
    )
    current = value.get("current")
    current_identity = (
        current.get("execution_identity") if isinstance(current, Mapping) else None
    )
    if (
        _FIXED_CONTINUATION_BINDING != expected
        or not _is_sha256(expected["sha256"])
        or type(expected["size_bytes"]) is not int
        or {
            "path": SOURCE_REPAIR_CONTINUATION_NAME,
            "sha256": _sha256(continuation_payload),
            "size_bytes": len(continuation_payload),
        }
        != expected
        or value.get("schema_version") != SOURCE_REPAIR_CONTINUATION_SCHEMA
        or value.get("repair_id") != SOURCE_REPAIR_CONTINUATION_ID
        or value.get("repair_ordinal") != 2
        or not isinstance(current_identity, Mapping)
        or _canonical_sha256(dict(current_identity))
        != _FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256
    ):
        raise Stage6SourceRepairError("source-repair continuation drifted")
    return expected


def _require_fixed_supplement(
    supplement_payload: bytes,
) -> dict[str, object]:
    expected = {
        "path": SOURCE_REPAIR_SUPPLEMENT_NAME,
        "sha256": _FIXED_SUPPLEMENT_BINDING.get("sha256"),
        "size_bytes": _FIXED_SUPPLEMENT_BINDING.get("size_bytes"),
    }
    value = _strict_json(
        supplement_payload,
        label="source-repair supplement",
    )
    current = value.get("current")
    current_identity = (
        current.get("execution_identity") if isinstance(current, Mapping) else None
    )
    if (
        _FIXED_SUPPLEMENT_BINDING != expected
        or not _is_sha256(expected["sha256"])
        or type(expected["size_bytes"]) is not int
        or {
            "path": SOURCE_REPAIR_SUPPLEMENT_NAME,
            "sha256": _sha256(supplement_payload),
            "size_bytes": len(supplement_payload),
        }
        != expected
        or value.get("schema_version") != SOURCE_REPAIR_SUPPLEMENT_SCHEMA
        or value.get("repair_id") != SOURCE_REPAIR_SUPPLEMENT_ID
        or value.get("repair_ordinal") != 3
        or not isinstance(current_identity, Mapping)
        or _canonical_sha256(dict(current_identity))
        != _FIXED_SUPPLEMENT_EXECUTION_IDENTITY_SHA256
    ):
        raise Stage6SourceRepairError("source-repair supplement drifted")
    return expected


def _require_fixed_closure(
    closure_payload: bytes,
) -> dict[str, object]:
    expected = {
        "path": SOURCE_REPAIR_CLOSURE_NAME,
        "sha256": _FIXED_CLOSURE_BINDING.get("sha256"),
        "size_bytes": _FIXED_CLOSURE_BINDING.get("size_bytes"),
    }
    value = _strict_json(
        closure_payload,
        label="source-repair closure",
    )
    current = value.get("current")
    current_identity = (
        current.get("execution_identity") if isinstance(current, Mapping) else None
    )
    if (
        _FIXED_CLOSURE_BINDING != expected
        or not _is_sha256(expected["sha256"])
        or type(expected["size_bytes"]) is not int
        or {
            "path": SOURCE_REPAIR_CLOSURE_NAME,
            "sha256": _sha256(closure_payload),
            "size_bytes": len(closure_payload),
        }
        != expected
        or value.get("schema_version") != SOURCE_REPAIR_CLOSURE_SCHEMA
        or value.get("repair_id") != SOURCE_REPAIR_CLOSURE_ID
        or value.get("repair_ordinal") != 4
        or not isinstance(current_identity, Mapping)
        or _canonical_sha256(dict(current_identity))
        != _FIXED_CLOSURE_EXECUTION_IDENTITY_SHA256
    ):
        raise Stage6SourceRepairError("source-repair closure drifted")
    return expected


def _require_fixed_frontier_recovery(
    frontier_recovery_payload: bytes,
) -> dict[str, object]:
    expected = {
        "path": SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
        "sha256": _FIXED_FRONTIER_RECOVERY_BINDING.get("sha256"),
        "size_bytes": _FIXED_FRONTIER_RECOVERY_BINDING.get("size_bytes"),
    }
    value = _strict_json(
        frontier_recovery_payload,
        label="source-repair frontier recovery",
    )
    current = value.get("current")
    current_identity = (
        current.get("execution_identity") if isinstance(current, Mapping) else None
    )
    if (
        _FIXED_FRONTIER_RECOVERY_BINDING != expected
        or not _is_sha256(expected["sha256"])
        or type(expected["size_bytes"]) is not int
        or {
            "path": SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
            "sha256": _sha256(frontier_recovery_payload),
            "size_bytes": len(frontier_recovery_payload),
        }
        != expected
        or value.get("schema_version") != SOURCE_REPAIR_FRONTIER_RECOVERY_SCHEMA
        or value.get("repair_id") != SOURCE_REPAIR_FRONTIER_RECOVERY_ID
        or value.get("repair_ordinal") != 5
        or not isinstance(current_identity, Mapping)
        or _canonical_sha256(dict(current_identity))
        != _FIXED_FRONTIER_RECOVERY_EXECUTION_IDENTITY_SHA256
    ):
        raise Stage6SourceRepairError("source-repair frontier recovery drifted")
    return expected


def _continuation_value(
    *,
    stage: Path,
    primary_context: "Stage6SourceRepairContext",
    primary_payload: bytes,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    allow_growth: bool,
) -> dict[str, object]:
    primary_binding = _require_fixed_primary_amendment(primary_payload)
    current_identity, current_immutable, current_review = (
        _validated_successor_identity(
            stage=stage,
            parent_execution_identity=(
                primary_context.current_execution_identity
            ),
            parent_verified_review_authorization=(
                primary_context.current_verified_review_authorization
            ),
            parent_immutable_bindings=primary_context.current_immutable_bindings,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
        )
    )
    prefix = _continuation_prefix_payloads(stage, allow_growth=allow_growth)
    _continuation_evidence_payloads(stage=stage, primary_payload=primary_payload)
    checkpoint_lineage, checkpoint_identity, prefix_semantics = (
        _validate_continuation_prefix_semantics(
            stage,
            prefix,
            origin_immutable=primary_context.origin_immutable_bindings,
            at_creation=not allow_growth,
        )
    )
    if checkpoint_lineage != dict(primary_context.origin_checkpoint_lineage):
        raise Stage6SourceRepairError(
            "continuation origin checkpoint lineage drifted"
        )
    return {
        "schema_version": SOURCE_REPAIR_CONTINUATION_SCHEMA,
        "repair_id": SOURCE_REPAIR_CONTINUATION_ID,
        "formal_run_id": FIXED_FORMAL_RUN_ID,
        "repair_ordinal": 2,
        "parent": {
            "artifact": primary_binding,
            "repair_id": SOURCE_REPAIR_ID,
            "repair_ordinal": 1,
            "execution_identity_sha256": _canonical_sha256(
                dict(primary_context.current_execution_identity)
            ),
            "immutable_bindings_sha256": _canonical_sha256(
                dict(primary_context.current_immutable_bindings)
            ),
            "verified_review_authorization_sha256": _canonical_sha256(
                dict(primary_context.current_verified_review_authorization)
            ),
            "source_set_sha256": primary_context.current_execution_identity[
                "source_set_sha256"
            ],
        },
        "cutover": {
            "seed": FIXED_SEED,
            "accepted_prefix_updates": LAST_ORIGIN_UPDATE,
            "first_repaired_update": FIRST_REPAIRED_UPDATE,
            "next_transaction_key": NEXT_TRANSACTION_KEY,
            "failed_resource_attempt": 2,
            "next_resource_attempt": 3,
        },
        "current": {
            "execution_identity": current_identity,
            "execution_identity_sha256": _canonical_sha256(current_identity),
            "immutable_bindings": current_immutable,
            "immutable_bindings_sha256": _canonical_sha256(current_immutable),
            "verified_review_authorization": current_review,
            "verified_review_authorization_sha256": _canonical_sha256(
                current_review
            ),
        },
        "checkpoint_identity": checkpoint_identity,
        "prefix_semantics": prefix_semantics,
        "prefix_artifacts": [
            dict(_CONTINUATION_PREFIX_BINDINGS[path])
            for path in sorted(_CONTINUATION_PREFIX_BINDINGS)
        ],
        "failure_evidence": [
            dict(_CONTINUATION_EVIDENCE_BINDINGS[kind])
            for kind in sorted(_CONTINUATION_EVIDENCE_BINDINGS)
        ],
    }


def _supplement_value(
    *,
    stage: Path,
    primary_context: "Stage6SourceRepairContext",
    continuation_context: "Stage6SourceRepairContext",
    primary_payload: bytes,
    continuation_payload: bytes,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    allow_growth: bool,
) -> dict[str, object]:
    primary_binding = _require_fixed_primary_amendment(primary_payload)
    continuation_binding = _require_fixed_continuation(continuation_payload)
    if (
        continuation_context.continuation_sha256 != continuation_binding["sha256"]
        or continuation_context.continuation_size_bytes
        != continuation_binding["size_bytes"]
    ):
        raise Stage6SourceRepairError("supplement parent continuation drifted")
    current_identity, current_immutable, current_review = (
        _validated_successor_identity(
            stage=stage,
            parent_execution_identity=(
                continuation_context.current_execution_identity
            ),
            parent_verified_review_authorization=(
                continuation_context.current_verified_review_authorization
            ),
            parent_immutable_bindings=(
                continuation_context.current_immutable_bindings
            ),
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
        )
    )
    prefix = _supplement_prefix_payloads(stage, allow_growth=allow_growth)
    _supplement_evidence_payloads()
    checkpoint_identity, prefix_semantics = _validate_supplement_prefix_semantics(
        stage,
        prefix,
        parent_context=continuation_context,
        at_creation=not allow_growth,
    )
    return {
        "schema_version": SOURCE_REPAIR_SUPPLEMENT_SCHEMA,
        "repair_id": SOURCE_REPAIR_SUPPLEMENT_ID,
        "formal_run_id": FIXED_FORMAL_RUN_ID,
        "repair_ordinal": 3,
        "repair_chain": [
            {
                "artifact": primary_binding,
                "repair_id": SOURCE_REPAIR_ID,
                "repair_ordinal": 1,
            },
            {
                "artifact": continuation_binding,
                "repair_id": SOURCE_REPAIR_CONTINUATION_ID,
                "repair_ordinal": 2,
                "execution_identity_sha256": _canonical_sha256(
                    dict(continuation_context.current_execution_identity)
                ),
                "immutable_bindings_sha256": _canonical_sha256(
                    dict(continuation_context.current_immutable_bindings)
                ),
                "verified_review_authorization_sha256": _canonical_sha256(
                    dict(
                        continuation_context.current_verified_review_authorization
                    )
                ),
            },
        ],
        "cutover": {
            "seed": FIXED_SEED,
            "accepted_prefix_updates": LAST_CONTINUATION_UPDATE,
            "last_parent_update": LAST_CONTINUATION_UPDATE,
            "first_repaired_update": FIRST_SUPPLEMENT_UPDATE,
            "next_transaction_key": SUPPLEMENT_NEXT_TRANSACTION_KEY,
            "failed_resource_attempt": 1,
            "next_resource_attempt": 2,
        },
        "current": {
            "execution_identity": current_identity,
            "execution_identity_sha256": _canonical_sha256(current_identity),
            "immutable_bindings": current_immutable,
            "immutable_bindings_sha256": _canonical_sha256(current_immutable),
            "verified_review_authorization": current_review,
            "verified_review_authorization_sha256": _canonical_sha256(
                current_review
            ),
        },
        "checkpoint_identity": checkpoint_identity,
        "prefix_semantics": prefix_semantics,
        "prefix_artifacts": [
            dict(_SUPPLEMENT_PREFIX_BINDINGS[path])
            for path in sorted(_SUPPLEMENT_PREFIX_BINDINGS)
        ],
        "failure_evidence": [
            dict(_SUPPLEMENT_EVIDENCE_BINDINGS[kind])
            for kind in sorted(_SUPPLEMENT_EVIDENCE_BINDINGS)
        ],
    }


def _closure_value(
    *,
    stage: Path,
    primary_context: "Stage6SourceRepairContext",
    continuation_context: "Stage6SourceRepairContext",
    supplement_context: "Stage6SourceRepairContext",
    primary_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    allow_growth: bool,
) -> dict[str, object]:
    primary_binding = _require_fixed_primary_amendment(primary_payload)
    continuation_binding = _require_fixed_continuation(continuation_payload)
    supplement_binding = _require_fixed_supplement(supplement_payload)
    if (
        supplement_context.supplement_sha256 != supplement_binding["sha256"]
        or supplement_context.supplement_size_bytes
        != supplement_binding["size_bytes"]
    ):
        raise Stage6SourceRepairError("closure parent supplement drifted")
    current_identity, current_immutable, current_review = (
        _validated_successor_identity(
            stage=stage,
            parent_execution_identity=supplement_context.current_execution_identity,
            parent_verified_review_authorization=(
                supplement_context.current_verified_review_authorization
            ),
            parent_immutable_bindings=supplement_context.current_immutable_bindings,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
        )
    )
    prefix = _closure_prefix_payloads(stage, allow_growth=allow_growth)
    _closure_evidence_payloads(supplement_payload=supplement_payload)
    checkpoint_identity, prefix_semantics = _validate_closure_prefix_semantics(
        stage,
        prefix,
        continuation_context=continuation_context,
        supplement_context=supplement_context,
        at_creation=not allow_growth,
    )
    return {
        "schema_version": SOURCE_REPAIR_CLOSURE_SCHEMA,
        "repair_id": SOURCE_REPAIR_CLOSURE_ID,
        "formal_run_id": FIXED_FORMAL_RUN_ID,
        "repair_ordinal": 4,
        "repair_chain": [
            {
                "artifact": primary_binding,
                "repair_id": SOURCE_REPAIR_ID,
                "repair_ordinal": 1,
            },
            {
                "artifact": continuation_binding,
                "repair_id": SOURCE_REPAIR_CONTINUATION_ID,
                "repair_ordinal": 2,
                "execution_identity_sha256": _canonical_sha256(
                    dict(continuation_context.current_execution_identity)
                ),
            },
            {
                "artifact": supplement_binding,
                "repair_id": SOURCE_REPAIR_SUPPLEMENT_ID,
                "repair_ordinal": 3,
                "execution_identity_sha256": _canonical_sha256(
                    dict(supplement_context.current_execution_identity)
                ),
                "immutable_bindings_sha256": _canonical_sha256(
                    dict(supplement_context.current_immutable_bindings)
                ),
                "verified_review_authorization_sha256": _canonical_sha256(
                    dict(supplement_context.current_verified_review_authorization)
                ),
            },
        ],
        "cutover": {
            "seed": FIXED_SEED,
            "accepted_prefix_updates": LAST_CONTINUATION_UPDATE,
            "last_parent_update": LAST_CONTINUATION_UPDATE,
            "first_repaired_update": FIRST_CLOSURE_UPDATE,
            "next_transaction_key": CLOSURE_NEXT_TRANSACTION_KEY,
            "failed_resource_attempt": 2,
            "next_resource_attempt": 3,
            "next_segment_index": 5,
        },
        "current": {
            "execution_identity": current_identity,
            "execution_identity_sha256": _canonical_sha256(current_identity),
            "immutable_bindings": current_immutable,
            "immutable_bindings_sha256": _canonical_sha256(current_immutable),
            "verified_review_authorization": current_review,
            "verified_review_authorization_sha256": _canonical_sha256(current_review),
        },
        "checkpoint_identity": checkpoint_identity,
        "prefix_semantics": prefix_semantics,
        "prefix_artifacts": [
            dict(_CLOSURE_PREFIX_BINDINGS[path])
            for path in sorted(_CLOSURE_PREFIX_BINDINGS)
        ],
        "review_evidence": [
            dict(_CLOSURE_EVIDENCE_BINDINGS[kind])
            for kind in sorted(_CLOSURE_EVIDENCE_BINDINGS)
        ],
    }


def _frontier_recovery_value(
    *,
    stage: Path,
    primary_context: "Stage6SourceRepairContext",
    continuation_context: "Stage6SourceRepairContext",
    supplement_context: "Stage6SourceRepairContext",
    closure_context: "Stage6SourceRepairContext",
    primary_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    closure_payload: bytes,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    allow_growth: bool,
    validate_current_sources: bool,
) -> dict[str, object]:
    primary_binding = _require_fixed_primary_amendment(primary_payload)
    continuation_binding = _require_fixed_continuation(continuation_payload)
    supplement_binding = _require_fixed_supplement(supplement_payload)
    closure_binding = _require_fixed_closure(closure_payload)
    if (
        closure_context.closure_sha256 != closure_binding["sha256"]
        or closure_context.closure_size_bytes != closure_binding["size_bytes"]
    ):
        raise Stage6SourceRepairError("frontier recovery parent closure drifted")
    current_identity, current_immutable, current_review = (
        _validated_successor_identity(
            stage=stage,
            parent_execution_identity=closure_context.current_execution_identity,
            parent_verified_review_authorization=(
                closure_context.current_verified_review_authorization
            ),
            parent_immutable_bindings=closure_context.current_immutable_bindings,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
        )
    )
    prefix = _frontier_recovery_prefix_payloads(
        stage,
        allow_growth=allow_growth,
    )
    _frontier_recovery_evidence_payloads()
    if validate_current_sources:
        _frontier_recovery_source_payloads()
    checkpoint_identity, prefix_semantics = (
        _validate_frontier_recovery_prefix_semantics(
            stage,
            prefix,
            parent_context=closure_context,
            at_creation=not allow_growth,
        )
    )
    return {
        "schema_version": SOURCE_REPAIR_FRONTIER_RECOVERY_SCHEMA,
        "repair_id": SOURCE_REPAIR_FRONTIER_RECOVERY_ID,
        "formal_run_id": FIXED_FORMAL_RUN_ID,
        "repair_ordinal": 5,
        "repair_chain": [
            {
                "artifact": primary_binding,
                "repair_id": SOURCE_REPAIR_ID,
                "repair_ordinal": 1,
            },
            {
                "artifact": continuation_binding,
                "repair_id": SOURCE_REPAIR_CONTINUATION_ID,
                "repair_ordinal": 2,
                "execution_identity_sha256": _canonical_sha256(
                    dict(continuation_context.current_execution_identity)
                ),
            },
            {
                "artifact": supplement_binding,
                "repair_id": SOURCE_REPAIR_SUPPLEMENT_ID,
                "repair_ordinal": 3,
                "execution_identity_sha256": _canonical_sha256(
                    dict(supplement_context.current_execution_identity)
                ),
            },
            {
                "artifact": closure_binding,
                "repair_id": SOURCE_REPAIR_CLOSURE_ID,
                "repair_ordinal": 4,
                "execution_identity_sha256": _canonical_sha256(
                    dict(closure_context.current_execution_identity)
                ),
                "immutable_bindings_sha256": _canonical_sha256(
                    dict(closure_context.current_immutable_bindings)
                ),
                "verified_review_authorization_sha256": _canonical_sha256(
                    dict(closure_context.current_verified_review_authorization)
                ),
            },
        ],
        "cutover": {
            "seed": FIXED_SEED,
            "accepted_prefix_updates": LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
            "last_parent_update": LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
            "first_repaired_update": FIRST_FRONTIER_RECOVERY_UPDATE,
            "next_transaction_key": FRONTIER_RECOVERY_NEXT_TRANSACTION_KEY,
            "failed_resource_attempt": 1,
            "next_resource_attempt": 2,
            "next_segment_index": 6,
        },
        "current": {
            "execution_identity": current_identity,
            "execution_identity_sha256": _canonical_sha256(current_identity),
            "immutable_bindings": current_immutable,
            "immutable_bindings_sha256": _canonical_sha256(current_immutable),
            "verified_review_authorization": current_review,
            "verified_review_authorization_sha256": _canonical_sha256(
                current_review
            ),
        },
        "checkpoint_identity": checkpoint_identity,
        "prefix_semantics": prefix_semantics,
        "prefix_artifacts": [
            dict(_FRONTIER_RECOVERY_PREFIX_BINDINGS[path])
            for path in sorted(_FRONTIER_RECOVERY_PREFIX_BINDINGS)
        ],
        "repair_evidence": [
            dict(_FRONTIER_RECOVERY_EVIDENCE_BINDINGS[kind])
            for kind in sorted(_FRONTIER_RECOVERY_EVIDENCE_BINDINGS)
        ],
        "reviewed_sources": [
            dict(_FRONTIER_RECOVERY_SOURCE_BINDINGS[path])
            for path in sorted(_FRONTIER_RECOVERY_SOURCE_BINDINGS)
        ],
    }


def _sensor_acceleration_value(
    *,
    stage: Path,
    parent_context: "Stage6SourceRepairContext",
    primary_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    closure_payload: bytes,
    frontier_recovery_payload: bytes,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    allow_growth: bool,
) -> dict[str, object]:
    _require_fixed_primary_amendment(primary_payload)
    _require_fixed_continuation(continuation_payload)
    _require_fixed_supplement(supplement_payload)
    _require_fixed_closure(closure_payload)
    frontier_binding = _require_fixed_frontier_recovery(frontier_recovery_payload)
    if (
        parent_context.frontier_recovery_sha256 != frontier_binding["sha256"]
        or parent_context.frontier_recovery_size_bytes != frontier_binding["size_bytes"]
    ):
        raise Stage6SourceRepairError(
            "sensor acceleration parent frontier recovery drifted"
        )
    current_identity, current_immutable, current_review = (
        _validated_sensor_acceleration_identity(
            stage=stage,
            parent_context=parent_context,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
        )
    )
    prefix = _sensor_acceleration_prefix_payloads(
        stage,
        allow_growth=allow_growth,
    )
    evidence_payloads = _sensor_acceleration_evidence_payloads()
    _sensor_acceleration_source_payloads()
    checkpoint_identity, prefix_semantics = (
        _validate_sensor_acceleration_prefix_semantics(
            stage,
            prefix,
            parent_context=parent_context,
            at_creation=not allow_growth,
        )
    )
    frontier_value = _strict_json(
        frontier_recovery_payload,
        label="source-repair frontier recovery",
    )
    parent_chain = frontier_value.get("repair_chain")
    if (
        not isinstance(parent_chain, list)
        or len(parent_chain) != 4
        or tuple(row.get("repair_ordinal") for row in parent_chain if isinstance(row, Mapping))
        != (1, 2, 3, 4)
    ):
        raise Stage6SourceRepairError("frontier recovery repair chain drifted")
    manifest = dict(_FIXED_COVERAGE_CACHE_MANIFEST_BINDING)
    manifest["runtime_mode"] = "persistent_exact_manifest_read_only/v1"
    audit_binding = dict(
        _SENSOR_ACCELERATION_EVIDENCE_BINDINGS["coverage_cache_formal_audit"]
    )
    return {
        "schema_version": SOURCE_REPAIR_SENSOR_ACCELERATION_SCHEMA,
        "repair_id": SOURCE_REPAIR_SENSOR_ACCELERATION_ID,
        "formal_run_id": FIXED_FORMAL_RUN_ID,
        "repair_ordinal": 6,
        "repair_chain": [
            *[dict(row) for row in parent_chain],
            {
                "artifact": frontier_binding,
                "repair_id": SOURCE_REPAIR_FRONTIER_RECOVERY_ID,
                "repair_ordinal": 5,
                "execution_identity_sha256": _canonical_sha256(
                    dict(parent_context.current_execution_identity)
                ),
                "immutable_bindings_sha256": _canonical_sha256(
                    dict(parent_context.current_immutable_bindings)
                ),
                "verified_review_authorization_sha256": _canonical_sha256(
                    dict(parent_context.current_verified_review_authorization)
                ),
            },
        ],
        "cutover": {
            "seed": FIXED_SEED,
            "accepted_prefix_updates": LAST_SENSOR_ACCELERATION_PARENT_UPDATE,
            "last_parent_update": LAST_SENSOR_ACCELERATION_PARENT_UPDATE,
            "first_repaired_update": FIRST_SENSOR_ACCELERATION_UPDATE,
            "next_transaction_key": SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY,
            "failed_resource_attempt": SENSOR_ACCELERATION_FAILED_RESOURCE_ATTEMPT,
            "next_resource_attempt": SENSOR_ACCELERATION_NEXT_RESOURCE_ATTEMPT,
            "next_segment_index": SENSOR_ACCELERATION_NEXT_SEGMENT_INDEX,
        },
        "current": {
            "execution_identity": current_identity,
            "execution_identity_sha256": _canonical_sha256(current_identity),
            "immutable_bindings": current_immutable,
            "immutable_bindings_sha256": _canonical_sha256(current_immutable),
            "verified_review_authorization": current_review,
            "verified_review_authorization_sha256": _canonical_sha256(
                current_review
            ),
        },
        "checkpoint_identity": checkpoint_identity,
        "prefix_semantics": prefix_semantics,
        "prefix_artifacts": [
            dict(_SENSOR_ACCELERATION_PREFIX_BINDINGS[path])
            for path in sorted(_SENSOR_ACCELERATION_PREFIX_BINDINGS)
        ],
        "production_cache": {
            "manifest": manifest,
            "formal_audit": audit_binding,
        },
        "repair_evidence": [
            dict(_SENSOR_ACCELERATION_EVIDENCE_BINDINGS[kind])
            for kind in sorted(_SENSOR_ACCELERATION_EVIDENCE_BINDINGS)
        ],
        "reviewed_sources": [
            dict(_SENSOR_ACCELERATION_SOURCE_BINDINGS[path])
            for path in sorted(_SENSOR_ACCELERATION_SOURCE_BINDINGS)
        ],
        "evidence_payload_sha256": _canonical_sha256(
            {
                kind: _sha256(payload)
                for kind, payload in sorted(evidence_payloads.items())
            }
        ),
    }


def _amendment_value(
    *,
    stage: Path,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    allow_growth: bool,
) -> dict[str, object]:
    prefix = _fixed_prefix_payloads(stage, allow_growth=allow_growth)
    _fixed_evidence_payloads()
    (
        origin_identity,
        origin_immutable,
        origin_review,
        current_identity,
        current_immutable,
    ) = _validated_identities(
        stage=stage,
        prefix=prefix,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=current_verified_review_authorization,
        current_immutable_bindings=current_immutable_bindings,
    )
    receipts, prefix_semantics = _validate_prefix_semantics(
        prefix, origin_immutable=origin_immutable
    )
    checkpoint_lineage, checkpoint_identity = _validate_checkpoint_prefix(
        prefix,
        config_sha256=str(origin_immutable["config_sha256"]),
        receipts=receipts,
    )
    if any(
        checkpoint_lineage.get(key) != origin_immutable.get(key)
        for key in (
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
            "formal_run_id",
            "changed_path_set_sha256",
            "review_authorization_record_sha256",
            "authorization_file_sha256",
            "review_identity_sha256",
            "reviewed_prospective_git_tree",
            "frozen_diff_sha256",
            "spec_review_sha256",
            "quality_review_sha256",
        )
    ):
        raise Stage6SourceRepairError("origin checkpoint/lineage binding drifted")
    return {
        "schema_version": SOURCE_REPAIR_SCHEMA,
        "repair_id": SOURCE_REPAIR_ID,
        "formal_run_id": FIXED_FORMAL_RUN_ID,
        "repair_ordinal": 1,
        "cutover": {
            "seed": FIXED_SEED,
            "accepted_prefix_updates": LAST_ORIGIN_UPDATE,
            "last_origin_update": LAST_ORIGIN_UPDATE,
            "first_repaired_update": FIRST_REPAIRED_UPDATE,
            "next_transaction_key": NEXT_TRANSACTION_KEY,
        },
        "origin": {
            "lineage_audit_sha256": _sha256(prefix["lineage_audit.json"]),
            "lineage_audit_size_bytes": len(prefix["lineage_audit.json"]),
            "execution_identity_sha256": _canonical_sha256(origin_identity),
            "immutable_bindings_sha256": _canonical_sha256(origin_immutable),
            "verified_review_authorization_sha256": _canonical_sha256(origin_review),
            "checkpoint_lineage_sha256": _canonical_sha256(checkpoint_lineage),
        },
        "current": {
            "execution_identity": current_identity,
            "execution_identity_sha256": _canonical_sha256(current_identity),
            "immutable_bindings": current_immutable,
            "immutable_bindings_sha256": _canonical_sha256(current_immutable),
            "verified_review_authorization": dict(
                current_verified_review_authorization
            ),
            "verified_review_authorization_sha256": _canonical_sha256(
                dict(current_verified_review_authorization)
            ),
        },
        "checkpoint_identity": checkpoint_identity,
        "prefix_semantics": prefix_semantics,
        "prefix_artifacts": [
            dict(_FIXED_PREFIX_BINDINGS[path]) for path in sorted(_FIXED_PREFIX_BINDINGS)
        ],
        "repair_evidence": [
            dict(_FIXED_EVIDENCE_BINDINGS[kind])
            for kind in sorted(_FIXED_EVIDENCE_BINDINGS)
        ],
    }


@dataclass(frozen=True, slots=True)
class Stage6SourceRepairContext:
    stage_root: Path
    amendment_sha256: str
    amendment_size_bytes: int
    continuation_sha256: str | None
    continuation_size_bytes: int | None
    supplement_sha256: str | None
    supplement_size_bytes: int | None
    closure_sha256: str | None
    closure_size_bytes: int | None
    frontier_recovery_sha256: str | None
    frontier_recovery_size_bytes: int | None
    origin_execution_identity: Mapping[str, object]
    origin_immutable_bindings: Mapping[str, object]
    origin_verified_review_authorization: Mapping[str, object]
    parent_execution_identity: Mapping[str, object]
    parent_immutable_bindings: Mapping[str, object]
    parent_verified_review_authorization: Mapping[str, object]
    continuation_execution_identity: Mapping[str, object] | None
    continuation_immutable_bindings: Mapping[str, object] | None
    continuation_verified_review_authorization: Mapping[str, object] | None
    supplement_execution_identity: Mapping[str, object] | None
    supplement_immutable_bindings: Mapping[str, object] | None
    supplement_verified_review_authorization: Mapping[str, object] | None
    closure_execution_identity: Mapping[str, object] | None
    closure_immutable_bindings: Mapping[str, object] | None
    closure_verified_review_authorization: Mapping[str, object] | None
    current_execution_identity: Mapping[str, object]
    current_immutable_bindings: Mapping[str, object]
    current_verified_review_authorization: Mapping[str, object]
    origin_checkpoint_lineage: Mapping[str, object]
    sensor_acceleration_sha256: str | None = None
    sensor_acceleration_size_bytes: int | None = None
    frontier_recovery_execution_identity: Mapping[str, object] | None = None
    frontier_recovery_immutable_bindings: Mapping[str, object] | None = None
    frontier_recovery_verified_review_authorization: (
        Mapping[str, object] | None
    ) = None
    coverage_cache_manifest_path: str | None = None
    coverage_cache_manifest_sha256: str | None = None
    coverage_cache_manifest_size_bytes: int | None = None
    coverage_cache_entry_set_sha256: str | None = None
    coverage_cache_root: str | None = None
    coverage_cache_entry_count: int | None = None
    coverage_cache_split_counts: Mapping[str, int] | None = None
    coverage_cache_runtime_mode: str | None = None
    coverage_cache_formal_audit_sha256: str | None = None
    coverage_cache_formal_audit_size_bytes: int | None = None

    def checkpoint_lineage_for_update(self, update: int) -> dict[str, object]:
        if type(update) is not int or update <= 0:
            raise Stage6SourceRepairError("checkpoint update must be positive")
        if update <= LAST_ORIGIN_UPDATE:
            return dict(self.origin_checkpoint_lineage)
        if self.closure_sha256 is not None and self.supplement_sha256 is None:
            raise Stage6SourceRepairError("closure supplement lineage is missing")
        if self.frontier_recovery_sha256 is not None and self.closure_sha256 is None:
            raise Stage6SourceRepairError(
                "frontier recovery closure lineage is missing"
            )
        if (
            self.sensor_acceleration_sha256 is not None
            and self.frontier_recovery_sha256 is None
        ):
            raise Stage6SourceRepairError(
                "sensor acceleration frontier recovery lineage is missing"
            )
        immutable = self.current_immutable_bindings
        include_supplement = (
            self.supplement_sha256 is not None and update >= FIRST_SUPPLEMENT_UPDATE
        )
        include_closure = (
            self.closure_sha256 is not None and update >= FIRST_CLOSURE_UPDATE
        )
        include_frontier_recovery = (
            self.frontier_recovery_sha256 is not None
            and update >= FIRST_FRONTIER_RECOVERY_UPDATE
        )
        include_sensor_acceleration = (
            self.sensor_acceleration_sha256 is not None
            and update >= FIRST_SENSOR_ACCELERATION_UPDATE
        )
        if (
            self.sensor_acceleration_sha256 is not None
            and update <= LAST_SENSOR_ACCELERATION_PARENT_UPDATE
        ):
            if self.frontier_recovery_immutable_bindings is None:
                raise Stage6SourceRepairError(
                    "sensor acceleration frontier recovery lineage is missing"
                )
            immutable = self.frontier_recovery_immutable_bindings
            include_sensor_acceleration = False
        if (
            self.frontier_recovery_sha256 is not None
            and update <= LAST_FRONTIER_RECOVERY_PARENT_UPDATE
        ):
            if self.closure_immutable_bindings is None:
                raise Stage6SourceRepairError(
                    "frontier recovery closure lineage is missing"
                )
            immutable = self.closure_immutable_bindings
            include_frontier_recovery = False
        if self.supplement_sha256 is not None and update <= LAST_CONTINUATION_UPDATE:
            if self.continuation_immutable_bindings is None:
                raise Stage6SourceRepairError(
                    "supplement continuation lineage is missing"
                )
            immutable = self.continuation_immutable_bindings
            include_supplement = False
            include_closure = False
            include_frontier_recovery = False
            include_sensor_acceleration = False
        repaired = dict(self.origin_checkpoint_lineage)
        for key in (
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
            "formal_run_id",
            "changed_path_set_sha256",
            "review_authorization_record_sha256",
            "authorization_file_sha256",
            "review_identity_sha256",
            "reviewed_prospective_git_tree",
            "frozen_diff_sha256",
            "spec_review_sha256",
            "quality_review_sha256",
        ):
            repaired[key] = immutable[key]
        repaired.update(
            {
                "source_repair_schema_version": SOURCE_REPAIR_SCHEMA,
                "source_repair_amendment_sha256": self.amendment_sha256,
                "source_repair_seed": FIXED_SEED,
                "source_repair_last_origin_update": LAST_ORIGIN_UPDATE,
                "source_repair_first_repaired_update": FIRST_REPAIRED_UPDATE,
            }
        )
        if self.continuation_sha256 is not None:
            repaired.update(
                {
                    "source_repair_amendment_ordinal": 1,
                    "source_repair_continuation_schema_version": (
                        SOURCE_REPAIR_CONTINUATION_SCHEMA
                    ),
                    "source_repair_continuation_sha256": (
                        self.continuation_sha256
                    ),
                    "source_repair_continuation_ordinal": 2,
                }
            )
        if include_supplement:
            repaired.update(
                {
                    "source_repair_supplement_schema_version": (
                        SOURCE_REPAIR_SUPPLEMENT_SCHEMA
                    ),
                    "source_repair_supplement_sha256": self.supplement_sha256,
                    "source_repair_supplement_ordinal": 3,
                    "source_repair_supplement_last_parent_update": (
                        LAST_CONTINUATION_UPDATE
                    ),
                    "source_repair_supplement_first_repaired_update": (
                        FIRST_SUPPLEMENT_UPDATE
                    ),
                }
            )
        if include_closure:
            repaired.update(
                {
                    "source_repair_closure_schema_version": (
                        SOURCE_REPAIR_CLOSURE_SCHEMA
                    ),
                    "source_repair_closure_sha256": self.closure_sha256,
                    "source_repair_closure_ordinal": 4,
                    "source_repair_closure_last_parent_update": (
                        LAST_CONTINUATION_UPDATE
                    ),
                    "source_repair_closure_first_repaired_update": (
                        FIRST_CLOSURE_UPDATE
                    ),
                }
            )
        if include_frontier_recovery:
            repaired.update(
                {
                    "source_repair_frontier_recovery_schema_version": (
                        SOURCE_REPAIR_FRONTIER_RECOVERY_SCHEMA
                    ),
                    "source_repair_frontier_recovery_sha256": (
                        self.frontier_recovery_sha256
                    ),
                    "source_repair_frontier_recovery_ordinal": 5,
                    "source_repair_frontier_recovery_last_parent_update": (
                        LAST_FRONTIER_RECOVERY_PARENT_UPDATE
                    ),
                    "source_repair_frontier_recovery_first_repaired_update": (
                        FIRST_FRONTIER_RECOVERY_UPDATE
                    ),
                }
            )
        if include_sensor_acceleration:
            cache_values = {
                key: immutable[key]
                for key in _SENSOR_ACCELERATION_IMMUTABLE_KEYS
            }
            repaired.update(
                {
                    "source_repair_sensor_acceleration_schema_version": (
                        SOURCE_REPAIR_SENSOR_ACCELERATION_SCHEMA
                    ),
                    "source_repair_sensor_acceleration_sha256": (
                        self.sensor_acceleration_sha256
                    ),
                    "source_repair_sensor_acceleration_ordinal": 6,
                    "source_repair_sensor_acceleration_last_parent_update": (
                        LAST_SENSOR_ACCELERATION_PARENT_UPDATE
                    ),
                    "source_repair_sensor_acceleration_first_repaired_update": (
                        FIRST_SENSOR_ACCELERATION_UPDATE
                    ),
                    **cache_values,
                }
            )
        return repaired

    @property
    def protected_checkpoint_updates(self) -> tuple[int, ...]:
        if self.sensor_acceleration_sha256 is not None:
            return (
                LAST_ORIGIN_UPDATE,
                LAST_CONTINUATION_UPDATE,
                LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
                LAST_SENSOR_ACCELERATION_PARENT_UPDATE,
            )
        if self.frontier_recovery_sha256 is not None:
            return (
                LAST_ORIGIN_UPDATE,
                LAST_CONTINUATION_UPDATE,
                LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
            )
        if self.supplement_sha256 is not None:
            return (LAST_ORIGIN_UPDATE, LAST_CONTINUATION_UPDATE)
        return (LAST_ORIGIN_UPDATE,)

    @property
    def summary_binding(self) -> dict[str, object]:
        primary = {
            "schema_version": "stage6_source_repair_summary/v1",
            "repair_id": SOURCE_REPAIR_ID,
            "amendment_sha256": self.amendment_sha256,
            "seed": FIXED_SEED,
            "last_origin_update": LAST_ORIGIN_UPDATE,
            "first_repaired_update": FIRST_REPAIRED_UPDATE,
            "origin_execution_identity_sha256": _canonical_sha256(
                dict(self.origin_execution_identity)
            ),
            "current_execution_identity_sha256": _canonical_sha256(
                dict(self.current_execution_identity)
            ),
            "origin_source_set_sha256": self.origin_execution_identity[
                "source_set_sha256"
            ],
            "current_source_set_sha256": self.current_execution_identity[
                "source_set_sha256"
            ],
        }
        if self.continuation_sha256 is None:
            return primary
        continued = {
            **primary,
            "schema_version": "stage6_source_repair_summary/v2",
            "continuation_id": SOURCE_REPAIR_CONTINUATION_ID,
            "continuation_sha256": self.continuation_sha256,
            "repair_ordinal": 2,
            "parent_execution_identity_sha256": _canonical_sha256(
                dict(self.parent_execution_identity)
            ),
            "parent_source_set_sha256": self.parent_execution_identity[
                "source_set_sha256"
            ],
        }
        if self.supplement_sha256 is None:
            return continued
        if self.continuation_execution_identity is None:
            raise Stage6SourceRepairError(
                "supplement continuation identity is missing"
            )
        supplemented = {
            **continued,
            "schema_version": "stage6_source_repair_summary/v3",
            "supplement_id": SOURCE_REPAIR_SUPPLEMENT_ID,
            "supplement_sha256": self.supplement_sha256,
            "repair_ordinal": 3,
            "last_continuation_update": LAST_CONTINUATION_UPDATE,
            "first_supplement_update": FIRST_SUPPLEMENT_UPDATE,
            "continuation_execution_identity_sha256": _canonical_sha256(
                dict(self.continuation_execution_identity)
            ),
            "continuation_source_set_sha256": self.continuation_execution_identity[
                "source_set_sha256"
            ],
            "ordinal_artifacts": [
                {
                    "repair_ordinal": 1,
                    "path": SOURCE_REPAIR_AMENDMENT_NAME,
                    "sha256": self.amendment_sha256,
                },
                {
                    "repair_ordinal": 2,
                    "path": SOURCE_REPAIR_CONTINUATION_NAME,
                    "sha256": self.continuation_sha256,
                },
                {
                    "repair_ordinal": 3,
                    "path": SOURCE_REPAIR_SUPPLEMENT_NAME,
                    "sha256": self.supplement_sha256,
                },
            ],
        }
        if self.closure_sha256 is None:
            return supplemented
        if self.supplement_execution_identity is None:
            raise Stage6SourceRepairError("closure supplement identity is missing")
        closed = {
            **supplemented,
            "schema_version": "stage6_source_repair_summary/v4",
            "closure_id": SOURCE_REPAIR_CLOSURE_ID,
            "closure_sha256": self.closure_sha256,
            "repair_ordinal": 4,
            "first_closure_update": FIRST_CLOSURE_UPDATE,
            "supplement_execution_identity_sha256": _canonical_sha256(
                dict(self.supplement_execution_identity)
            ),
            "supplement_source_set_sha256": self.supplement_execution_identity[
                "source_set_sha256"
            ],
            "ordinal_artifacts": [
                *supplemented["ordinal_artifacts"],  # type: ignore[misc]
                {
                    "repair_ordinal": 4,
                    "path": SOURCE_REPAIR_CLOSURE_NAME,
                    "sha256": self.closure_sha256,
                },
            ],
        }
        if self.frontier_recovery_sha256 is None:
            return closed
        if self.closure_execution_identity is None:
            raise Stage6SourceRepairError(
                "frontier recovery closure identity is missing"
            )
        frontier_recovered = {
            **closed,
            "schema_version": "stage6_source_repair_summary/v5",
            "frontier_recovery_id": SOURCE_REPAIR_FRONTIER_RECOVERY_ID,
            "frontier_recovery_sha256": self.frontier_recovery_sha256,
            "repair_ordinal": 5,
            "last_closure_update": LAST_FRONTIER_RECOVERY_PARENT_UPDATE,
            "first_frontier_recovery_update": FIRST_FRONTIER_RECOVERY_UPDATE,
            "closure_execution_identity_sha256": _canonical_sha256(
                dict(self.closure_execution_identity)
            ),
            "closure_source_set_sha256": self.closure_execution_identity[
                "source_set_sha256"
            ],
            "ordinal_artifacts": [
                *closed["ordinal_artifacts"],  # type: ignore[misc]
                {
                    "repair_ordinal": 5,
                    "path": SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
                    "sha256": self.frontier_recovery_sha256,
                },
            ],
        }
        if self.sensor_acceleration_sha256 is None:
            return frontier_recovered
        if self.frontier_recovery_execution_identity is None:
            raise Stage6SourceRepairError(
                "sensor acceleration frontier recovery identity is missing"
            )
        if (
            self.coverage_cache_manifest_path is None
            or self.coverage_cache_manifest_sha256 is None
            or self.coverage_cache_manifest_size_bytes is None
            or self.coverage_cache_entry_set_sha256 is None
            or self.coverage_cache_root is None
            or self.coverage_cache_entry_count is None
            or self.coverage_cache_split_counts is None
            or self.coverage_cache_runtime_mode is None
        ):
            raise Stage6SourceRepairError(
                "sensor acceleration coverage cache identity is missing"
            )
        return {
            **frontier_recovered,
            "schema_version": SOURCE_REPAIR_SENSOR_ACCELERATION_SUMMARY_SCHEMA,
            "sensor_acceleration_id": SOURCE_REPAIR_SENSOR_ACCELERATION_ID,
            "sensor_acceleration_sha256": self.sensor_acceleration_sha256,
            "repair_ordinal": 6,
            "last_frontier_recovery_update": (
                LAST_SENSOR_ACCELERATION_PARENT_UPDATE
            ),
            "first_sensor_acceleration_update": FIRST_SENSOR_ACCELERATION_UPDATE,
            "frontier_recovery_execution_identity_sha256": _canonical_sha256(
                dict(self.frontier_recovery_execution_identity)
            ),
            "frontier_recovery_source_set_sha256": (
                self.frontier_recovery_execution_identity["source_set_sha256"]
            ),
            "coverage_cache_manifest": {
                "path": self.coverage_cache_manifest_path,
                "sha256": self.coverage_cache_manifest_sha256,
                "size_bytes": self.coverage_cache_manifest_size_bytes,
                "cache_root": self.coverage_cache_root,
                "entry_set_sha256": self.coverage_cache_entry_set_sha256,
                "entry_count": self.coverage_cache_entry_count,
                "split_counts": dict(self.coverage_cache_split_counts),
                "runtime_mode": self.coverage_cache_runtime_mode,
            },
            "ordinal_artifacts": [
                *frontier_recovered["ordinal_artifacts"],  # type: ignore[misc]
                {
                    "repair_ordinal": 6,
                    "path": SOURCE_REPAIR_SENSOR_ACCELERATION_NAME,
                    "sha256": self.sensor_acceleration_sha256,
                },
            ],
        }

    def require_current(self, label: str) -> None:
        if not isinstance(label, str) or not label.strip():
            raise Stage6SourceRepairError("source-repair revalidation label is invalid")
        current = load_stage6_source_repair_context(
            stage_root=self.stage_root,
            current_execution_identity=self.current_execution_identity,
            current_verified_review_authorization=self.current_verified_review_authorization,
            current_immutable_bindings=self.current_immutable_bindings,
        )
        if current is None or (
            current.amendment_sha256,
            current.amendment_size_bytes,
            current.continuation_sha256,
            current.continuation_size_bytes,
            current.supplement_sha256,
            current.supplement_size_bytes,
            current.closure_sha256,
            current.closure_size_bytes,
            current.frontier_recovery_sha256,
            current.frontier_recovery_size_bytes,
            current.sensor_acceleration_sha256,
            current.sensor_acceleration_size_bytes,
            current.coverage_cache_manifest_path,
            current.coverage_cache_manifest_sha256,
            current.coverage_cache_manifest_size_bytes,
            current.coverage_cache_entry_set_sha256,
            current.coverage_cache_root,
            current.coverage_cache_runtime_mode,
            current.coverage_cache_formal_audit_sha256,
            current.coverage_cache_formal_audit_size_bytes,
        ) != (
            self.amendment_sha256,
            self.amendment_size_bytes,
            self.continuation_sha256,
            self.continuation_size_bytes,
            self.supplement_sha256,
            self.supplement_size_bytes,
            self.closure_sha256,
            self.closure_size_bytes,
            self.frontier_recovery_sha256,
            self.frontier_recovery_size_bytes,
            self.sensor_acceleration_sha256,
            self.sensor_acceleration_size_bytes,
            self.coverage_cache_manifest_path,
            self.coverage_cache_manifest_sha256,
            self.coverage_cache_manifest_size_bytes,
            self.coverage_cache_entry_set_sha256,
            self.coverage_cache_root,
            self.coverage_cache_runtime_mode,
            self.coverage_cache_formal_audit_sha256,
            self.coverage_cache_formal_audit_size_bytes,
        ):
            raise Stage6SourceRepairError(f"source repair changed at {label}")


def _context_from_value(
    *,
    stage: Path,
    amendment_payload: bytes,
    value: Mapping[str, object],
    prefix: Mapping[str, bytes],
    continuation_payload: bytes | None = None,
    continuation_value: Mapping[str, object] | None = None,
    supplement_payload: bytes | None = None,
    supplement_value: Mapping[str, object] | None = None,
    closure_payload: bytes | None = None,
    closure_value: Mapping[str, object] | None = None,
    frontier_recovery_payload: bytes | None = None,
    frontier_recovery_value: Mapping[str, object] | None = None,
    sensor_acceleration_payload: bytes | None = None,
    sensor_acceleration_value: Mapping[str, object] | None = None,
) -> Stage6SourceRepairContext:
    lineage = _strict_json(prefix["lineage_audit.json"], label="origin lineage audit")
    parent = value["current"]
    continuation_current = (
        continuation_value["current"]
        if continuation_value is not None
        else None
    )
    supplement_current = (
        supplement_value["current"] if supplement_value is not None else None
    )
    closure_current = closure_value["current"] if closure_value is not None else None
    frontier_recovery_current = (
        frontier_recovery_value["current"]
        if frontier_recovery_value is not None
        else None
    )
    sensor_acceleration_current = (
        sensor_acceleration_value["current"]
        if sensor_acceleration_value is not None
        else None
    )
    current = (
        sensor_acceleration_current
        if sensor_acceleration_current is not None
        else (
            frontier_recovery_current
            if frontier_recovery_current is not None
            else (
                closure_current
                if closure_current is not None
                else (
                    supplement_current
                    if supplement_current is not None
                    else (
                        continuation_current
                        if continuation_current is not None
                        else parent
                    )
                )
            )
        )
    )
    if not isinstance(parent, Mapping) or not isinstance(current, Mapping):
        raise Stage6SourceRepairError("source-repair current identity is missing")
    if (continuation_payload is None) != (continuation_value is None):
        raise Stage6SourceRepairError("source-repair continuation context drifted")
    if (
        (supplement_payload is None) != (supplement_value is None)
        or (supplement_payload is not None and continuation_payload is None)
        or (
            continuation_current is not None
            and not isinstance(continuation_current, Mapping)
        )
    ):
        raise Stage6SourceRepairError("source-repair supplement context drifted")
    if (
        (closure_payload is None) != (closure_value is None)
        or (closure_payload is not None and supplement_payload is None)
        or (
            supplement_current is not None
            and not isinstance(supplement_current, Mapping)
        )
        or (closure_current is not None and not isinstance(closure_current, Mapping))
    ):
        raise Stage6SourceRepairError("source-repair closure context drifted")
    if (
        (frontier_recovery_payload is None) != (frontier_recovery_value is None)
        or (frontier_recovery_payload is not None and closure_payload is None)
        or (closure_current is not None and not isinstance(closure_current, Mapping))
        or (
            frontier_recovery_current is not None
            and not isinstance(frontier_recovery_current, Mapping)
        )
    ):
        raise Stage6SourceRepairError(
            "source-repair frontier recovery context drifted"
        )
    if (
        (sensor_acceleration_payload is None)
        != (sensor_acceleration_value is None)
        or (
            sensor_acceleration_payload is not None
            and frontier_recovery_payload is None
        )
        or (
            sensor_acceleration_current is not None
            and not isinstance(sensor_acceleration_current, Mapping)
        )
    ):
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration context drifted"
        )
    cache_manifest: Mapping[str, object] | None = None
    cache_audit: Mapping[str, object] | None = None
    if sensor_acceleration_value is not None:
        production_cache = sensor_acceleration_value.get("production_cache")
        if not isinstance(production_cache, Mapping):
            raise Stage6SourceRepairError(
                "source-repair sensor acceleration cache binding is missing"
            )
        candidate_manifest = production_cache.get("manifest")
        candidate_audit = production_cache.get("formal_audit")
        if not isinstance(candidate_manifest, Mapping) or not isinstance(
            candidate_audit, Mapping
        ):
            raise Stage6SourceRepairError(
                "source-repair sensor acceleration cache binding is missing"
            )
        cache_manifest = candidate_manifest
        cache_audit = candidate_audit
    checkpoint_payload = _read_cutover_checkpoint_payload(
        prefix[
            f"checkpoints/seed-{FIXED_SEED}/update-{LAST_ORIGIN_UPDATE:08d}/checkpoint.pt"
        ]
    )
    checkpoint_lineage = checkpoint_payload.get("lineage")
    if not isinstance(checkpoint_lineage, Mapping):
        raise Stage6SourceRepairError("source-repair checkpoint lineage is missing")
    return Stage6SourceRepairContext(
        stage_root=stage,
        amendment_sha256=_sha256(amendment_payload),
        amendment_size_bytes=len(amendment_payload),
        continuation_sha256=(
            _sha256(continuation_payload)
            if continuation_payload is not None
            else None
        ),
        continuation_size_bytes=(
            len(continuation_payload)
            if continuation_payload is not None
            else None
        ),
        supplement_sha256=(
            _sha256(supplement_payload)
            if supplement_payload is not None
            else None
        ),
        supplement_size_bytes=(
            len(supplement_payload) if supplement_payload is not None else None
        ),
        closure_sha256=(
            _sha256(closure_payload) if closure_payload is not None else None
        ),
        closure_size_bytes=(len(closure_payload) if closure_payload is not None else None),
        frontier_recovery_sha256=(
            _sha256(frontier_recovery_payload)
            if frontier_recovery_payload is not None
            else None
        ),
        frontier_recovery_size_bytes=(
            len(frontier_recovery_payload)
            if frontier_recovery_payload is not None
            else None
        ),
        sensor_acceleration_sha256=(
            _sha256(sensor_acceleration_payload)
            if sensor_acceleration_payload is not None
            else None
        ),
        sensor_acceleration_size_bytes=(
            len(sensor_acceleration_payload)
            if sensor_acceleration_payload is not None
            else None
        ),
        origin_execution_identity=MappingProxyType(dict(lineage["execution_identity"])),
        origin_immutable_bindings=MappingProxyType(dict(lineage["immutable_bindings"])),
        origin_verified_review_authorization=MappingProxyType(
            dict(lineage["verified_review_authorization"])
        ),
        parent_execution_identity=MappingProxyType(
            dict(parent["execution_identity"])  # type: ignore[arg-type]
        ),
        parent_immutable_bindings=MappingProxyType(
            dict(parent["immutable_bindings"])  # type: ignore[arg-type]
        ),
        parent_verified_review_authorization=MappingProxyType(
            dict(parent["verified_review_authorization"])  # type: ignore[arg-type]
        ),
        continuation_execution_identity=(
            MappingProxyType(
                dict(continuation_current["execution_identity"])  # type: ignore[index,arg-type]
            )
            if continuation_current is not None
            else None
        ),
        continuation_immutable_bindings=(
            MappingProxyType(
                dict(continuation_current["immutable_bindings"])  # type: ignore[index,arg-type]
            )
            if continuation_current is not None
            else None
        ),
        continuation_verified_review_authorization=(
            MappingProxyType(
                dict(continuation_current["verified_review_authorization"])  # type: ignore[index,arg-type]
            )
            if continuation_current is not None
            else None
        ),
        supplement_execution_identity=(
            MappingProxyType(
                dict(supplement_current["execution_identity"])  # type: ignore[index,arg-type]
            )
            if supplement_current is not None
            else None
        ),
        supplement_immutable_bindings=(
            MappingProxyType(
                dict(supplement_current["immutable_bindings"])  # type: ignore[index,arg-type]
            )
            if supplement_current is not None
            else None
        ),
        supplement_verified_review_authorization=(
            MappingProxyType(
                dict(supplement_current["verified_review_authorization"])  # type: ignore[index,arg-type]
            )
            if supplement_current is not None
            else None
        ),
        closure_execution_identity=(
            MappingProxyType(
                dict(closure_current["execution_identity"])  # type: ignore[index,arg-type]
            )
            if closure_current is not None
            else None
        ),
        closure_immutable_bindings=(
            MappingProxyType(
                dict(closure_current["immutable_bindings"])  # type: ignore[index,arg-type]
            )
            if closure_current is not None
            else None
        ),
        closure_verified_review_authorization=(
            MappingProxyType(
                dict(closure_current["verified_review_authorization"])  # type: ignore[index,arg-type]
            )
            if closure_current is not None
            else None
        ),
        frontier_recovery_execution_identity=(
            MappingProxyType(
                dict(frontier_recovery_current["execution_identity"])  # type: ignore[index,arg-type]
            )
            if frontier_recovery_current is not None
            else None
        ),
        frontier_recovery_immutable_bindings=(
            MappingProxyType(
                dict(frontier_recovery_current["immutable_bindings"])  # type: ignore[index,arg-type]
            )
            if frontier_recovery_current is not None
            else None
        ),
        frontier_recovery_verified_review_authorization=(
            MappingProxyType(
                dict(frontier_recovery_current["verified_review_authorization"])  # type: ignore[index,arg-type]
            )
            if frontier_recovery_current is not None
            else None
        ),
        current_execution_identity=MappingProxyType(
            dict(current["execution_identity"])  # type: ignore[arg-type]
        ),
        current_immutable_bindings=MappingProxyType(
            dict(current["immutable_bindings"])  # type: ignore[arg-type]
        ),
        current_verified_review_authorization=MappingProxyType(
            dict(current["verified_review_authorization"])  # type: ignore[arg-type]
        ),
        origin_checkpoint_lineage=MappingProxyType(dict(checkpoint_lineage)),
        coverage_cache_manifest_path=(
            str(cache_manifest["path"])
            if cache_manifest is not None
            else None
        ),
        coverage_cache_manifest_sha256=(
            str(cache_manifest["sha256"])
            if cache_manifest is not None
            else None
        ),
        coverage_cache_manifest_size_bytes=(
            int(cache_manifest["size_bytes"])
            if cache_manifest is not None
            else None
        ),
        coverage_cache_entry_set_sha256=(
            str(cache_manifest["entry_set_sha256"])
            if cache_manifest is not None
            else None
        ),
        coverage_cache_root=(
            str(cache_manifest["cache_root"])
            if cache_manifest is not None
            else None
        ),
        coverage_cache_entry_count=(
            int(cache_manifest["entry_count"])
            if cache_manifest is not None
            else None
        ),
        coverage_cache_split_counts=(
            MappingProxyType(dict(cache_manifest["split_counts"]))  # type: ignore[arg-type]
            if cache_manifest is not None
            else None
        ),
        coverage_cache_runtime_mode=(
            str(cache_manifest["runtime_mode"])
            if cache_manifest is not None
            else None
        ),
        coverage_cache_formal_audit_sha256=(
            str(cache_audit["sha256"])
            if cache_audit is not None
            else None
        ),
        coverage_cache_formal_audit_size_bytes=(
            int(cache_audit["size_bytes"])
            if cache_audit is not None
            else None
        ),
    )


def _primary_context_from_payload(
    *,
    stage: Path,
    payload: bytes,
) -> Stage6SourceRepairContext:
    _require_fixed_primary_amendment(payload)
    value = _strict_json(payload, label="source-repair amendment")
    current = value.get("current")
    if not isinstance(current, Mapping):
        raise Stage6SourceRepairError("source-repair current identity is missing")
    current_identity = current.get("execution_identity")
    current_review = current.get("verified_review_authorization")
    current_immutable = current.get("immutable_bindings")
    if (
        not isinstance(current_identity, Mapping)
        or not isinstance(current_review, Mapping)
        or not isinstance(current_immutable, Mapping)
    ):
        raise Stage6SourceRepairError("source-repair current authority is missing")
    expected = _amendment_value(
        stage=stage,
        current_execution_identity=current_identity,
        current_verified_review_authorization=current_review,
        current_immutable_bindings=current_immutable,
        allow_growth=True,
    )
    if value != expected:
        raise Stage6SourceRepairError("source-repair amendment identity drifted")
    prefix = _fixed_prefix_payloads(stage, allow_growth=True)
    return _context_from_value(
        stage=stage,
        amendment_payload=payload,
        value=value,
        prefix=prefix,
    )


def _continuation_context_from_payload(
    *,
    stage: Path,
    primary_context: Stage6SourceRepairContext,
    amendment_payload: bytes,
    continuation_payload: bytes,
) -> Stage6SourceRepairContext:
    _require_fixed_continuation(continuation_payload)
    value = _strict_json(
        continuation_payload,
        label="source-repair continuation",
    )
    current = value.get("current")
    if not isinstance(current, Mapping):
        raise Stage6SourceRepairError(
            "source-repair continuation current identity is missing"
        )
    current_identity = current.get("execution_identity")
    current_review = current.get("verified_review_authorization")
    current_immutable = current.get("immutable_bindings")
    if (
        not isinstance(current_identity, Mapping)
        or not isinstance(current_review, Mapping)
        or not isinstance(current_immutable, Mapping)
    ):
        raise Stage6SourceRepairError(
            "source-repair continuation current authority is missing"
        )
    expected = _continuation_value(
        stage=stage,
        primary_context=primary_context,
        primary_payload=amendment_payload,
        current_execution_identity=current_identity,
        current_verified_review_authorization=current_review,
        current_immutable_bindings=current_immutable,
        allow_growth=True,
    )
    if value != expected:
        raise Stage6SourceRepairError(
            "source-repair continuation identity drifted"
        )
    prefix = _continuation_prefix_payloads(stage, allow_growth=True)
    return _context_from_value(
        stage=stage,
        amendment_payload=amendment_payload,
        value=_strict_json(
            amendment_payload,
            label="source-repair amendment",
        ),
        prefix=prefix,
        continuation_payload=continuation_payload,
        continuation_value=value,
    )


def _supplement_context_from_payload(
    *,
    stage: Path,
    primary_context: Stage6SourceRepairContext,
    continuation_context: Stage6SourceRepairContext,
    amendment_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
) -> Stage6SourceRepairContext:
    value = _strict_json(
        supplement_payload,
        label="source-repair supplement",
    )
    current = value.get("current")
    if not isinstance(current, Mapping):
        raise Stage6SourceRepairError(
            "source-repair supplement current identity is missing"
        )
    current_identity = current.get("execution_identity")
    current_review = current.get("verified_review_authorization")
    current_immutable = current.get("immutable_bindings")
    if (
        not isinstance(current_identity, Mapping)
        or not isinstance(current_review, Mapping)
        or not isinstance(current_immutable, Mapping)
    ):
        raise Stage6SourceRepairError(
            "source-repair supplement current authority is missing"
        )
    expected = _supplement_value(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        primary_payload=amendment_payload,
        continuation_payload=continuation_payload,
        current_execution_identity=current_identity,
        current_verified_review_authorization=current_review,
        current_immutable_bindings=current_immutable,
        allow_growth=True,
    )
    if value != expected:
        raise Stage6SourceRepairError("source-repair supplement identity drifted")
    prefix = _supplement_prefix_payloads(stage, allow_growth=True)
    return _context_from_value(
        stage=stage,
        amendment_payload=amendment_payload,
        value=_strict_json(
            amendment_payload,
            label="source-repair amendment",
        ),
        prefix=prefix,
        continuation_payload=continuation_payload,
        continuation_value=_strict_json(
            continuation_payload,
            label="source-repair continuation",
        ),
        supplement_payload=supplement_payload,
        supplement_value=value,
    )


def _closure_context_from_payload(
    *,
    stage: Path,
    primary_context: Stage6SourceRepairContext,
    continuation_context: Stage6SourceRepairContext,
    supplement_context: Stage6SourceRepairContext,
    amendment_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    closure_payload: bytes,
) -> Stage6SourceRepairContext:
    value = _strict_json(
        closure_payload,
        label="source-repair closure",
    )
    current = value.get("current")
    if not isinstance(current, Mapping):
        raise Stage6SourceRepairError(
            "source-repair closure current identity is missing"
        )
    current_identity = current.get("execution_identity")
    current_review = current.get("verified_review_authorization")
    current_immutable = current.get("immutable_bindings")
    if (
        not isinstance(current_identity, Mapping)
        or not isinstance(current_review, Mapping)
        or not isinstance(current_immutable, Mapping)
    ):
        raise Stage6SourceRepairError(
            "source-repair closure current authority is missing"
        )
    expected = _closure_value(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        supplement_context=supplement_context,
        primary_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        current_execution_identity=current_identity,
        current_verified_review_authorization=current_review,
        current_immutable_bindings=current_immutable,
        allow_growth=True,
    )
    if value != expected:
        raise Stage6SourceRepairError("source-repair closure identity drifted")
    prefix = _closure_prefix_payloads(stage, allow_growth=True)
    return _context_from_value(
        stage=stage,
        amendment_payload=amendment_payload,
        value=_strict_json(
            amendment_payload,
            label="source-repair amendment",
        ),
        prefix=prefix,
        continuation_payload=continuation_payload,
        continuation_value=_strict_json(
            continuation_payload,
            label="source-repair continuation",
        ),
        supplement_payload=supplement_payload,
        supplement_value=_strict_json(
            supplement_payload,
            label="source-repair supplement",
        ),
        closure_payload=closure_payload,
        closure_value=value,
    )


def _frontier_recovery_context_from_payload(
    *,
    stage: Path,
    primary_context: Stage6SourceRepairContext,
    continuation_context: Stage6SourceRepairContext,
    supplement_context: Stage6SourceRepairContext,
    closure_context: Stage6SourceRepairContext,
    amendment_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    closure_payload: bytes,
    frontier_recovery_payload: bytes,
    requested_execution_identity: Mapping[str, object],
) -> Stage6SourceRepairContext:
    value = _strict_json(
        frontier_recovery_payload,
        label="source-repair frontier recovery",
    )
    current = value.get("current")
    if not isinstance(current, Mapping):
        raise Stage6SourceRepairError(
            "source-repair frontier recovery current identity is missing"
        )
    current_identity = current.get("execution_identity")
    current_review = current.get("verified_review_authorization")
    current_immutable = current.get("immutable_bindings")
    if (
        not isinstance(current_identity, Mapping)
        or not isinstance(current_review, Mapping)
        or not isinstance(current_immutable, Mapping)
    ):
        raise Stage6SourceRepairError(
            "source-repair frontier recovery current authority is missing"
        )
    fixed_binding = {
        "path": SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
        "sha256": _FIXED_FRONTIER_RECOVERY_BINDING.get("sha256"),
        "size_bytes": _FIXED_FRONTIER_RECOVERY_BINDING.get("size_bytes"),
    }
    as_fixed_historical_parent = (
        dict(current_identity) != dict(requested_execution_identity)
        and _FIXED_FRONTIER_RECOVERY_BINDING == fixed_binding
        and {
            "path": SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
            "sha256": _sha256(frontier_recovery_payload),
            "size_bytes": len(frontier_recovery_payload),
        }
        == fixed_binding
        and _canonical_sha256(dict(current_identity))
        == _FIXED_FRONTIER_RECOVERY_EXECUTION_IDENTITY_SHA256
    )
    if as_fixed_historical_parent:
        _require_fixed_frontier_recovery(frontier_recovery_payload)
    expected = _frontier_recovery_value(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        supplement_context=supplement_context,
        closure_context=closure_context,
        primary_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        closure_payload=closure_payload,
        current_execution_identity=current_identity,
        current_verified_review_authorization=current_review,
        current_immutable_bindings=current_immutable,
        allow_growth=True,
        validate_current_sources=not as_fixed_historical_parent,
    )
    if value != expected:
        raise Stage6SourceRepairError(
            "source-repair frontier recovery identity drifted"
        )
    prefix = _frontier_recovery_prefix_payloads(stage, allow_growth=True)
    return _context_from_value(
        stage=stage,
        amendment_payload=amendment_payload,
        value=_strict_json(
            amendment_payload,
            label="source-repair amendment",
        ),
        prefix=prefix,
        continuation_payload=continuation_payload,
        continuation_value=_strict_json(
            continuation_payload,
            label="source-repair continuation",
        ),
        supplement_payload=supplement_payload,
        supplement_value=_strict_json(
            supplement_payload,
            label="source-repair supplement",
        ),
        closure_payload=closure_payload,
        closure_value=_strict_json(
            closure_payload,
            label="source-repair closure",
        ),
        frontier_recovery_payload=frontier_recovery_payload,
        frontier_recovery_value=value,
    )


def _sensor_acceleration_context_from_payload(
    *,
    stage: Path,
    parent_context: Stage6SourceRepairContext,
    amendment_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    closure_payload: bytes,
    frontier_recovery_payload: bytes,
    sensor_acceleration_payload: bytes,
) -> Stage6SourceRepairContext:
    value = _strict_json(
        sensor_acceleration_payload,
        label="source-repair sensor acceleration",
    )
    current = value.get("current")
    if not isinstance(current, Mapping):
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration current identity is missing"
        )
    current_identity = current.get("execution_identity")
    current_review = current.get("verified_review_authorization")
    current_immutable = current.get("immutable_bindings")
    if (
        not isinstance(current_identity, Mapping)
        or not isinstance(current_review, Mapping)
        or not isinstance(current_immutable, Mapping)
    ):
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration current authority is missing"
        )
    expected = _sensor_acceleration_value(
        stage=stage,
        parent_context=parent_context,
        primary_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        closure_payload=closure_payload,
        frontier_recovery_payload=frontier_recovery_payload,
        current_execution_identity=current_identity,
        current_verified_review_authorization=current_review,
        current_immutable_bindings=current_immutable,
        allow_growth=True,
    )
    if value != expected:
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration identity drifted"
        )
    prefix = _sensor_acceleration_prefix_payloads(stage, allow_growth=True)
    return _context_from_value(
        stage=stage,
        amendment_payload=amendment_payload,
        value=_strict_json(
            amendment_payload,
            label="source-repair amendment",
        ),
        prefix=prefix,
        continuation_payload=continuation_payload,
        continuation_value=_strict_json(
            continuation_payload,
            label="source-repair continuation",
        ),
        supplement_payload=supplement_payload,
        supplement_value=_strict_json(
            supplement_payload,
            label="source-repair supplement",
        ),
        closure_payload=closure_payload,
        closure_value=_strict_json(
            closure_payload,
            label="source-repair closure",
        ),
        frontier_recovery_payload=frontier_recovery_payload,
        frontier_recovery_value=_strict_json(
            frontier_recovery_payload,
            label="source-repair frontier recovery",
        ),
        sensor_acceleration_payload=sensor_acceleration_payload,
        sensor_acceleration_value=value,
    )


def _context_current_matches(
    context: Stage6SourceRepairContext,
    *,
    execution_identity: Mapping[str, object],
    verified_review_authorization: Mapping[str, object],
    immutable_bindings: Mapping[str, object],
) -> bool:
    return (
        dict(context.current_execution_identity) == dict(execution_identity)
        and dict(context.current_verified_review_authorization)
        == dict(verified_review_authorization)
        and dict(context.current_immutable_bindings) == dict(immutable_bindings)
    )


def load_stage6_source_repair_context(
    *,
    stage_root: str | Path,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
) -> Stage6SourceRepairContext | None:
    candidate = lexical_absolute(stage_root)
    amendment_path = candidate / SOURCE_REPAIR_AMENDMENT_NAME
    continuation_path = candidate / SOURCE_REPAIR_CONTINUATION_NAME
    supplement_path = candidate / SOURCE_REPAIR_SUPPLEMENT_NAME
    closure_path = candidate / SOURCE_REPAIR_CLOSURE_NAME
    frontier_recovery_path = candidate / SOURCE_REPAIR_FRONTIER_RECOVERY_NAME
    sensor_acceleration_path = candidate / SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    successor_paths = (
        continuation_path,
        supplement_path,
        closure_path,
        frontier_recovery_path,
        sensor_acceleration_path,
    )
    if not os.path.lexists(amendment_path):
        if any(os.path.lexists(path) for path in successor_paths):
            raise Stage6SourceRepairError(
                "source-repair successor exists without primary amendment"
            )
        return None
    stage = _stage_root(candidate)
    amendment_payload = _secure_payload(
        amendment_path,
        base=stage,
        label="source-repair amendment",
    )
    primary_context = _primary_context_from_payload(
        stage=stage,
        payload=amendment_payload,
    )
    contexts = [primary_context]
    if not os.path.lexists(continuation_path):
        if any(
            os.path.lexists(path)
            for path in (
                supplement_path,
                closure_path,
                frontier_recovery_path,
                sensor_acceleration_path,
            )
        ):
            raise Stage6SourceRepairError(
                "source-repair successor exists without continuation"
            )
        if _context_current_matches(
            primary_context,
            execution_identity=current_execution_identity,
            verified_review_authorization=current_verified_review_authorization,
            immutable_bindings=current_immutable_bindings,
        ):
            return primary_context
        raise Stage6SourceRepairError(
            "source-repair continuation is required for the current identity"
        )
    continuation_payload = _secure_payload(
        continuation_path,
        base=stage,
        label="source-repair continuation",
    )
    continuation_context = _continuation_context_from_payload(
        stage=stage,
        primary_context=primary_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
    )
    contexts.append(continuation_context)
    if not os.path.lexists(supplement_path):
        if (
            os.path.lexists(closure_path)
            or os.path.lexists(frontier_recovery_path)
            or os.path.lexists(sensor_acceleration_path)
        ):
            raise Stage6SourceRepairError(
                "source-repair closure exists without supplement"
            )
        for context in contexts:
            if _context_current_matches(
                context,
                execution_identity=current_execution_identity,
                verified_review_authorization=current_verified_review_authorization,
                immutable_bindings=current_immutable_bindings,
            ):
                return context
        raise Stage6SourceRepairError(
            "source-repair supplement is required for the current identity"
        )
    supplement_payload = _secure_payload(
        supplement_path,
        base=stage,
        label="source-repair supplement",
    )
    supplement_context = _supplement_context_from_payload(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
    )
    contexts.append(supplement_context)
    if not os.path.lexists(closure_path):
        if os.path.lexists(frontier_recovery_path) or os.path.lexists(
            sensor_acceleration_path
        ):
            raise Stage6SourceRepairError(
                "source-repair frontier recovery exists without closure"
            )
        for context in contexts:
            if _context_current_matches(
                context,
                execution_identity=current_execution_identity,
                verified_review_authorization=current_verified_review_authorization,
                immutable_bindings=current_immutable_bindings,
            ):
                return context
        raise Stage6SourceRepairError(
            "future source-repair identity requires closure"
        )
    closure_payload = _secure_payload(
        closure_path,
        base=stage,
        label="source-repair closure",
    )
    closure_context = _closure_context_from_payload(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        supplement_context=supplement_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        closure_payload=closure_payload,
    )
    contexts.append(closure_context)
    if not os.path.lexists(frontier_recovery_path):
        if os.path.lexists(sensor_acceleration_path):
            raise Stage6SourceRepairError(
                "source-repair sensor acceleration exists without frontier recovery"
            )
        for context in contexts:
            if _context_current_matches(
                context,
                execution_identity=current_execution_identity,
                verified_review_authorization=current_verified_review_authorization,
                immutable_bindings=current_immutable_bindings,
            ):
                return context
        raise Stage6SourceRepairError(
            "source-repair frontier recovery is required for the current identity; "
            "future source-repair identity requires a fifth immutable ordinal"
        )
    frontier_recovery_payload = _secure_payload(
        frontier_recovery_path,
        base=stage,
        label="source-repair frontier recovery",
    )
    frontier_recovery_context = _frontier_recovery_context_from_payload(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        supplement_context=supplement_context,
        closure_context=closure_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        closure_payload=closure_payload,
        frontier_recovery_payload=frontier_recovery_payload,
        requested_execution_identity=current_execution_identity,
    )
    contexts.append(frontier_recovery_context)
    if not os.path.lexists(sensor_acceleration_path):
        for context in contexts:
            if _context_current_matches(
                context,
                execution_identity=current_execution_identity,
                verified_review_authorization=current_verified_review_authorization,
                immutable_bindings=current_immutable_bindings,
            ):
                return context
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration is required for the current identity; "
            "future source-repair identity requires a sixth immutable ordinal"
        )
    sensor_acceleration_payload = _secure_payload(
        sensor_acceleration_path,
        base=stage,
        label="source-repair sensor acceleration",
    )
    sensor_acceleration_context = _sensor_acceleration_context_from_payload(
        stage=stage,
        parent_context=frontier_recovery_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        closure_payload=closure_payload,
        frontier_recovery_payload=frontier_recovery_payload,
        sensor_acceleration_payload=sensor_acceleration_payload,
    )
    contexts.append(sensor_acceleration_context)
    for context in contexts:
        if _context_current_matches(
            context,
            execution_identity=current_execution_identity,
            verified_review_authorization=current_verified_review_authorization,
            immutable_bindings=current_immutable_bindings,
        ):
            return context
    raise Stage6SourceRepairError(
        "future source-repair identity requires a seventh immutable ordinal"
    )


def _publish_source_repair_payload(
    *,
    stage: Path,
    name: str,
    payload: bytes,
    label: str,
) -> None:
    path = stage / name
    try:
        ArtifactStore(stage).write_bytes_exclusive(name, payload)
    except FileExistsError:
        current = _secure_payload(path, base=stage, label=label)
        if current != payload:
            raise Stage6SourceRepairError(f"{label} publication raced")


def _create_or_preview_closure(
    *,
    stage: Path,
    primary_context: Stage6SourceRepairContext,
    continuation_context: Stage6SourceRepairContext,
    supplement_context: Stage6SourceRepairContext,
    amendment_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    publish: bool,
) -> Stage6SourceRepairContext:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    lease = RunLease(stage.parent / ".stage6.lease")
    try:
        lease.acquire()
    except RunLeaseError as exc:
        raise Stage6SourceRepairError(
            "closure requires an available Stage 6 run lease"
        ) from exc
    try:
        value = _closure_value(
            stage=stage,
            primary_context=primary_context,
            continuation_context=continuation_context,
            supplement_context=supplement_context,
            primary_payload=amendment_payload,
            continuation_payload=continuation_payload,
            supplement_payload=supplement_payload,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
            allow_growth=False,
        )
        payload = ArtifactStore.canonical_json_bytes(value)
        if not publish:
            return _context_from_value(
                stage=stage,
                amendment_payload=amendment_payload,
                value=_strict_json(
                    amendment_payload,
                    label="source-repair amendment",
                ),
                prefix=_closure_prefix_payloads(stage, allow_growth=False),
                continuation_payload=continuation_payload,
                continuation_value=_strict_json(
                    continuation_payload,
                    label="source-repair continuation",
                ),
                supplement_payload=supplement_payload,
                supplement_value=_strict_json(
                    supplement_payload,
                    label="source-repair supplement",
                ),
                closure_payload=payload,
                closure_value=value,
            )
        _publish_source_repair_payload(
            stage=stage,
            name=SOURCE_REPAIR_CLOSURE_NAME,
            payload=payload,
            label="source-repair closure",
        )
        context = load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
        )
        if context is None or context.closure_sha256 is None:
            raise Stage6SourceRepairError(
                "source-repair closure did not become durable"
            )
        return context
    finally:
        try:
            lease.release()
        except RunLeaseError as exc:
            raise Stage6SourceRepairError("closure run lease release failed") from exc


def _create_or_preview_frontier_recovery(
    *,
    stage: Path,
    primary_context: Stage6SourceRepairContext,
    continuation_context: Stage6SourceRepairContext,
    supplement_context: Stage6SourceRepairContext,
    closure_context: Stage6SourceRepairContext,
    amendment_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    closure_payload: bytes,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    publish: bool,
) -> Stage6SourceRepairContext:
    def _payload_and_value() -> tuple[bytes, dict[str, object]]:
        value = _frontier_recovery_value(
            stage=stage,
            primary_context=primary_context,
            continuation_context=continuation_context,
            supplement_context=supplement_context,
            closure_context=closure_context,
            primary_payload=amendment_payload,
            continuation_payload=continuation_payload,
            supplement_payload=supplement_payload,
            closure_payload=closure_payload,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
            allow_growth=False,
            validate_current_sources=True,
        )
        return ArtifactStore.canonical_json_bytes(value), value

    if not publish:
        payload, value = _payload_and_value()
        return _context_from_value(
            stage=stage,
            amendment_payload=amendment_payload,
            value=_strict_json(
                amendment_payload,
                label="source-repair amendment",
            ),
            prefix=_frontier_recovery_prefix_payloads(
                stage,
                allow_growth=False,
            ),
            continuation_payload=continuation_payload,
            continuation_value=_strict_json(
                continuation_payload,
                label="source-repair continuation",
            ),
            supplement_payload=supplement_payload,
            supplement_value=_strict_json(
                supplement_payload,
                label="source-repair supplement",
            ),
            closure_payload=closure_payload,
            closure_value=_strict_json(
                closure_payload,
                label="source-repair closure",
            ),
            frontier_recovery_payload=payload,
            frontier_recovery_value=value,
        )

    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    lease = RunLease(stage.parent / ".stage6.lease")
    try:
        lease.acquire()
    except RunLeaseError as exc:
        raise Stage6SourceRepairError(
            "frontier recovery requires an available Stage 6 run lease"
        ) from exc
    try:
        payload, _value = _payload_and_value()
        _publish_source_repair_payload(
            stage=stage,
            name=SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
            payload=payload,
            label="source-repair frontier recovery",
        )
        context = load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
        )
        if context is None or context.frontier_recovery_sha256 is None:
            raise Stage6SourceRepairError(
                "source-repair frontier recovery did not become durable"
            )
        return context
    finally:
        try:
            lease.release()
        except RunLeaseError as exc:
            raise Stage6SourceRepairError(
                "frontier recovery run lease release failed"
            ) from exc


def _create_or_preview_sensor_acceleration(
    *,
    stage: Path,
    parent_context: Stage6SourceRepairContext,
    amendment_payload: bytes,
    continuation_payload: bytes,
    supplement_payload: bytes,
    closure_payload: bytes,
    frontier_recovery_payload: bytes,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    publish: bool,
) -> Stage6SourceRepairContext:
    value = _sensor_acceleration_value(
        stage=stage,
        parent_context=parent_context,
        primary_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        closure_payload=closure_payload,
        frontier_recovery_payload=frontier_recovery_payload,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=(
            current_verified_review_authorization
        ),
        current_immutable_bindings=current_immutable_bindings,
        allow_growth=False,
    )
    payload = ArtifactStore.canonical_json_bytes(value)
    if not publish:
        return _context_from_value(
            stage=stage,
            amendment_payload=amendment_payload,
            value=_strict_json(
                amendment_payload,
                label="source-repair amendment",
            ),
            prefix=_sensor_acceleration_prefix_payloads(
                stage,
                allow_growth=False,
            ),
            continuation_payload=continuation_payload,
            continuation_value=_strict_json(
                continuation_payload,
                label="source-repair continuation",
            ),
            supplement_payload=supplement_payload,
            supplement_value=_strict_json(
                supplement_payload,
                label="source-repair supplement",
            ),
            closure_payload=closure_payload,
            closure_value=_strict_json(
                closure_payload,
                label="source-repair closure",
            ),
            frontier_recovery_payload=frontier_recovery_payload,
            frontier_recovery_value=_strict_json(
                frontier_recovery_payload,
                label="source-repair frontier recovery",
            ),
            sensor_acceleration_payload=payload,
            sensor_acceleration_value=value,
        )

    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    lease = RunLease(stage.parent / ".stage6.lease")
    try:
        lease.acquire()
    except RunLeaseError as exc:
        raise Stage6SourceRepairError(
            "sensor acceleration requires an available Stage 6 run lease"
        ) from exc
    try:
        lease.require_current()
        current_parent = _secure_payload(
            stage / SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
            base=stage,
            label="source-repair frontier recovery",
        )
        if current_parent != frontier_recovery_payload:
            raise Stage6SourceRepairError(
                "sensor acceleration parent changed before publication"
            )
        if os.path.lexists(
            stage
            / f"checkpoints/seed-{FIXED_SEED}/"
            f"update-{FIRST_SENSOR_ACCELERATION_UPDATE:08d}"
        ):
            raise Stage6SourceRepairError(
                "sensor acceleration requires an Update 50 checkpoint-free boundary"
            )
        critical_prefix = _sensor_acceleration_prefix_payloads(
            stage,
            allow_growth=False,
        )
        _checkpoint_identity, critical_prefix_semantics = (
            _validate_sensor_acceleration_prefix_semantics(
                stage,
                critical_prefix,
                parent_context=parent_context,
                at_creation=False,
            )
        )
        critical_root_pid = critical_prefix_semantics.get(
            "previous_resource_root_pid"
        )
        if type(critical_root_pid) is not int or critical_root_pid <= 0:
            raise Stage6SourceRepairError(
                "sensor acceleration process snapshot is uncertain"
            )
        _require_sensor_acceleration_process_tree_stopped(critical_root_pid)
        _sensor_acceleration_source_payloads()
        _sensor_acceleration_evidence_payloads()
        _require_fixed_frontier_recovery(current_parent)
        _publish_source_repair_payload(
            stage=stage,
            name=SOURCE_REPAIR_SENSOR_ACCELERATION_NAME,
            payload=payload,
            label="source-repair sensor acceleration",
        )
        context = load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
        )
        if context is None or context.sensor_acceleration_sha256 is None:
            raise Stage6SourceRepairError(
                "source-repair sensor acceleration did not become durable"
            )
        return context
    finally:
        try:
            lease.release()
        except RunLeaseError as exc:
            raise Stage6SourceRepairError(
                "sensor acceleration run lease release failed"
            ) from exc


def create_stage6_source_repair_amendment(
    *,
    stage_root: str | Path,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
    publish: bool = False,
) -> Stage6SourceRepairContext:
    if type(publish) is not bool:
        raise Stage6SourceRepairError("source-repair publish flag is invalid")
    stage = _stage_root(stage_root)
    amendment_path = stage / SOURCE_REPAIR_AMENDMENT_NAME
    continuation_path = stage / SOURCE_REPAIR_CONTINUATION_NAME
    supplement_path = stage / SOURCE_REPAIR_SUPPLEMENT_NAME
    closure_path = stage / SOURCE_REPAIR_CLOSURE_NAME
    frontier_recovery_path = stage / SOURCE_REPAIR_FRONTIER_RECOVERY_NAME
    sensor_acceleration_path = stage / SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    successor_paths = (
        continuation_path,
        supplement_path,
        closure_path,
        frontier_recovery_path,
        sensor_acceleration_path,
    )

    if not os.path.lexists(amendment_path):
        if any(os.path.lexists(path) for path in successor_paths):
            raise Stage6SourceRepairError(
                "source-repair successor exists without primary amendment"
            )
        value = _amendment_value(
            stage=stage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=current_verified_review_authorization,
            current_immutable_bindings=current_immutable_bindings,
            allow_growth=False,
        )
        payload = ArtifactStore.canonical_json_bytes(value)
        if not publish:
            return _context_from_value(
                stage=stage,
                amendment_payload=payload,
                value=value,
                prefix=_fixed_prefix_payloads(stage, allow_growth=False),
            )
        _publish_source_repair_payload(
            stage=stage,
            name=SOURCE_REPAIR_AMENDMENT_NAME,
            payload=payload,
            label="source-repair amendment",
        )
        context = load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=current_verified_review_authorization,
            current_immutable_bindings=current_immutable_bindings,
        )
        if context is None:
            raise Stage6SourceRepairError(
                "source-repair amendment did not become durable"
            )
        return context

    amendment_payload = _secure_payload(
        amendment_path,
        base=stage,
        label="source-repair amendment",
    )
    primary_context = _primary_context_from_payload(
        stage=stage,
        payload=amendment_payload,
    )
    contexts = [primary_context]

    if not os.path.lexists(continuation_path):
        if any(
            os.path.lexists(path)
            for path in (
                supplement_path,
                closure_path,
                frontier_recovery_path,
                sensor_acceleration_path,
            )
        ):
            raise Stage6SourceRepairError(
                "source-repair successor exists without continuation"
            )
        if _context_current_matches(
            primary_context,
            execution_identity=current_execution_identity,
            verified_review_authorization=current_verified_review_authorization,
            immutable_bindings=current_immutable_bindings,
        ):
            return primary_context
        value = _continuation_value(
            stage=stage,
            primary_context=primary_context,
            primary_payload=amendment_payload,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
            allow_growth=False,
        )
        payload = ArtifactStore.canonical_json_bytes(value)
        if not publish:
            return _context_from_value(
                stage=stage,
                amendment_payload=amendment_payload,
                value=_strict_json(
                    amendment_payload,
                    label="source-repair amendment",
                ),
                prefix=_continuation_prefix_payloads(stage, allow_growth=False),
                continuation_payload=payload,
                continuation_value=value,
            )
        _publish_source_repair_payload(
            stage=stage,
            name=SOURCE_REPAIR_CONTINUATION_NAME,
            payload=payload,
            label="source-repair continuation",
        )
        context = load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=current_verified_review_authorization,
            current_immutable_bindings=current_immutable_bindings,
        )
        if context is None or context.continuation_sha256 is None:
            raise Stage6SourceRepairError(
                "source-repair continuation did not become durable"
            )
        return context

    continuation_payload = _secure_payload(
        continuation_path,
        base=stage,
        label="source-repair continuation",
    )
    continuation_context = _continuation_context_from_payload(
        stage=stage,
        primary_context=primary_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
    )
    contexts.append(continuation_context)

    if not os.path.lexists(supplement_path):
        if (
            os.path.lexists(closure_path)
            or os.path.lexists(frontier_recovery_path)
            or os.path.lexists(sensor_acceleration_path)
        ):
            raise Stage6SourceRepairError(
                "source-repair closure exists without supplement"
            )
        for context in contexts:
            if _context_current_matches(
                context,
                execution_identity=current_execution_identity,
                verified_review_authorization=current_verified_review_authorization,
                immutable_bindings=current_immutable_bindings,
            ):
                return context
        value = _supplement_value(
            stage=stage,
            primary_context=primary_context,
            continuation_context=continuation_context,
            primary_payload=amendment_payload,
            continuation_payload=continuation_payload,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
            allow_growth=False,
        )
        payload = ArtifactStore.canonical_json_bytes(value)
        if not publish:
            return _context_from_value(
                stage=stage,
                amendment_payload=amendment_payload,
                value=_strict_json(
                    amendment_payload,
                    label="source-repair amendment",
                ),
                prefix=_supplement_prefix_payloads(stage, allow_growth=False),
                continuation_payload=continuation_payload,
                continuation_value=_strict_json(
                    continuation_payload,
                    label="source-repair continuation",
                ),
                supplement_payload=payload,
                supplement_value=value,
            )
        _publish_source_repair_payload(
            stage=stage,
            name=SOURCE_REPAIR_SUPPLEMENT_NAME,
            payload=payload,
            label="source-repair supplement",
        )
        context = load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=current_verified_review_authorization,
            current_immutable_bindings=current_immutable_bindings,
        )
        if context is None or context.supplement_sha256 is None:
            raise Stage6SourceRepairError(
                "source-repair supplement did not become durable"
            )
        return context

    supplement_payload = _secure_payload(
        supplement_path,
        base=stage,
        label="source-repair supplement",
    )
    supplement_context = _supplement_context_from_payload(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
    )
    contexts.append(supplement_context)

    if not os.path.lexists(closure_path):
        if os.path.lexists(frontier_recovery_path) or os.path.lexists(
            sensor_acceleration_path
        ):
            raise Stage6SourceRepairError(
                "source-repair frontier recovery exists without closure"
            )
        for context in contexts:
            if _context_current_matches(
                context,
                execution_identity=current_execution_identity,
                verified_review_authorization=current_verified_review_authorization,
                immutable_bindings=current_immutable_bindings,
            ):
                return context
        return _create_or_preview_closure(
            stage=stage,
            primary_context=primary_context,
            continuation_context=continuation_context,
            supplement_context=supplement_context,
            amendment_payload=amendment_payload,
            continuation_payload=continuation_payload,
            supplement_payload=supplement_payload,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
            publish=publish,
        )

    closure_payload = _secure_payload(
        closure_path,
        base=stage,
        label="source-repair closure",
    )
    closure_context = _closure_context_from_payload(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        supplement_context=supplement_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        closure_payload=closure_payload,
    )
    contexts.append(closure_context)

    if not os.path.lexists(frontier_recovery_path):
        if os.path.lexists(sensor_acceleration_path):
            raise Stage6SourceRepairError(
                "source-repair sensor acceleration exists without frontier recovery"
            )
        for context in contexts:
            if _context_current_matches(
                context,
                execution_identity=current_execution_identity,
                verified_review_authorization=current_verified_review_authorization,
                immutable_bindings=current_immutable_bindings,
            ):
                return context
        return _create_or_preview_frontier_recovery(
            stage=stage,
            primary_context=primary_context,
            continuation_context=continuation_context,
            supplement_context=supplement_context,
            closure_context=closure_context,
            amendment_payload=amendment_payload,
            continuation_payload=continuation_payload,
            supplement_payload=supplement_payload,
            closure_payload=closure_payload,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
            publish=publish,
        )

    frontier_recovery_payload = _secure_payload(
        frontier_recovery_path,
        base=stage,
        label="source-repair frontier recovery",
    )
    frontier_recovery_context = _frontier_recovery_context_from_payload(
        stage=stage,
        primary_context=primary_context,
        continuation_context=continuation_context,
        supplement_context=supplement_context,
        closure_context=closure_context,
        amendment_payload=amendment_payload,
        continuation_payload=continuation_payload,
        supplement_payload=supplement_payload,
        closure_payload=closure_payload,
        frontier_recovery_payload=frontier_recovery_payload,
        requested_execution_identity=current_execution_identity,
    )
    contexts.append(frontier_recovery_context)
    if not os.path.lexists(sensor_acceleration_path):
        for context in contexts:
            if _context_current_matches(
                context,
                execution_identity=current_execution_identity,
                verified_review_authorization=current_verified_review_authorization,
                immutable_bindings=current_immutable_bindings,
            ):
                return context
        return _create_or_preview_sensor_acceleration(
            stage=stage,
            parent_context=frontier_recovery_context,
            amendment_payload=amendment_payload,
            continuation_payload=continuation_payload,
            supplement_payload=supplement_payload,
            closure_payload=closure_payload,
            frontier_recovery_payload=frontier_recovery_payload,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=(
                current_verified_review_authorization
            ),
            current_immutable_bindings=current_immutable_bindings,
            publish=publish,
        )

    if publish:
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration is already published"
        )
    context = load_stage6_source_repair_context(
        stage_root=stage,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=current_verified_review_authorization,
        current_immutable_bindings=current_immutable_bindings,
    )
    if context is None or context.sensor_acceleration_sha256 is None:
        raise Stage6SourceRepairError(
            "source-repair sensor acceleration did not remain durable"
        )
    return context


__all__ = [
    "FIRST_SENSOR_ACCELERATION_UPDATE",
    "FIRST_FRONTIER_RECOVERY_UPDATE",
    "FIRST_REPAIRED_UPDATE",
    "FIRST_CLOSURE_UPDATE",
    "FIRST_SUPPLEMENT_UPDATE",
    "FRONTIER_RECOVERY_NEXT_TRANSACTION_KEY",
    "LAST_CONTINUATION_UPDATE",
    "LAST_FRONTIER_RECOVERY_PARENT_UPDATE",
    "LAST_ORIGIN_UPDATE",
    "LAST_SENSOR_ACCELERATION_PARENT_UPDATE",
    "SENSOR_ACCELERATION_FAILED_RESOURCE_ATTEMPT",
    "SENSOR_ACCELERATION_NEXT_RESOURCE_ATTEMPT",
    "SENSOR_ACCELERATION_NEXT_SEGMENT_INDEX",
    "SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY",
    "SOURCE_REPAIR_AMENDMENT_NAME",
    "SOURCE_REPAIR_CONTINUATION_ID",
    "SOURCE_REPAIR_CONTINUATION_NAME",
    "SOURCE_REPAIR_CONTINUATION_SCHEMA",
    "SOURCE_REPAIR_CLOSURE_ID",
    "SOURCE_REPAIR_CLOSURE_NAME",
    "SOURCE_REPAIR_CLOSURE_SCHEMA",
    "SOURCE_REPAIR_FRONTIER_RECOVERY_ID",
    "SOURCE_REPAIR_FRONTIER_RECOVERY_NAME",
    "SOURCE_REPAIR_FRONTIER_RECOVERY_SCHEMA",
    "SOURCE_REPAIR_SCHEMA",
    "SOURCE_REPAIR_SENSOR_ACCELERATION_ID",
    "SOURCE_REPAIR_SENSOR_ACCELERATION_NAME",
    "SOURCE_REPAIR_SENSOR_ACCELERATION_SCHEMA",
    "SOURCE_REPAIR_SENSOR_ACCELERATION_SUMMARY_SCHEMA",
    "SOURCE_REPAIR_SUPPLEMENT_ID",
    "SOURCE_REPAIR_SUPPLEMENT_NAME",
    "SOURCE_REPAIR_SUPPLEMENT_SCHEMA",
    "Stage6SourceRepairContext",
    "Stage6SourceRepairError",
    "create_stage6_source_repair_amendment",
    "load_stage6_source_repair_context",
]
