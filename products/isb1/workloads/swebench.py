"""SWE-bench coding workload generator for ISB-1 benchmarks.

Produces long-context coding traces from SWE-bench_Verified instances
(real GitHub issue → code change tasks).  Each session groups instances
from the same repository so that the system prompt + repo context form a
shared prefix — stressing KV cache prefix reuse exactly the way
production coding assistants work.

Context length is controlled via *context_bucket* which determines how
much CodeSearchNet code is prepended as synthetic repository context:
  8k ≈ 2 files, 16k ≈ 5 files, 32k ≈ 12 files, 64k ≈ 25 files, 128k ≈ 50 files.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from workloads.base import Request, WorkloadGenerator, _new_request_id
from workloads.datasets import (
    build_repo_context,
    load_codesearchnet,
    load_swebench_verified,
)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a production software engineer. You have access to the full repository context below.
Your task is to diagnose issues and produce minimal, correct patches.
Preserve exact file paths, keep edits minimal, and explain your reasoning.

Repository: {repo}

{repo_context}"""

_TURN2_PROMPT = (
    "Now write the minimal patch that fixes this issue. "
    "Output a unified diff and explain each change."
)


class SWEBenchCodingGenerator(WorkloadGenerator):
    """Generate long-context coding traces from SWE-bench_Verified.

    Each instance becomes a 2-turn session: the issue text as turn 1,
    and a patch request as turn 2.  Instances from the same repository
    share a session_id so the prefix (system prompt + repo context) is
    reused in the KV cache.

    Args:
        seed: Random seed for reproducibility.
        context_bucket: Target context length bucket
            (``"8k"``, ``"16k"``, ``"32k"``, ``"64k"``, ``"128k"``).
        max_sessions: Maximum number of unique sessions (repos) to use.
    """

    def __init__(
        self,
        seed: int = 42,
        context_bucket: str = "32k",
        max_sessions: int = 50,
    ) -> None:
        super().__init__(seed=seed)
        self.context_bucket = context_bucket
        self.max_sessions = max_sessions

    def generate(self, num_requests: int) -> list[Request]:
        instances = load_swebench_verified()
        if not instances:
            raise RuntimeError(
                "SWE-bench_Verified returned no instances. "
                "Install: pip install 'isb1[datasets]' and ensure network access to HuggingFace Hub."
            )

        # Load CodeSearchNet for context padding
        csn = load_codesearchnet(languages=("python", "javascript"))
        all_functions = csn.get("python", []) + csn.get("javascript", [])

        # Group instances by repo for prefix reuse
        by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for inst in instances:
            repo = inst.get("repo", "unknown/unknown")
            by_repo[repo].append(inst)

        # Pick repos up to max_sessions
        repo_names = list(by_repo.keys())
        self.rng.shuffle(repo_names)
        selected_repos = repo_names[: self.max_sessions]

        # Build repo context once per repo (the shared prefix)
        repo_contexts: dict[str, str] = {}
        for repo in selected_repos:
            repo_contexts[repo] = build_repo_context(
                all_functions, self.context_bucket, self.rng
            )

        # Generate requests: 2 turns per instance (issue + patch request)
        requests: list[Request] = []
        instance_cycle = 0

        while len(requests) < num_requests:
            for repo in selected_repos:
                if len(requests) >= num_requests:
                    break
                repo_instances = by_repo[repo]
                inst = repo_instances[instance_cycle % len(repo_instances)]
                instance_id = inst.get("instance_id", f"swe-{instance_cycle}")
                problem = inst.get("problem_statement", "Fix the failing test.")
                session_id = f"swe-{repo.replace('/', '-')}-{instance_id[:8]}"

                system_content = _SYSTEM_PROMPT_TEMPLATE.format(
                    repo=repo, repo_context=repo_contexts[repo]
                )

                # Turn 1: Issue text
                requests.append(
                    Request(
                        request_id=_new_request_id(self.rng),
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": problem},
                        ],
                        expected_output_tokens=512,
                        session_id=session_id,
                        metadata={
                            "workload": "swebench",
                            "repo": repo,
                            "instance_id": instance_id,
                            "turn": 0,
                            "context_bucket": self.context_bucket,
                        },
                    )
                )
                if len(requests) >= num_requests:
                    break

                # Turn 2: Patch request (reuses prefix)
                requests.append(
                    Request(
                        request_id=_new_request_id(self.rng),
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": problem},
                            {
                                "role": "assistant",
                                "content": (
                                    "I've analyzed the issue. The root cause is in the "
                                    "request handling path where the session context is "
                                    "not properly propagated."
                                ),
                            },
                            {"role": "user", "content": _TURN2_PROMPT},
                        ],
                        expected_output_tokens=1024,
                        session_id=session_id,
                        metadata={
                            "workload": "swebench",
                            "repo": repo,
                            "instance_id": instance_id,
                            "turn": 1,
                            "context_bucket": self.context_bucket,
                        },
                    )
                )

            instance_cycle += 1

        return requests[:num_requests]
