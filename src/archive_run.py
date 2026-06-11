import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BASE_DIR / "runs"

CURATED_ARTIFACTS = [
    "README.md",
    "requirements.txt",
    "requirements-research.txt",
    "reports/adaptive_alpha_lab_report.md",
    "reports/model_card.md",
    "reports/compute_budget.md",
    "reports/related_work.md",
    "reports/literature_matrix.csv",
    "models/experiment_results.csv",
    "models/strategy_comparison.csv",
    "models/regime_benchmark_summary.csv",
    "models/regime_stability_summary.csv",
    "models/regime_quality_summary.csv",
    "models/regime_agreement_matrix.csv",
    "models/compute_profile.csv",
    "models/ablation_budget.csv",
    "models/compute_budget_summary.csv",
    "models/guided_encoder_summary.csv",
    "models/guided_encoder_loss.csv",
    "models/guided_encoder_comparison.csv",
    "models/guided_alpha_comparison.csv",
    "models/per_regime_stats.csv",
    "models/validation_audit.csv",
    "models/fold_audit.csv",
    "models/walkforward_experiment_results.csv",
    "models/walkforward_comparison.csv",
    "models/walkforward_regime_summary.csv",
    "models/robustness_results.csv",
    "models/robustness_summary.csv",
    "models/robustness_wins.csv",
    "models/robustness_stress_results.csv",
    "models/robustness_stress_summary.csv",
    "models/robustness_stress_wins.csv",
    "models/statistical_fold_metrics.csv",
    "models/statistical_method_summary.csv",
    "models/statistical_pairwise_tests.csv",
    "models/statistical_test_summary.csv",
    "models/statistical_multiple_testing.csv",
    "models/statistical_claims.csv",
    "models/statistical_sharpe_diagnostics.csv",
    "models/target_distribution.csv",
    "models/target_quality.csv",
    "models/equity_curve.png",
    "models/loss_curve.png",
    "models/phase4_dashboard.png",
    "models/regime_stability.png",
    "models/regime_quality_heatmap.png",
    "models/regime_agreement_heatmap.png",
    "models/compute_budget_plan.png",
    "models/guided_encoder_loss_curve.png",
    "models/guided_encoder_transition_hmm_guided_gmm.png",
    "models/guided_encoder_transition_hmm_guided_hmm.png",
    "models/regime_timeline.png",
    "models/robustness_heatmap.png",
    "models/robustness_stress_heatmap.png",
    "models/statistical_ic_confidence_intervals.png",
    "models/statistical_multiple_testing.png",
    "models/statistical_sharpe_diagnostics.png",
    "models/target_distribution.png",
    "models/transition_matrix_contrastive.png",
    "models/transition_matrix_contrastive_hmm.png",
    "models/transition_matrix_hmm.png",
    "models/transition_matrix_kmeans.png",
    "models/transition_matrix_vol_bucket.png",
    "models/umap_improved.png",
    "models/umap_regimes.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive curated research artifacts into a versioned runs/ snapshot."
    )
    parser.add_argument("--phase", default="phase15_baseline", help="Human-readable phase label.")
    parser.add_argument("--run-id", default="", help="Optional run id. Defaults to timestamp + phase.")
    parser.add_argument("--source-ref", default="HEAD", help="Git ref that produced the archived artifacts.")
    parser.add_argument("--notes", default="", help="Short notes stored in the manifest and run index.")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Overwrite an existing run directory with the same run id.",
    )
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=BASE_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def safe_phase(value: str) -> str:
    keep = []
    for char in value.strip().lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join("".join(keep).split("_")).strip("_") or "run"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_artifact(relative_path: str, run_dir: Path) -> dict | None:
    source = BASE_DIR / relative_path
    if not source.exists() or not source.is_file():
        return None

    target = run_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source_path": relative_path.replace("\\", "/"),
        "archived_path": str(target.relative_to(BASE_DIR)).replace("\\", "/"),
        "bytes": int(target.stat().st_size),
        "sha256": sha256_file(target),
    }


def write_artifact_manifest(run_dir: Path, artifacts: list[dict]) -> None:
    path = run_dir / "artifact_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "archived_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(artifacts)


def append_run_index(row: dict) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    index_path = RUNS_DIR / "run_index.csv"
    fieldnames = [
        "run_id",
        "created_at_utc",
        "phase",
        "source_ref",
        "source_commit",
        "git_branch",
        "dirty_worktree",
        "artifact_count",
        "missing_artifact_count",
        "notes",
    ]
    rows = []
    if index_path.exists():
        with index_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [existing for existing in rows if existing.get("run_id") != row["run_id"]]
    rows.append(row)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_phase(args.phase)}"
    run_dir = RUNS_DIR / run_id

    if run_dir.exists():
        if not args.allow_overwrite:
            raise SystemExit(f"Run directory already exists: {run_dir}. Pass --allow-overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    artifacts = []
    missing = []
    for relative_path in CURATED_ARTIFACTS:
        copied = copy_artifact(relative_path, run_dir)
        if copied is None:
            missing.append(relative_path)
        else:
            artifacts.append(copied)

    source_commit = run_git(["rev-parse", args.source_ref])
    branch = run_git(["branch", "--show-current"])
    status_short = run_git(["status", "--short"])
    dirty = bool(status_short)

    manifest = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "phase": args.phase,
        "notes": args.notes,
        "source_ref": args.source_ref,
        "source_commit": source_commit,
        "git_branch": branch,
        "git_tags_pointing_at_source": run_git(["tag", "--points-at", source_commit]).splitlines() if source_commit else [],
        "dirty_worktree_at_archive_time": dirty,
        "git_status_short_at_archive_time": status_short.splitlines() if status_short else [],
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "artifact_count": len(artifacts),
        "missing_artifacts": missing,
        "artifacts": artifacts,
    }

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    write_artifact_manifest(run_dir, artifacts)

    index_row = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "phase": args.phase,
        "source_ref": args.source_ref,
        "source_commit": source_commit,
        "git_branch": branch,
        "dirty_worktree": str(dirty),
        "artifact_count": str(len(artifacts)),
        "missing_artifact_count": str(len(missing)),
        "notes": args.notes,
    }
    append_run_index(index_row)

    latest_path = RUNS_DIR / "latest_run.json"
    with latest_path.open("w", encoding="utf-8") as handle:
        json.dump(index_row, handle, indent=2)
        handle.write("\n")

    print(f"Archived run: {run_id}")
    print(f"Artifacts copied: {len(artifacts)}")
    print(f"Missing optional artifacts: {len(missing)}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Run index: {RUNS_DIR / 'run_index.csv'}")


if __name__ == "__main__":
    main()
