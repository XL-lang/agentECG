from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thread_executor import run_multithreaded_processing
from agent_runner import process_sample
from webapp_support import DEFAULT_DATASET_DIR, DEFAULT_SNAPSHOT_ROOT, ECGSampleStore, SnapshotPublisher, load_json_file


QUERY_SAMPLE_IDS = [
    "65087",
    "69458",
    "98228",
    "107808",
    "113310",
]
RESULT_FILE = "tmp/web_results.json"
AGENT_MEM_FILE = "tmp/web_agent_memories.json"


def _validate_result_file(expected_sample_ids: list[str]) -> None:
    result_path = Path(RESULT_FILE)
    if not result_path.exists():
        raise SystemExit(f"Result file was not written: {RESULT_FILE}")

    rows = load_json_file(result_path)
    if not isinstance(rows, list):
        raise SystemExit(f"Result file is invalid: {RESULT_FILE}")

    actual_sample_ids = {str(row.get("sample_id")) for row in rows if isinstance(row, dict)}
    missing = [sample_id for sample_id in expected_sample_ids if sample_id not in actual_sample_ids]
    if len(rows) != len(expected_sample_ids) or missing:
        print(f"Incomplete results detected in {RESULT_FILE}")
        print(f"Expected sample_ids: {expected_sample_ids}")
        print(f"Actual sample_ids: {sorted(actual_sample_ids)}")
        print(f"Missing sample_ids: {missing}")
        raise SystemExit(1)


def main() -> None:
    sample_ids = QUERY_SAMPLE_IDS
    print(f"Running offline eval for query sample_ids={sample_ids}")
    sample_store = ECGSampleStore(dataset_dir=DEFAULT_DATASET_DIR)
    samples = [sample_store.get_sample_by_id(sample_id) for sample_id in sample_ids]
    count = run_multithreaded_processing(
        dataset_iter=iter(samples),
        process_func=process_sample,
        max_workers=2,
        output_file=RESULT_FILE,
        save_agent_mem=True,
        agent_mem_file=AGENT_MEM_FILE,
        run_mode="eval",
    )
    print(f"Submitted/processed {count} samples")
    _validate_result_file(sample_ids)

    publisher = SnapshotPublisher(
        dataset_dir=DEFAULT_DATASET_DIR,
        snapshot_root=DEFAULT_SNAPSHOT_ROOT,
    )
    publish_result = publisher.publish(
        result_file=RESULT_FILE,
        agent_mem_file=AGENT_MEM_FILE,
    )
    print("Published snapshot:")
    print(publish_result)


if __name__ == "__main__":
    main()
