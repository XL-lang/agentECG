from data.EcgQaData import EcgQaDataset, QueryType, Special_query
from thread_executor import run_multithreaded_processing
from agent_runner import process_sample


SAMPLE_IDS = ["132622", "132623", "132624", "132625"]


def main() -> None:
    print(f"Starting skill_build background run for sample_ids={SAMPLE_IDS}")
    special_queries = [
        Special_query(QueryType.sample_id_query, sample_id)
        for sample_id in SAMPLE_IDS
    ]
    dataset = EcgQaDataset(
        "dataset/ecgqa_ptbxl/paraphrased/train",
        question_types=["single-query"],
        sample_limit=None,
        shuffle=False,
        special_queries=special_queries,
    )
    count = run_multithreaded_processing(
        dataset_iter=iter(dataset),
        process_func=process_sample,
        max_workers=1,
        output_file="tmp/run_four_query_skill_build_results_20260505.json",
        save_agent_mem=False,
        agent_mem_file="tmp/run_four_query_skill_build_agent_mem_20260505.json",
        run_mode="skill_build",
    )
    print(f"Submitted/processed {count} samples")


if __name__ == "__main__":
    main()
