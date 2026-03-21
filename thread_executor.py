import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class AgentExecutionOptions:
    save_agent_mem: bool = False
    agent_mem_file: str = "agent_memories.json"
    run_mode: str = "eval"


@dataclass(frozen=True)
class AgentTask:
    iteration: int
    sample: dict[str, Any]
    options: AgentExecutionOptions


@dataclass(frozen=True)
class AgentTaskResult:
    iteration: int
    result_entry: dict[str, Any] | None
    error: str | None = None


def _write_ordered_results(results_by_iter: dict[int, dict[str, Any]], output_file: str) -> None:
    ordered_results = [results_by_iter[i] for i in sorted(results_by_iter)]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2, ensure_ascii=False)


def _normalize_result(result: AgentTaskResult | tuple[int, dict[str, Any] | None, str | None]) -> AgentTaskResult:
    if isinstance(result, AgentTaskResult):
        return result

    iter_idx, result_entry, error = result
    return AgentTaskResult(iteration=iter_idx, result_entry=result_entry, error=error)


def run_multithreaded_processing(
    dataset_iter: Iterator[dict[str, Any]],
    process_func: Callable[[AgentTask], AgentTaskResult | tuple[int, dict[str, Any] | None, str | None]],
    max_workers: int = 5,
    output_file: str = "test_results.json",
    save_agent_mem: bool = False,
    agent_mem_file: str = "agent_memories.json",
    run_mode: str = "eval",
) -> int:
    """
    使用多线程处理数据集样本。

    `thread_executor` 只负责调度和结果落盘，具体 agent 工作流由 `process_func`
    处理。每个 worker 接收一个 `AgentTask`，返回 `AgentTaskResult`。
    """
    results_by_iter: dict[int, dict[str, Any]] = {}
    futures = []
    iteration = 0
    options = AgentExecutionOptions(
        save_agent_mem=save_agent_mem,
        agent_mem_file=agent_mem_file,
        run_mode=run_mode,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            try:
                sample = next(dataset_iter)
            except StopIteration:
                break

            iteration += 1
            task = AgentTask(iteration=iteration, sample=sample, options=options)
            futures.append(executor.submit(process_func, task))
            print(f"Queued iteration {iteration} for processing...")

        for future in as_completed(futures):
            task_result = _normalize_result(future.result())

            if task_result.error:
                print(f"Iteration {task_result.iteration} ✗ Error encountered. Full traceback:")
                print(task_result.error)
                continue

            if task_result.result_entry is None:
                print(f"Iteration {task_result.iteration} ✗ Missing result entry.")
                continue

            results_by_iter[task_result.iteration] = task_result.result_entry
            _write_ordered_results(results_by_iter, output_file)
            print(f"Iteration {task_result.iteration} ✓ Completed")

    return iteration
