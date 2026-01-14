import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_multithreaded_processing(dataset_iter, process_func, max_workers=5, output_file="test_results.json"):
    """
    使用多线程处理数据集样本
    
    Args:
        dataset_iter: 数据集迭代器
        process_func: 处理函数，接收(iter_idx, sample)参数，返回(iter_idx, result_entry, error)
        max_workers: 最大工作线程数，默认为5
        output_file: 输出结果文件名，默认为"test_results.json"
    
    Returns:
        int: 处理的总迭代次数
    """
    results_by_iter = {}
    futures = []
    iteration = 0

    # Dispatch processing to worker threads while keeping data read and writes on the main thread
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            try:
                sample = next(dataset_iter)
            except StopIteration:
                break

            iteration += 1
            futures.append(executor.submit(process_func, iteration, sample))
            print(f"Queued iteration {iteration} for processing...")

        for future in as_completed(futures):
            iter_idx, result_entry, error = future.result()

            if error:
                print(f"Iteration {iter_idx} ✗ Error encountered. Full traceback:")
                print(error)
                continue

            results_by_iter[iter_idx] = result_entry
            ordered_results = [results_by_iter[i] for i in sorted(results_by_iter)]

            with open(output_file, "w") as f:
                json.dump(ordered_results, f, indent=2, ensure_ascii=False)

            print(f"Iteration {iter_idx} ✓ Completed")

    return iteration

