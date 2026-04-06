import argparse
import time

import torch
from torch import nn


class DemoModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def warmup(model: nn.Module, inputs: list[torch.Tensor], steps: int) -> None:
    with torch.inference_mode():
        for i in range(steps):
            model(inputs[i % len(inputs)])
    torch.cuda.synchronize()


def run_eager(
    model: nn.Module,
    inputs: list[torch.Tensor],
    steps: int,
) -> tuple[float, torch.Tensor]:
    output = torch.empty_like(inputs[0])
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(steps):
            output = model(inputs[i % len(inputs)])
    torch.cuda.synchronize()
    return time.perf_counter() - start, output


def capture_graph(
    model: nn.Module,
    sample_input: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
    """
    把model(static_input)这段GPU计算录下来，以后只改static_input里的数据，然后用graph.replay()反复重放
    """
    # 准备固定地址，固定shape的GPU内存
    static_input = torch.empty_like(sample_input)
    static_output = torch.empty_like(sample_input)
    # 创建一个CUDA Graph对象。它现在还是空的，后面要往里面“录制”一段GPU执行过程。
    graph = torch.cuda.CUDAGraph()

    with torch.inference_mode():
        # 预热，很多CUDA算子第一次执行时会做一些初始化
        # 比如分配内部workspace, 选择kernel，建立缓存
        static_input.copy_(sample_input)
        static_output.copy_(model(static_input))
        # 等GPU前面的活都真正做完
        torch.cuda.synchronize()
        # 真正录图的地方，意思不是现在立刻执行一遍，然后结束，而是在这个with块里发生的GPU操作，会被CUDA Graph捕获下来，记录成一段可重放的执行流程
        with torch.cuda.graph(graph):
            static_output.copy_(model(static_input))

    return graph, static_input, static_output


def run_graph(
    graph: torch.cuda.CUDAGraph,
    static_input: torch.Tensor,
    static_output: torch.Tensor,
    inputs: list[torch.Tensor],
    steps: int,
) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(steps):
            static_input.copy_(inputs[i % len(inputs)])
            graph.replay()
    torch.cuda.synchronize()
    return time.perf_counter() - start, static_output.clone()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eager CUDA execution with CUDA Graph replay.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--input-pool", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this demo.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = DemoModel(args.hidden_size).to(device).eval()
    inputs = [
        torch.randn(args.batch_size, args.hidden_size, device=device)
        for _ in range(args.input_pool)
    ]

    warmup(model, inputs, args.warmup_steps)
    eager_time, eager_output = run_eager(model, inputs, args.steps)

    graph, static_input, static_output = capture_graph(model, inputs[0])
    graph_time, graph_output = run_graph(graph, static_input, static_output, inputs, args.steps)

    max_diff = (eager_output - graph_output).abs().max().item()
    eager_ms = eager_time * 1000 / args.steps
    graph_ms = graph_time * 1000 / args.steps
    speedup = eager_time / graph_time

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Steps: {args.steps}")
    print(f"Eager total: {eager_time:.4f}s ({eager_ms:.4f} ms/step)")
    print(f"Graph total: {graph_time:.4f}s ({graph_ms:.4f} ms/step)")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Max output diff: {max_diff:.6f}")


if __name__ == "__main__":
    main()
