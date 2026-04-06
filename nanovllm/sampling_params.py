from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        """
        不允许把temperature设成0或极其接近0，因为那基本就等价于贪心了
        """

        assert self.temperature > 1e-10, "greedy sampling is not permitted"
