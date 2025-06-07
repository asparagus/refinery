"""Module to try an alternative to dspy.Refine."""

import contextlib
from typing import Optional

import dspy
import litellm
from dspy.adapters.base import Adapter


litellm.print_verbose = True


class Output[T: dspy.Signature](dspy.Prediction):
    """Output type for a signature."""


class Input[T: dspy.Signature](dspy.Example):
    """Input type for a signature."""


class Predictor[T: dspy.Signature](dspy.Module):
    """A module that is capable of fulfilling a signature."""

    def forward(self, input: Input[T]) -> Output[T]:
        raise NotImplementedError()


class Feedback[T: dspy.Signature](dspy.Module):
    """A feedback module capable of determining the quality of a response and feedback to improve it."""

    def forward(self, input: Input[T], answer: Output[T]) -> tuple[float, str]:
        raise NotImplementedError()


class RetryAdapter(Adapter):
    """An adapter that adds a hint argument based on the received feedback."""

    def __init__(self, adapter: Adapter, feedback: str):
        super().__init__()
        self.adapter = adapter
        self.feedback = feedback

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        modified_signature = signature.append(
            "hint_", dspy.InputField(desc="A hint to the module from an earlier run")
        )
        return self.adapter(lm, lm_kwargs, modified_signature, demos, inputs | {"hint_": self.feedback})


class Retrier[T: dspy.Signature](dspy.Module):
    """A module that will retry the underlying module based on feedback."""

    def __init__(
        self,
        module: Predictor[T],
        feedback_module: Feedback[T],
        N: int,  # noqa: N803
        threshold: float,
    ):
        self.module = module
        self.feedback_module = feedback_module
        self.N = N  # noqa: N803
        self.threshold = threshold

    @classmethod
    def retry_context(cls, feedback: Optional[str]):
        """Context within which adapters pass in the feedback if available."""
        if not feedback:
            return contextlib.nullcontext()
        base_adapter = dspy.settings.adapter or dspy.ChatAdapter()
        return dspy.context(
            adapter=RetryAdapter(
                adapter=base_adapter,
                feedback=feedback,
            ),
        )
        
    def forward(self, input: Input[T]) -> Output[T]:
        """Retry the module until the output meets the threshold."""
        feedback: Optional[str] = None
        for _ in range(self.N):
            with self.retry_context(feedback=feedback):
                output = self.module.forward(input=input)
            result, feedback = self.feedback_module.forward(input=input, output=output)
            if result >= self.threshold:
                return output
        raise RuntimeError("Could not obtain proper result")
