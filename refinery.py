"""Module to try an alternative to dspy.Refine."""

import contextlib
from typing import Any, Optional, Self

import dspy
import litellm
from dspy.adapters.base import Adapter


litellm.print_verbose = True


class Output[T: dspy.Signature](dspy.Prediction):
    """Output type for a signature."""


class Input[T: dspy.Signature](dspy.Example):
    """Input type for a signature."""

    def inputs(self) -> Self:
        return self


class Feedback(dspy.Prediction):
    """Feedback for a given input / output pair."""

    def __init__(self, evaluation: float, feedback: Optional[str] = None):
        super().__init__()
        self.evaluation = evaluation
        self.feedback = feedback

    def __float__(self):
        return self.evaluation


class FeedbackModule[T: dspy.Signature](dspy.Module):
    """A feedback module capable of determining the quality of a response and feedback to improve it."""

    def __init__(self, callbacks: Any | None = None):
        super().__init__(callbacks=callbacks)

    def forward(self, input: Input[T], output: Output[T]) -> Feedback:
        raise NotImplementedError()

    def __call__(self, input: Input[T], output: Output[T], _trace = None) -> Feedback:
        trace = dspy.settings.trace
        feedback = self.forward(input=input, output=output)
        trace.append((self, {"input": input, "output": output}, feedback))
        return feedback


class Validation[T: dspy.Signature](dspy.Module):
    """A module used for validation of an output."""

    def __init__(
        self,
        feedback_module: FeedbackModule[T],
        validation_threshold: float,
        callbacks: Any | None = None,
    ):
        self.feedback_module = feedback_module
        self.validation_threshold = validation_threshold
        self.callbacks = callbacks

    def forward(self, input: Input[T], output: Output[T]) -> bool:
        feedback = self.feedback_module(input=input, output=output)
        return feedback.evaluation >= self.validation_threshold

    def __call__(self, input: Input[T], output: Output[T]) -> bool:
        return self.forward(input=input, output=output)


class Predictor[T: dspy.Signature](dspy.Module):
    """A module that is capable of fulfilling a signature."""

    def __init__(self, validation: Validation[T] | None = None, callbacks: Any | None = None):
        super().__init__(callbacks=callbacks)
        self.validation = validation

    def forward(self, input: Input[T]) -> Output[T]:
        raise NotImplementedError()

    def __call__(self, input: Input[T]) -> Output[T]:
        output = self.forward(input=input)
        if self.validation:
            assert self.validation(input=input, output=output), "Output failed to pass validation"
        return self.forward(input=input)


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
        feedback_module: FeedbackModule[T],
        N: int,  # noqa: N803
        threshold: float,
        callbacks: Any | None = None,
    ):
        super().__init__(callbacks=callbacks)
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
        feedback: Optional[Feedback] = None
        for _ in range(self.N):
            with self.retry_context(feedback=feedback.feedback if feedback else None):
                output = self.module(input=input)
            feedback = self.feedback_module(input=input, output=output)
            if feedback.evaluation >= self.threshold:
                return output
        raise RuntimeError("Could not obtain proper result")
