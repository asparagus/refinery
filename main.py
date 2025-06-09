"""Try out the refinery module."""

from typing import Any

import dspy

from refinery import (
    Feedback,
    FeedbackModule,
    Input,
    Output,
    Predictor,
    Retrier,
    Validation,
)


class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Question(Input[QA]):
    def __init__(self, question: str):
        super().__init__()
        self.question = question


class Answer(Output[QA]):
    def __init__(self, answer: str):
        super().__init__()
        self.answer = answer


class QAModule(Predictor[QA]):
    def __init__(
        self, validation: Validation[QA] | None = None, callbacks: Any | None = None
    ):
        super().__init__(validation=validation, callbacks=callbacks)
        self.predictor = dspy.Predict(signature=QA)

    def forward(self, input: Question) -> Answer:
        result = self.predictor(**input.toDict())
        return Answer(answer=result.answer)


class QAFeedback(FeedbackModule[QA]):
    def forward(self, input: Question, output: Answer, _trace=None) -> Feedback:
        if count := output.answer.count(" "):
            return Feedback(
                evaluation=1.0 / (1.0 + count), feedback="The answer is too long"
            )
        return Feedback(evaluation=1.0, feedback=None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["eval", "opt"])
    parser.add_argument("--retry", action=argparse._StoreTrueAction, default=False)
    parser.add_argument("--load", help="Path to load the model from")
    parser.add_argument("--trace", action=argparse._StoreTrueAction, default=False)
    parser.add_argument("--validation", action=argparse._StoreTrueAction, default=False)
    parser.add_argument("--debug", action=argparse._StoreTrueAction, default=False)
    args = parser.parse_args()

    lm = dspy.LM(model="claude-3-5-haiku-20241022", cache=True)
    validation: Validation[QA] | None = None
    module = QAModule()
    feedback_module = QAFeedback()

    if args.debug:
        import litellm

        litellm._turn_on_debug()

    if args.validation:
        module.validation = Validation(
            feedback_module=feedback_module, validation_threshold=1.0
        )

    if args.retry:
        module = Retrier(
            module=module,
            feedback_module=feedback_module,
            N=3,
            threshold=1.0,
        )

    if args.load:
        module.load(args.load)

    trainset = [
        dspy.Example(
            input=Question("What land animal has a shell?"),
            output=Answer(answer="Tortoise"),
        ).with_inputs("input"),
        dspy.Example(
            input=Question("What animal is a man's best friend?"),
            output=Answer(answer="Dog"),
        ).with_inputs("input"),
    ]
    evaluation = dspy.Evaluate(
        devset=trainset, metric=feedback_module, provide_traceback=True
    )
    tp = dspy.MIPROv2(metric=feedback_module, auto="light", num_threads=1)
    with dspy.context(lm=lm):
        match args.mode:
            case "eval":
                result, outputs = evaluation(program=module, return_outputs=True)
                print(result)
                print(outputs)
            case "opt":
                optimized: dspy.Module = tp.compile(module, trainset=trainset)
                print(optimized)
                optimized.save(path="optimized.json")
        if args.trace:
            print("Trace")
            for i, el in enumerate(dspy.settings.trace):
                print(f"{i}:\t{el}")
