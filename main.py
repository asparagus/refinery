"""Try out the refinery module."""

from typing import Optional

import dspy

from refinery import Feedback, Input, Output, Predictor, Retrier



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

    def __init__(self):
        self.predictor = dspy.Predict(signature=QA)

    def forward(self, input: Question) -> Answer:
        result = self.predictor(**input.toDict())
        return Answer(answer=result.answer)


class QAFeedback(Feedback[QA]):

    def forward(self, input: Question, output: Answer) -> tuple[float, Optional[str]]:
        if " " in output.answer:
            return (0.0, "The answer is too long")
        return (1.0, None)


if __name__ == "__main__":
    lm = dspy.LM(model="claude-3-5-sonnet-20241022")
    retrier = Retrier(
        module=QAModule(),
        feedback_module=QAFeedback(),
        N=3,
        threshold=1.0,
    )
    with dspy.context(lm=lm):
        result = retrier.forward(
            input=Input[QA](question="What animal has 4 legs and a shell?")
        )
        print(result)
