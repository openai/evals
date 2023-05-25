from evals.elsuite.basic.match import Match


class TestMatch(Match):
    def __init__(self,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs, samples_jsonl="")

    def get_samples(self):
        return [
            {"input": [{"role": "system", "content": "Complete the phrase as concisely as possible."}, {"role": "user", "content": "Once upon a "}], "ideal": "time"},
            {"input": [{"role": "system", "content": "Complete the phrase as concisely as possible."}, {"role": "user", "content": "The first US president was "}], "ideal": "George Washington"},
            {"input": [{"role": "system", "content": "Complete the phrase as concisely as possible."}, {"role": "user", "content": "OpenAI was founded in 20"}], "ideal": "15"}
        ]
