from evals.completion_fns.openai import OpenAIChatCompletionFn, OpenAICompletionFn


def test_openai_chat_completion_fn_promotes_extra_kwargs_to_extra_options() -> None:
    completion_fn = OpenAIChatCompletionFn(model="gpt-3.5-turbo", temperature=0.5, registry=object())

    assert completion_fn.extra_options == {"temperature": 0.5}


def test_openai_completion_fn_promotes_extra_kwargs_to_extra_options() -> None:
    completion_fn = OpenAICompletionFn(model="text-davinci-003", temperature=0.5, registry=object())

    assert completion_fn.extra_options == {"temperature": 0.5}