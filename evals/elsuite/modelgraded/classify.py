"""
Generic eval that uses a prompt + classification.
"""
from collections import Counter
from random import Random
from typing import Any, Optional, Union

import evals
import evals.record
from evals.elsuite.modelgraded.classify_utils import classify, sample_and_concat_n_completions
from evals.elsuite.utils import PromptFn, scrub_formatting_from_prompt

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS



def text_to_docs(text):
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    print("Length of doc_chunks: ", len(doc_chunks))
    return doc_chunks


def process_text():
    text = test_sample['context']
    docs = text_to_docs(text)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def retrieve_context_from_db(query, db, k=3):
    documents = db.similarity_search(query=test_query, k=3)
    return ''.join([doc.page_content for doc in documents])


class ModelBasedClassify(evals.Eval):
    def __init__(
        self,
        modelgraded_spec: str,
        *args,
        modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None,
        dataset,
        sample_kwargs: Optional[dict[str, Any]] = None,
        eval_kwargs: Optional[dict[str, Any]] = None,
        multicomp_n: Union[int, str] = 1,
        eval_type: Optional[str] = None,
        metaeval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # treat last completion_fn as eval_completion_fn
        self.eval_completion_fn = self.completion_fns[-1]
        if len(self.completion_fns) > 1:
            self.completion_fns = self.completion_fns[:-1]
        n_models = len(self.completion_fns)
        self.sample_kwargs = {"max_tokens": 1024}
        self.sample_kwargs.update(sample_kwargs or {})
        self.eval_kwargs = {"max_tokens": 1024}
        self.eval_kwargs.update(eval_kwargs or {})
        self.metaeval = metaeval
        self.modelgraded_spec_args = modelgraded_spec_args or {}
        self.eval_type = eval_type
        if multicomp_n == "from_models":
            assert n_models > 1
            self.multicomp_n = n_models
        else:
            assert isinstance(multicomp_n, int)
            self.multicomp_n = multicomp_n
        if len(self.completion_fns) > 1:
            assert self.multicomp_n == n_models
        
        self.mg = self.registry.get_modelgraded_spec(modelgraded_spec)


    def eval_sample(self, test_sample: dict, rng: Random) -> None:
        """Evaluate a single sample.

        Recorded metrics are always: one of the self.choice_strings, or "__invalid__".
        """
        # process test_sample
        
        db = process_text(test_sample['context'])

        print("THE TEXT LENGTH ------------", len(text))
        for k in self.mg.input_outputs:
            test_sample[k] = scrub_formatting_from_prompt(test_sample[k])
            context = retrieve_context_from_db(test_sample[k], db)
            #print("Test sample ", k, " ", test_sample[k])
        # run policy completions
        completions = {}
        for k, v in self.mg.input_outputs.items():
            if v in test_sample:  # test_sample already has completion, skip.
                continue
            if self.multicomp_n > 1:
                completion = sample_and_concat_n_completions(
                    self.completion_fns,
                    prompt=test_sample[k],
                    template_i=self.mg.output_template,
                    sample_kwargs=self.sample_kwargs,
                    n=self.multicomp_n,
                )
            else:
                prompt = "\nUsing the provided context, Please answer a question at the end.Context:"+ "\n\n" + context + "\n\n"+"The question to answer:" + "\n\n" + test_sample[k] + "\n" 
                get_input_completion = PromptFn(
                    prompt, completion_fn=self.completion_fn, **self.sample_kwargs
                )
                completion, _ = get_input_completion()
                #print("Single competion", completion)
            completions[v] = completion
        #print("Completions ", completions)
        # run modelgraded eval
        metrics = {}
        choice, info = classify(
            mg=self.mg,
            completion_fn=self.eval_completion_fn,
            completion_kwargs=self.eval_kwargs,
            eval_type=self.eval_type,
            n=self.multicomp_n,
            format_kwargs={**completions, **test_sample, **self.modelgraded_spec_args},
        )
        metrics.update(dict(choice=choice, score=info["score"]))

        # run metaeval if requested
        if self.metaeval:
            assert "choice" in test_sample
            metrics["metascore"] = choice == test_sample["choice"]

        evals.record.record_metrics(**metrics)

        return choice

    def run(self, recorder):
        samples = self.get_samples()

        self.eval_all_samples(recorder, samples)
        record_metrics = {}

        all_sample_metrics = recorder.get_metrics()
        if not all_sample_metrics:
            return record_metrics

        # record the counts
        choices = [m["choice"] for m in all_sample_metrics]
        counts = dict(Counter(choices))
        record_metrics.update({f"counts/{k}": v for k, v in counts.items()})

        # record the scores
        scores = [m["score"] for m in all_sample_metrics if m["score"] is not None]
        if scores:
            record_metrics[f"score"] = sum(scores) / len(scores)
        metascores = [m["metascore"] for m in all_sample_metrics if "metascore" in m]
        if metascores:
            record_metrics[f"metascore"] = sum(metascores) / len(metascores)

        return record_metrics
