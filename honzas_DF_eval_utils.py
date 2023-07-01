# -*- coding: utf-8 -*-
""" How it got done: change sys.argv to our setup
    remake ../anthropics_evals/persona/machiavellianism.jsonl to evals/registry/data/macia/macia.jsonl
    run /Users/admin/prog/evals/evals/cli/oaieval.py  ( env keys are set in IDE)
    use DataFrame.to_str() to show output file as table -- needs be tried.
"""

# """# Setup
# Tato cast stahne evals repo a nainstaluje potrebne python balicky
# """
#
# !git clone https://github.com/openai/evals.git
# !cd evals && git lfs fetch --all && git lfs pull
# #!cat evals/evals/registry/data/logic/samples.jsonl
# !cd evals && pip install -e .

from jsonlines import jsonlines

from scripts.generate_machiavellianism_samples_homegrown_in_brno import generate_machi_fast
from scripts.view_result_as_DataFrame import load_result_to_DataFrame



if __name__ == "__main__":
    import os
    dirname = os.path.dirname(__file__)

    ## Reformating/migration of Anth. evals:
    # anthropic_eval_repo = "../anthropics_evals/"
    # src_filename = os.path.join(dirname, anthropic_eval_repo, "persona/machiavellianism.jsonl")
    # dst_path = os.path.join(dirname, "./evals/registry/data/macia/macia.jsonl")
    # reformat_ant_eval(src_filename,dst_path, max_lines=50)
    # print("Wrote reformatted eval to "+dst_path)

    ## Or reformatting our own funk:
    dst_path = os.path.join(dirname, "./evals/registry/data/macia/macia_our_brno_generated.jsonl")
    # brno_gen_machi=DataFrame("".splitlines())

    brno_gen_machi= generate_machi_fast()
    for line in brno_gen_machi:
        with jsonlines.open(dst_path, "w") as writer:
            # for line in brno_gen_machi:
            #     writer.write(line)
            writer.write_all(brno_gen_machi)

    print("Composed/formatted our new eval to "+dst_path)

    # ## Running the generated eval:
    # #Todo from sys.exec(..) #with same path?.?
    # from evals.cli.oaieval import main
    # import sys
    # out_path = "./output/machi_brno_01.jsonl"
    # sys.argv.append(f"--record_path={out_path}")
    # main()


    # quick way to export:
    # df = load_result_to_DataFrame("output/macia.dev.v1.out.jsonl")#(out_path)
    df = load_result_to_DataFrame("output/machi_brno_01.jsonl")
    print(df.to_string())
    df.to_clipboard()

