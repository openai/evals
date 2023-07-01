# -*- coding: utf-8 -*-
""" How it got done: change sys.argv to our setup
    remake ../anthropics_evals/persona/machiavellianism.jsonl to evals/registry/data/macia/macia.jsonl
    run /Users/admin/prog/evals/evals/cli/oaieval.py  ( env keys are set in IDE)
    use DataFrame.to_str() to show output file as table -- needs be tried.
"""

from scripts.generate_machiavellianism_samples_homegrown_in_brno import generate_machi_fast, write_list_to_jsonl
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
    dst_path = os.path.join(dirname, "./evals/registry/data/macia/macia_our_brno_generated_test.jsonl")
    # brno_gen_machi=DataFrame("".splitlines())

    write_list_to_jsonl(generate_machi_fast(), dst_path)

    print("Composed/formatted our new eval to "+dst_path)
    # os.system(f"code {dst_path}") # for inspection


    ## Running the generated eval:
    #Todo from sys.exec(..) #with same path?.?
    # os.system("echo 'shell sees key' $OPENAI_API_KEY") # seems ^it could work well
    from evals.cli.oaieval import main
    import sys
    out_path = "./output/machi_brno_03.jsonl"
    sys.argv.append(f"--record_path={out_path}")
    # main()


    ### quick way to export:
    # df = load_result_to_DataFrame("output/macia.dev.v1.out.jsonl")#(out_path)
    # df = load_result_to_DataFrame("output/machi_brno_01.jsonl")
    # print(df.to_string())
    # df.to_clipboard()

