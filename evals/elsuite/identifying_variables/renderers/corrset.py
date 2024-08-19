from typing import List, Set, Tuple

from evals.elsuite.identifying_variables.structs import Sample
from evals.elsuite.identifying_variables.renderers.base import RendererBase
import evals.elsuite.identifying_variables.graph_utils as graph_utils
import evals.elsuite.identifying_variables.renderers.templates as templates
from evals.elsuite.identifying_variables.constants import SPARSITY_FOR_UNOBS


class CorrSetRenderer(RendererBase):
    """
    Describes the correlation structure of variables
    """

    def determine_sample_type(self, sample: Sample) -> Tuple[str, List[Set[str]]]:
        """
        Determines the type of sample we have, returning the correlation sets in
        the process. Accounts for unobserved variables by removing them from
        the correlation sets.

        Returns:
            str: The type of causal graph we have, ignoring unobserved variables.
                Either
                    - "many_correl_sets": there are at least two correlation sets, at least
                          one of which has at least two variables.
                    - "single_correl_set": there is only one correlation set.
                    - "only_ind": there are at least two correlation sets, all of which
                        have exactly one variable.
            List[Set[str]]: The list of correlation sets. A correlation set is the
                set of observed variables in a tree from the causal graph
        """
        causal_graph = sample.causal_graph
        graph_trees = graph_utils.find_graph_trees(causal_graph)
        correl_sets = []
        unobserved_vars = set(
            var
            for var in sample.variable_metadata
            if sample.variable_metadata[var]["extra"]["sparsity_rate"]
            > SPARSITY_FOR_UNOBS
        )
        for tree in graph_trees:
            correl_set = set(tree)
            for var in tree:
                if var in unobserved_vars:
                    # correlations to unobserved variables are, well, unobserved
                    correl_set.remove(var)
            correl_sets.append(correl_set)
        # need to check for empty sets, since we removed unobserved variables
        correl_sets = [correl_set for correl_set in correl_sets if len(correl_set) > 0]
        if len(correl_sets) == 1:
            return "single_correl_set", correl_sets
        else:
            for correl_set in correl_sets:
                if len(correl_set) > 1:
                    # at least one set with more than one observed var
                    return "many_correl_sets", correl_sets
            # all sets have only one node
            return "only_ind", correl_sets

    def _get_hypd_unobserved_vars(self, sample: Sample) -> List[str]:
        vars_to_mention = []
        hypotheses = sample.hypotheses

        hypothesized_vars = set(
            var
            for var in hypotheses
            if hypotheses.in_degree(var) > 0 or hypotheses.out_degree(var) > 0
        )
        vars_to_mention = [
            var
            for var in hypothesized_vars
            if sample.variable_metadata[var]["extra"]["sparsity_rate"]
            > SPARSITY_FOR_UNOBS
        ]
        return vars_to_mention


class PureCorrSetRenderer(CorrSetRenderer):
    def render_obs(self, sample: Sample) -> str:
        _, observed_sets = self.determine_sample_type(sample)

        render_string = (
            "The following correlation sets were observed. Variables in the"
            " same correlation set are correlated with each other, but not with variables in"
            " other correlation sets."
        )
        render_string += "\n\n" + self._render_observed_sets(observed_sets)
        render_string += "\n\n" + self._render_unobserved_vars(sample)

        return render_string

    def _render_observed_sets(self, observed_sets: List[Set[str]]) -> str:
        """
        Renders the observed sets.
        """
        render_string = ""
        for idx, correl_set in enumerate(observed_sets, start=1):
            render_string += f"\nCorrelation set {idx}: {{{', '.join(correl_set)}}}."
        return render_string.strip()

    def _render_unobserved_vars(self, sample: Sample) -> str:
        """
        Renders the unobserved variables.
        """
        unobserved_variables = self._get_hypd_unobserved_vars(sample)
        if len(unobserved_variables) == 0:
            render_string = "There were no unobserved variables."
        else:
            render_string = f"Unobserved variables: [{', '.join(unobserved_variables)}]."
        return render_string.strip()


class LanguageCorrSetRenderer(CorrSetRenderer):
    """
    Describes the correlation structure of variables in natural language.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.type_to_renderer = {
            "many_correl_sets": self.render_many_sets,
            "single_correl_set": self.render_single_set,
            "only_ind": self.render_only_ind,
        }

    def render_obs(self, sample: Sample) -> str:
        """
        Describes the interactions between variables in the sample.

        The description looks like
        ```
        {opening statement}

        {description of the interactions}

        {optional mention of unobserved variables that were hypothesized about}
        ```

        The description of the interactions depends on the type of causal graph.
        """
        sample_type, observed_sets = self.determine_sample_type(sample)

        opening_statement = templates.OPENING_STATEMENT
        main_observation = self.type_to_renderer[sample_type](observed_sets)
        unobserved_variables = self.mention_unobserved_vars(sample)
        return "\n\n".join([opening_statement, main_observation, unobserved_variables])

    def render_many_sets(self, correl_sets: List[Set[str]]):
        """
        Renders a causal graph where we have at least two correlation
        sets, one of which has at least two variables.
        The description looks like:
        ```
        In general, there were cases where some variables changed in tandem with each
        other, while others did not.
        {example of two variables that changed in tandem}
        {interleaved mentions of remaining variables, specifying which other already
        mentioned variables they changed in tandem with, if any}
        ```
        """
        # Sort the sets by size, largest first
        correl_sets = sorted(correl_sets, key=lambda x: len(x), reverse=True)
        variables = [var for correl_set in correl_sets for var in correl_set]

        correl_set_idx_to_already_mentioned_vars = [set() for _ in correl_sets]
        var_to_correl_set_idx = {
            var: idx for idx, correl_set in enumerate(correl_sets) for var in correl_set
        }
        return_string = templates.MANY_CORREL_SETS_MAIN

        # hard-code mention first two variables, from first (largest) set
        current_set_idx = 0
        return_string += "\n" + templates.CORREL_VARS_EXAMPLE.format(
            optional_transition="For example, ",
            # the first set is guaranteed to have at least two variables
            var_1=variables[0],
            var_2=variables[1],
        )
        correl_set_idx_to_already_mentioned_vars[0].update([variables[0], variables[1]])

        # go through remaining variables, randomly
        variables = variables[2:]
        self.rng.shuffle(variables)

        for var in variables:
            correl_set_idx = var_to_correl_set_idx[var]
            if correl_set_idx == current_set_idx:
                transition_word = self.rng.choice(["Similarly", "Likewise"])
                transition_phrase = f"{transition_word}, "
            else:
                transition_phrase = ""
                current_set_idx = correl_set_idx

            mentioned_vars_from_set = correl_set_idx_to_already_mentioned_vars[
                correl_set_idx
            ]
            if len(mentioned_vars_from_set) == 0:  # first time mentioning this set
                mention_string = templates.IND_VARS_EXAMPLE.format(
                    optional_transition=transition_phrase,
                    var_1=var,
                    var_2="previously mentioned variables",
                )
            else:  # variables from this set have been mentioned
                mention_string = templates.CORREL_VARS_EXAMPLE.format(
                    optional_transition=transition_phrase,
                    var_1=var,
                    var_2=templates.list_to_nl_list(list(mentioned_vars_from_set)),
                )
            return_string += "\n" + mention_string.capitalize()
            # we have now mentioned this variable
            correl_set_idx_to_already_mentioned_vars[correl_set_idx].add(var)

        return return_string

    def render_single_set(self, correl_sets: List[Set[str]]) -> str:
        """
        Renders a causal graph where we have only one correlation set.
        By definition, this set has at least two variables.
        The description looks like:
        ```
        In general, all of the variables seemed to change in tandem with each other.
        For example, changes in {var_1} were observed to reflect changes in {var_2} and
        viceversa.
        {optional example of other pair}
        {optional concluding statement that this holds for all pairs}
        ```
        """
        correl_set = correl_sets[0]
        # we won't use more than 3 variables in the examples.
        exemplar_vars = list(correl_set)[:3]
        remaining_vars = correl_set - set(exemplar_vars)
        # always have at least 2 vars
        example_1 = templates.CORREL_VARS_EXAMPLE.format(
            optional_transition="",
            var_1=exemplar_vars[0],
            var_2=exemplar_vars[1],
        )
        example_2 = ""
        concluding_statement = ""
        if len(exemplar_vars) == 3:
            example_2 = templates.CORREL_VARS_EXAMPLE.format(
                optional_transition="Additionally, ",
                var_1=exemplar_vars[2],
                var_2=templates.list_to_nl_list(exemplar_vars[:2]),
            )
        if len(remaining_vars) > 0:
            concluding_statement = templates.SPECIFIC_CONCL_STATEMENT.format(
                already_mentioned=templates.list_to_nl_list(exemplar_vars),
                remaining_vars=templates.list_to_nl_list(list(remaining_vars)),
            )
        return templates.SINGLE_CORREL_SET_MAIN.format(
            example_1=example_1,
            optional_example_2=example_2,
            optional_concluding_statement=concluding_statement,
        )

    def render_only_ind(self, correl_sets: List[Set[str]]) -> str:
        """
        Describes a causal graph where we have at least two correlation
        sets, all of which have only one variable, i.e. each variable
        in the causal graph is independent of all other variables. The
        description looks like:
        ```
        In general, no discernible patterns were noticed between the variables.
        For example, changes in {var_1} were not observed to reflect any changes in
        {var_2}.
        {optional example of other pair}
        {optional concluding statement that this holds for all pairs}
        ```
        """
        variables = [var for correl_set in correl_sets for var in correl_set]
        num_vars = len(variables)  # equal to the number of sets
        # there's always at least 2 variables.
        example_1 = templates.IND_VARS_EXAMPLE.format(
            optional_transition="",
            var_1=variables[0],
            var_2=variables[1],
        )
        example_2 = ""
        concluding_statement = ""
        if num_vars > 2:
            example_2 = templates.IND_VARS_EXAMPLE.format(
                optional_transition="Similarly, ",
                var_1=variables[0],
                var_2=variables[2],
            )
            if num_vars > 3:
                concluding_statement = templates.SPECIFIC_CONCL_STATEMENT.format(
                    already_mentioned=templates.list_to_nl_list(variables[:3]),
                    remaining_vars=templates.list_to_nl_list(variables[3:]),
                )
            else:
                concluding_statement = templates.GENERIC_CONCL_STATEMENT

        return templates.ONLY_IND_MAIN.format(
            example_1=example_1,
            optional_example_2=example_2,
            optional_concluding_statement=concluding_statement,
        )

    def mention_unobserved_vars(self, sample: Sample) -> str:
        """
        Mentions any unobserved variables that also hypothesized about.
        """
        vars_to_mention = self._get_hypd_unobserved_vars(sample)

        n_vars_to_mention = len(vars_to_mention)
        if n_vars_to_mention == 0:
            return_string = ""
        else:
            be_plurality = {"singular": "is", "plural": "are"}
            be_string = be_plurality["plural" if n_vars_to_mention > 1 else "singular"]
            return_string = templates.UNOBS_BUT_HYP_VARS.format(
                unobs_but_hyp_vars=templates.list_to_nl_list(vars_to_mention),
                be_string=be_string,
            )
        return return_string


if __name__ == "__main__":
    import random
    import numpy as np

    list_of_lists = [
        [{"x_1004"}, {"x_1005", "x_1006", "x_1007", "x_1008", "x_1009"}],
        [{"x_1007", "x_1008", "x_1009"}, {"x_1010"}],
        [{"x_1011"}, {"x_1012", "x_1013"}, {"x_1014"}],  # 3 elements
        [{"x_1022"}, {"x_1023", "x_1024"}, {"x_1025", "x_1026"}],
        [{"x_1030"}, {"x_1031", "x_1032", "x_1033"}, {"x_1034"}, {"x_1035"}],
    ]

    np_rng = np.random.default_rng(0)
    renderer = PureCorrSetRenderer(random.Random(0), np_rng)

    from evals.elsuite.identifying_variables.scripts.gen_data import gen_samples
    import networkx as nx
    from pprint import pprint

    samples = gen_samples(10, None, np_rng)

    for sample in samples:
        print("causal graph", nx.to_dict_of_lists(sample.causal_graph))
        print("hypotheses", list(sample.hypotheses.edges))
        pprint(sample.variable_metadata)
        print(renderer.render_obs(sample))
        print("================")
