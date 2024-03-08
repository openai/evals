import copy
import json
import re
import string
from collections import Counter, defaultdict
from typing import Optional, Union, List, Any
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd

from evals import CompletionFn
from evals.prompt.base import (
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
    chat_prompt_to_text_prompt,
    is_chat_prompt,
)
from evals.record import record_match

synonyms = {
    'Hydrogen': 'H',
    'Helium': 'He',
    'Lithium': 'Li',
    'Beryllium': 'Be',
    'Boron': 'B',
    'Carbon': 'C',
    'Nitrogen': 'N',
    'Oxygen': 'O',
    'Fluorine': 'F',
    'Neon': 'Ne',
    'Sodium': 'Na',
    'Magnesium': 'Mg',
    'Aluminium': 'Al',
    'Aluminium(aluminum)': 'Al',
    'Silicon': 'Si',
    'Phosphorus': 'P',
    'Sulfur': 'S',
    'Chlorine': 'Cl',
    'Argon': 'Ar',
    'Potassium': 'K',
    'Calcium': 'Ca',
    'Scandium': 'Sc',
    'Titanium': 'Ti',
    'Vanadium': 'V',
    'Chromium': 'Cr',
    'Manganese': 'Mn',
    'Iron': 'Fe',
    'Cobalt': 'Co',
    'Nickel': 'Ni',
    'Copper': 'Cu',
    'Zinc': 'Zn',
    'Gallium': 'Ga',
    'Germanium': 'Ge',
    'Arsenic': 'As',
    'Selenium': 'Se',
    'Bromine': 'Br',
    'Krypton': 'Kr',
    'Rubidium': 'Rb',
    'Strontium': 'Sr',
    'Yttrium': 'Y',
    'Zirconium': 'Zr',
    'Niobium': 'Nb',
    'Molybdenum': 'Mo',
    'Technetium': 'Tc',
    'Ruthenium': 'Ru',
    'Rhodium': 'Rh',
    'Palladium': 'Pd',
    'Silver': 'Ag',
    'Cadmium': 'Cd',
    'Indium': 'In',
    'Tin': 'Sn',
    'Antimony': 'Sb',
    'Tellurium': 'Te',
    'Iodine': 'I',
    'Xenon': 'Xe',
    'Cesium': 'Cs',
    'Barium': 'Ba',
    'Lanthanum': 'La',
    'Cerium': 'Ce',
    'Praseodymium': 'Pr',
    'Neodymium': 'Nd',
    'Promethium': 'Pm',
    'Samarium': 'Sm',
    'Europium': 'Eu',
    'Gadolinium': 'Gd',
    'Terbium': 'Tb',
    'Dysprosium': 'Dy',
    'Holmium': 'Ho',
    'Erbium': 'Er',
    'Thulium': 'Tm',
    'Ytterbium': 'Yb',
    'Lutetium': 'Lu',
    'Hafnium': 'Hf',
    'Tantalum': 'Ta',
    'Tungsten': 'W',
    'Rhenium': 'Re',
    'Osmium': 'Os',
    'Iridium': 'Ir',
    'Platinum': 'Pt',
    'Gold': 'Au',
    'Mercury': 'Hg',
    'Thallium': 'Tl',
    'Lead': 'Pb',
    'Bismuth': 'Bi',
    'Polonium': 'Po',
    'Astatine': 'At',
    'Radon': 'Rn',
    'Francium': 'Fr',
    'Radium': 'Ra',
    'Actinium': 'Ac',
    'Thorium': 'Th',
    'Protactinium': 'Pa',
    'Uranium': 'U',
    'Neptunium': 'Np',
    'Plutonium': 'Pu',
    'Americium': 'Am',
    'Curium': 'Cm',
    'Berkelium': 'Bk',
    'Californium': 'Cf',
    'Einsteinium': 'Es',
    'Fermium': 'Fm'
}


def get_answer(text, answer_prompt, ignore_case=False):
    if ignore_case:
        idx = text.lower().rfind(answer_prompt.lower())
    else:
        idx = text.rfind(answer_prompt)

    if idx == -1:
        return None
    return text[idx: idx + len(answer_prompt)]


def get_consensus(answers):
    counts = defaultdict(int)
    for answer in answers:
        counts[answer] += 1
    counts[None] = 0
    return max(counts, key=counts.get)


def compare_molecule(smi1, smi2) -> bool:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    else:
        return Chem.MolToSmiles(Chem.RemoveHs(mol1)) == Chem.MolToSmiles(Chem.RemoveHs(mol2))
    # return False
    # fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    # fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    # return DataStructs.TanimotoSimilarity(fp1, fp2)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


def fuzzy_compare_name(a: str, b: str, metric="EditDistance", compare_value=False) -> Union[bool, float]:
    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    a = a.strip()
    b = b.strip()

    if a == "" or b == "" and not a+b == "":
        return False
    if is_float(a) and is_float(b):
        return np.allclose(float(a), float(b), equal_nan=True, atol=1e-2, rtol=1e-2)

    if ((a.lower().startswith(b.lower()) or a.lower().endswith(b.lower())) or
        (b.lower().startswith(a.lower()) or b.lower().endswith(a.lower()))):
        return True
    else:
        if metric == "EditDistance":
            import Levenshtein
            return 1 - Levenshtein.distance(a.lower(), b.lower()) / (len(a) + len(b))
        elif metric == "Word2Vec":
            pass


def fuzzy_compare_value(a: str, b: str, metric="EditDistance") -> Union[bool, float]:
    """
    Compare two strings with fuzzy matching.
    """

    def standardize_unit(s: str) -> str:
        """
        Standardize a (affinity) string to common units.
        """
        mark = "" if re.search(r"[><=]", s) is None else re.search(r"[><=]", s).group()
        unit = s.rstrip()[-2:]
        number = float(re.search(r"[\+\-]*[0-9.]+", s).group())

        if unit in ["µM", "uM"]:
            unit = "nM"
            number *= 1000
        elif unit in ["mM", "mm"]:
            unit = "nM"
            number *= 1000000

        if mark == "=":
            mark = ""
        return f"{mark}{number:.1f} {unit}"

    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    unit_str = ["nM", "uM", "µM", "mM", "%", " %", "wt.%", "at.%", "at%", "wt%"]
    nan_str = ["n/a", "nan", "na", "n.a.", "nd", "not determined", "not tested", "inactive"]
    a = a.strip()
    b = b.strip()
    if is_float(a) and is_float(b):
        return np.allclose(float(a), float(b), equal_nan=True, atol=1e-2, rtol=1e-2)
    elif fuzzy_normalize_value(a) == "bal" or fuzzy_normalize_value(b) == "bal":
        return True
    elif fuzzy_normalize_value(a) == fuzzy_normalize_value(b):
        return True
    elif ((a[-2:] in unit_str or a[-1] in unit_str or a.split()[-1] in unit_str) and
          (b[-2:] in unit_str or b[-1] in unit_str or b.split()[-1] in unit_str)):
        a = standardize_unit(a)
        b = standardize_unit(b)
        return a == b
    elif a.lower() in nan_str and b.lower() in nan_str:
        return True
    if ((a.lower().startswith(b.lower()) or a.lower().endswith(b.lower())) or
        (b.lower().startswith(a.lower()) or b.lower().endswith(a.lower()))):
        return True
    else:
        if metric == "EditDistance":
            import Levenshtein
            return 1 - Levenshtein.distance(a.lower(), b.lower()) / (len(a) + len(b))
        elif metric == "Word2Vec":
            pass


def fuzzy_normalize_name(s):
    if s.startswith("Unnamed"):
        return ""
    else:
        """ 标准化字符串 """
        # 定义需要移除的单位和符号
        units = ["µM", "µg/mL", "nM", "%", "wt.%", "at.%", "at%", "wt%"]
        for unit in units:
            s = s.replace(unit, "")

        # 定义特定关键字
        keywords = ["pIC50", "IC50", "EC50", "TC50", "GI50", "Ki", "Kd", "Kb", "pKb"]

        # 移除非字母数字的字符，除了空格
        s = re.sub(r'[^\w\s.\-\(\)]', '', s)
        if s in synonyms:
            s = synonyms[s]

        # 分割字符串为单词列表
        words = s.split()

        # 将关键字移到末尾
        reordered_words = [word for word in words if word not in keywords]
        keywords_in_string = [word for word in words if word in keywords]
        reordered_words.extend(keywords_in_string)
        # 重新组合为字符串
        return ' '.join(reordered_words)


def fuzzy_normalize_value(vi):
    try:
        vi = str(vi).lower()

        if "bal" in vi or "remainder" in vi or "bas" in vi:
            vi = "bal"
            return "bal"

        if "nan" in vi or "/" == vi or "n/a" in vi or "na" in vi or vi == "":
            vi = "0"

        vi = vi.replace("~", "-")

        pattern = r"\d+(?:\.\d+)?"
        matches = re.findall(pattern, vi)
        if len(matches) == 2:
            vi = f"{matches[0]}-{matches[1]}"
        elif len(matches) == 1:
            vi = matches[0]

        if "<" in vi:
            vi = vi.replace("<", "")
        if ">" in vi:
            vi = vi.replace(">", "")

        try:
            vi = float(vi)
            vi = round(vi, 3)
        except:
            # print(vi)
            pass
    except:
        pass

    return vi


def tableMatching(df_ref, df_prompt, index='Compound', compare_fields=[], record=True, file_name=None):
    from munkres import Munkres
    assert len(df_ref) > 0, "Prompt table is empty."

    if df_prompt is None or len(df_prompt) == 0:
        return {"recall_field": 0.0, "recall_index": 0.0, "recall_value": 0.0, "recall_value_strict": 0.0,
                "accuracy_value": 0.0, "accuracy_value_strict": 0.0, "recall_SMILES": 0.0}
    metrics = {}
    index_names = ["Compound", "Name", "SMILES", "Nickname", "Substrate"]

    df_prompt[index] = df_prompt[index].astype(str)
    df_ref[index] = df_ref[index].astype(str)
    df_ref = df_ref.set_index(index)
    df_prompt = df_prompt.set_index(index)

    def match_indices(ind0, ind1, threshold=0.9) -> dict:
        """
        Match the indices of two dataframes.
        """
        renames = {}
        name2query = lambda name: name if type(name) != tuple else name[0] if len(name) == 1 or name[1] == "" else name[1]
        similarities = np.array(np.ones([len(ind0) + 15, len(ind1) + 15]), dtype=np.float64)
        querys0 = [name2query(name) for name in ind0]
        querys1 = [name2query(name) for name in ind1]
        for i, name_i in enumerate(ind0):
            query_i = name2query(name_i)
            for j, name_j in enumerate(ind1):
                query_j = name2query(name_j)
                if fuzzy_normalize_name(query_i) == "" or fuzzy_normalize_name(query_j) == "":
                    similarities[i, j] = 0
                result = fuzzy_compare_name(fuzzy_normalize_name(query_i), fuzzy_normalize_name(query_j))
                if type(result) == bool:
                    similarities[i, j] = 1 if result else 0
                elif type(result) == float:
                    similarities[i, j] = result

        for k in range(15):
            for i in range(len(ind0)):
                similarities[i][len(ind1) + k] = threshold
            for j in range(len(ind1)):
                similarities[len(ind0) + k][j] = threshold
        dists = 1 - similarities
        # print(pd.DataFrame(dists, index=querys0 + ["v"] * 15, columns=querys1 + ["v"] * 15))

        # Kuhn-Munkres algorithm for useful solving the rectangular Assignment Problem
        mu = Munkres()
        indexes = mu.compute(dists.tolist())

        # 根据最优匹配下标输出映射
        for i, j in indexes:
            if (i < len(ind0)) and (j < len(ind1)):
                renames[name2query(ind1[j])] = name2query(ind0[i])
        return renames

    renames = match_indices(compare_fields, df_prompt.columns)
    renames = {key: value for key, value in renames.items() if key not in index_names}
    if len(renames) > 0:
        print("Find similar fields between answer and correct:", renames)
        df_prompt.rename(columns=renames, inplace=True)

    renames = match_indices(df_ref.index, df_prompt.index)
    renames = {key: value for key, value in renames.items() if key not in index_names}
    if len(renames) > 0:
        print("Find similar indices between answer and correct:", renames)
        df_prompt.rename(index=renames, inplace=True)

    compare_fields_ = [col for col in compare_fields if
                       col not in [index] + ([index[0]] if type(index) == tuple else [])]
    metrics["recall_field"] = max(
        len([item for item in compare_fields_ if item in df_prompt.columns]) / len(compare_fields_), 1.0)
    metrics["recall_index"] = max(len([item for item in df_ref.index if item in df_prompt.index]) / df_ref.shape[0], 1.0)

    if record:
        for col in compare_fields_:
            record_match(
                correct=col in df_prompt.columns,
                expected=col,
                picked=str(list(df_prompt.columns)),
                file_name=file_name,
                jobtype="match_field"
            )
        for ind in df_ref.index:
            record_match(
                correct=ind in df_prompt.index,
                expected=ind,
                picked=str(list(df_prompt.index)),
                file_name=file_name,
                jobtype="match_index"
            )

    match_score, total_match_score, smiles_match_score = 0.0, 0.0, 0.0
    N, M = len(df_ref.index), len(compare_fields_)
    for idx in df_ref.index:
        _total_matching = 1.0
        for col in compare_fields_:
            gtval = df_ref.loc[idx, col]
            gt = str(gtval.iloc[0]) if type(gtval) == pd.Series else str(gtval)
            try:
                pval = df_prompt.loc[idx, col]
                p = str(pval.iloc[0]) if type(pval) == pd.Series else str(pval)
            except:
                p = 'not found'

            _is_matching = fuzzy_compare_name(gt, p, compare_value=True) if col != "SMILES" else compare_molecule(gt, p)
            if col == "SMILES":
                smiles_match_score += float(_is_matching)
            if record:
                record_match(
                    correct=_is_matching > 0,
                    expected=gt,
                    picked=p,
                    file_name=file_name,
                    jobtype="match_value" if col != "SMILES" else "match_SMILES"
                )
            _total_matching *= float(_is_matching)
            match_score += float(_is_matching) / M

        total_match_score += _total_matching
        _total_matching = 1.0

    metrics = {
        **metrics,
        "recall_value": match_score / N,
        "recall_value_strict": total_match_score / N,
        "accuracy_value": match_score / N * metrics["recall_index"],
        "accuracy_value_strict": total_match_score / N * metrics["recall_index"],
    }
    if "SMILES" in compare_fields_:
        metrics["recall_SMILES"] = smiles_match_score / N
    return metrics


def tableMatchingStrict(df_ref, df_prompt, idx_col='Nickname'):
    df_ref = df_ref.set_index(idx_col)
    if len(df_prompt) == 0:
        return 0.0, 0.0
    df_prompt = df_prompt.set_index(idx_col)
    ref_columns = [col for col in df_ref.columns if col not in [idx_col]]
    idx_list = df_ref.index.values.tolist()
    prompt_idx_list = df_prompt.index.values.tolist()
    N, M = len(idx_list), len(ref_columns)
    match_score, total_match_score = 0.0, 0.0
    for idx in idx_list:
        _total_matching = 1.0
        for col in ref_columns:
            gt = df_ref.loc[idx, col]
            try:
                pd = df_prompt.loc[idx, col]
            except:
                pd = 'not found'
            _is_matching = fuzzy_compare(gt, pd)
            _total_matching *= float(_is_matching)
            match_score += float(_is_matching) / M
        total_match_score += _total_matching
        _total_matching = 1.0
    recall = max(len([item for item in prompt_idx_list if item in idx_list]) / len(idx_list), 1.0)
    print(f'Recall:{recall}, Acc: {match_score / N * recall}, Strict Acc: {total_match_score / N * recall}')
    return match_score / N * recall, total_match_score / N * recall


def ReactionDictMatching(dict_ref, dict_prompt, content: str = "inputs"):
    from google.protobuf import json_format
    from ord_schema.proto import reaction_pb2
    from ord_diff.schema import MDict, MDictDiff, MDictListDiff, MessageType
    from ord_diff.report import report_diff

    mdict_empty = json_format.Parse(json.dumps({}), reaction_pb2.Reaction())
    mdict_ref = json_format.Parse(json.dumps(dict_ref), reaction_pb2.Reaction())
    if content == "inputs":
        # mdict_ref = mdict_ref.inputs[0]
        mdict_prompt = json_format.Parse(json.dumps({"inputs": dict_prompt}), reaction_pb2.Reaction())
    else:
        mdict_prompt = json_format.Parse(json.dumps(dict_prompt), reaction_pb2.Reaction())

    diff = MDictDiff.from_message_pair(mdict_ref, mdict_prompt, message_type=MessageType.REACTION)
    df = report_diff(diff, message_type=MessageType.REACTION)

    diff_empty = MDictDiff.from_message_pair(mdict_ref, mdict_empty, message_type=MessageType.REACTION)
    df_empty = report_diff(diff_empty, message_type=MessageType.REACTION)
    print("############# Output:", mdict_prompt)

    accuracy = 1.0 - df.shape[0] / df_empty.shape[0]
    return accuracy, df


def ReactionDictMatchingSimple(dict_ref, dict_prompt, content: str = "inputs"):
    """
    Calculates the ratio of different leaves in two nested dictionaries using the deepdiff library.

    Parameters:
    - dict1: First dictionary to compare.
    - dict2: Second dictionary to compare.

    Returns:
    - Ratio of different leaves.
    """
    from deepdiff import DeepDiff
    # Compare the two dictionaries
    if content == "inputs":
        dict_ref = dict_ref["inputs"]
    diff = DeepDiff(dict_ref, dict_prompt, ignore_order=True, report_repetition=True)

    # Extract the count of different leaves
    # The 'values_changed', 'type_changes', 'dictionary_item_added', and 'dictionary_item_removed'
    # can be considered as indicators of different leaves
    diff_keys = ['values_changed',
                 'type_changes',
                 # 'dictionary_item_added',
                 'dictionary_item_removed']
    total_diff_leaves = sum(len(diff.get(key, {})) for key in diff_keys)

    # Count total leaves in both dictionaries (assuming all values are leaves)
    def count_leaves(d, count=0):
        for v in d.values():
            if isinstance(v, dict):
                count = count_leaves(v, count)
            else:
                count += 1
        return count

    total_leaves_dict1 = count_leaves(dict_ref)

    # Calculate the ratio of different leaves to total leaves
    if total_leaves_dict1 == 0:  # Prevent division by zero
        return 0
    ratio = total_diff_leaves / total_leaves_dict1

    return 1.0 - ratio, diff


def get_scores_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?(\d)/5"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: int(v) for k, v in dict(matches).items()}


def get_yesno_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?([yn])"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: v for k, v in dict(matches).items()}


def get_letter_from_data(data: str) -> str:
    last_y = (data.rfind("y"), "y")
    last_n = (data.rfind("n"), "n")
    char = max(last_y, last_n)[1]
    return char


# 定义使用Word2Vec计算词间相似度的metric
# sentence -> vector
def sentence_to_vec(sentence: str, model):
    words = sentence.split()
    word_vectors = []
    for word in words:
        if word in model:
            word_vectors.append(model[word])

    if not word_vectors:
        return np.zeros(model.vector_size)

    word_vectors = np.array(word_vectors)
    sentence_vector = word_vectors.mean(axis=0)
    return sentence_vector


def cosine_similarity(model, sentence1: str, sentence2: str):
    try:
        vec1 = sentence_to_vec(sentence1, model)
        vec2 = sentence_to_vec(sentence2, model)
        return 1 - cosine(vec1, vec2)
    except KeyError as e:
        return 0  # Return 0 similarity if word not found


def same_entities(model, word1: Union[list, str], word2: str, threshold: float = 0.9) -> bool:
    if isinstance(word1, list):
        for w in word1:
            if (cosine_similarity(model, w, word2) - threshold) > 1e-8:
                return True
        return False
    if (cosine_similarity(model, word1, word2) - threshold) > 1e-8:
        return True
    else:
        return False


def cosine_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2
    # if same_entities(s1,s2) == True:
    #     return s1
    # else:
    #     return
    return s1 in s2 or s2 in s1


# 线性查找算法（暴力穷举）：匹配两个list
def LinearSearch(item: List[Any], lst: List[List[Any]]) -> bool:
    # 检查lst中的每个元素的类型，如果是字符串，则尝试将其转换为列表
    lst = [eval(element) if isinstance(element, str) else element for element in lst]

    index = 0
    found = False
    while index < len(lst) and not found:
        # 只有当[compound, disease]两个都匹配上才算是同一条关系
        if same_entities(lst[index][0], item[0], 0.7) and same_entities(lst[index][1], item[1], 0.7):
            found = True
        else:
            index += 1
    return found


def list_match_ratio(s1: List[List[Any]], s2: List[List[Any]]) -> float:
    # s1 = normalize(s1)
    # s2 = normalize(s2)
    count = 0
    # 在正确答案（ideal）中遍历，以检查
    for item in s2:
        if LinearSearch(item, s1):
            count += 1
    return count / len(s2) if s2 else 0


def f1_score(prediction: str, answers: list[str]) -> float:
    def _f1_score(prediction: str, ground_truth: str):
        prediction_tokens = normalize(prediction).split()
        # 检查 ground_truth 是否是列表
        if not isinstance(ground_truth, list):
            ground_truth_tokens = normalize(ground_truth).split()
        else:
            ground_truth_tokens = ground_truth
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    return max([_f1_score(prediction, answer) for answer in answers])


def same_triplets(model, preds: List[List[Any]], true: List[Any]) -> bool:
    match = False
    for pred in preds:
        if (same_entities(model, pred[0], true[0], 0.9) and
                same_entities(model, pred[1], true[1],0.9) and
                same_entities(model, pred[2], true[2],0.9)):
            match = True
            break
    return match


def pick_same_triplets_in_pred(model, preds: List[List[Any]], true: List[Any]):
    matches = []
    for pred in preds:
        if same_entities(model, pred[0], true[0], 0.9) and same_entities(model, pred[1], true[1],
                                                                         0.9) and same_entities(model, pred[2], true[2],
                                                                                                0.9):
            matches.append(pred)
    return matches


def same_turples(model, preds: List[List[Any]], true: List[Any]) -> bool:
    match = False
    for pred in preds:
        if same_entities(model, pred[0], true[0], 0.9) and same_entities(model, pred[1], true[1], 0.9):
            match = True
            break
    return match


def pick_same_turples_in_pred(model, preds: List[List[Any]], true: List[Any]):
    matches = []
    for pred in preds:
        if same_entities(model, pred[0], true[0], 0.9) and same_entities(model, pred[1], true[1], 0.9):
            matches.append(pred)
    return matches


def entity_match(model, preds: List[Any], true: str) -> bool:
    match = False
    for pred in preds:
        if same_entities(model, pred, true, 0.9):
            match = True
    return match

def pick_most_similar_entity_in_pred(model, preds: List[Any], true: str):
    scores = [cosine_similarity(model, pred, true) for pred in preds]
    i = scores.index(max(scores))  # 获取最大数对应的下标
    return preds[i]

#查找compare这个二元组是否在bases这个二元组构成的list中出现，如果出现，则返回True
def match_turple(model, compare:List[Any],bases:List[List[Any]]) -> bool:
    found_match = False
    for base in bases:
        if compare[0] == base[0] and compare[1] == base[1] :
            found_match = True
            break
        elif same_entities(model, compare[0], base[0], 0.95) and same_entities(model, compare[1], base[1], 0.95):
            found_match = True
            break
        else:
            continue
    return found_match

#查找compare这个三元组是否在bases这个三元组构成的list中出现，如果出现，则返回True
def match_triplet(model, compare:List[Any],bases:List[List[Any]]) -> bool:
    found_match = False
    for base in bases:
        if compare[0] == base[0] and compare[1] == base[1] and compare[2] == base[2] :
            found_match = True
            break
        elif same_entities(model, compare[0], base[0], 0.95) and same_entities(model, compare[1], base[1], 0.95) and same_entities(model, compare[2], base[2], 0.95):
            found_match = True
            break
        else:
            continue
    return found_match


# 定义用于实体识别的f1-score
def cos_f1_score(model, prediction: List[str], answers: List[str]) -> float:
    """
    计算基于余弦相似度的 F1 分数。

    参数:
    prediction (List[str]): 预测的实体列表。
    answers (List[str]): 真实的实体列表。

    返回:
    float: F1 分数。
    """
    try:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        matched_ground_truth_tokens = set()
        similarity_threshold = 0.9  # 这里暂时把相同词义的判定阈值写死
        for pred_entity in prediction:
            found_match = False
            for idx, true_entity in enumerate(answers):
                similarity = cosine_similarity(model, pred_entity, true_entity)
                if similarity > similarity_threshold:
                    found_match = True
                    if idx not in matched_ground_truth_tokens:
                        true_positives += 1
                        matched_ground_truth_tokens.add(idx)
                    break
            if not found_match:
                false_positives += 1
        false_negatives = len(answers) - len(matched_ground_truth_tokens)

        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    except:
        return 0.0


def macro_f1_score_2(model, prediction: List[List[Any]], answers: List[List[Any]]) -> float:
    """
    计算用于二元组关系抽取的宏平均 F1 分数。

    参数:
    prediction (List[List[Any]]): 预测的二元组列表。
    answers (List[List[Any]]): 真实的二元组列表。

    返回:
    float: F1 分数。
    """
    try:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred_entity in prediction:
            if match_turple(model, pred_entity, answers):
                true_positives += 1
            else:
                false_positives += 1
        for true_entity in answers:
            if match_turple(model, true_entity, prediction) == False:
                false_negatives += 1
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    except:
        return 0.0

# 定义用于三元组关系抽取的f1-score
def macro_f1_score_3(model, prediction: List[List[Any]], answers: List[List[Any]]) -> float:
    """
    计算用于三元组关系抽取的宏平均 F1 分数。

    参数:
    prediction (List[List[Any]]): 预测的三元组列表。
    answers (List[List[Any]]): 真实的三元组列表。

    返回:
    float: F1 分数。
    """
    try:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        # matched_ground_truth_tokens = set()
        # similarity_threshold = 0.7 
        for pred_entity in prediction:
            if match_triplet(model, pred_entity, answers):
                true_positives += 1
            else:
                false_positives += 1
        for true_entity in answers:
            if match_triplet(model, true_entity, prediction) == False:
                false_negatives += 1
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    except:
        return 0.0

def scrub_formatting_from_prompt(prompt):
    scrubbed_prompt = copy.copy(prompt)

    if is_chat_prompt(prompt):
        for i, msg in enumerate(scrubbed_prompt):
            if "content" in msg:
                scrubbed_prompt[i]["content"] = msg["content"].replace("{", "{{").replace("}", "}}")
    else:
        scrubbed_prompt = scrubbed_prompt.replace("{", "{{").replace("}", "}}")
    return scrubbed_prompt


def format_necessary(template: str, allow_missing: bool = False, **kwargs: dict[str, str]) -> str:
    """Format a template string with only necessary kwargs."""
    keys = [k[1] for k in string.Formatter().parse(template) if k[1]]
    if allow_missing:
        assert (
            len([k for k in keys if k in kwargs]) > 0
        ), f"Required: {keys}, got: {sorted(kwargs)}, no inputs are used.\nTemplate:\n{template}"
        cur_keys = {k: kwargs.get(k, "{" + k + "}") for k in keys}
    else:
        assert all(
            k in kwargs for k in keys
        ), f"Required: {keys}, got: {sorted(kwargs)}.\nTemplate:\n{template}"
        cur_keys = {k: kwargs[k] for k in keys}
    return template.format(**cur_keys)


def format_prompt(
    prompt: OpenAICreatePrompt, allow_missing: bool = False, **kwargs: dict[str, str]
) -> OpenAICreatePrompt:
    """Format a prompt with only necessary kwargs."""
    # if any input kwargs is chat prompt, convert to text prompt
    kwargs = {
        k: chat_prompt_to_text_prompt(v, for_completion=False) if is_chat_prompt(v) else v
        for k, v in kwargs.items()
    }
    if is_chat_prompt(prompt):
        new_prompt = []
        for msg in prompt:
            formatted_msg = copy.copy(msg)
            if "content" in formatted_msg:
                formatted_msg["content"] = format_necessary(
                    formatted_msg["content"], allow_missing=allow_missing, **kwargs
                )
            new_prompt.append(formatted_msg)
        prompt = new_prompt
    else:
        # Prompt is a string
        prompt = format_necessary(prompt, allow_missing=allow_missing, **kwargs)
    return prompt


class PromptFn:
    """
    Wrap calls to a completion_fn with a prompt template with applicable keyword args.
    This will pass many args relevant to OpenAI Completion API, may be ignored by other completion_fn.
    """

    def __init__(
        self,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        completion_fn: CompletionFn,
        max_tokens: int,
        temperature: int = 0,
        n_samples: Optional[int] = None,
        completion_kwargs: Optional[dict] = {},
    ):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.completion_fn = completion_fn
        self.temperature = temperature
        self.completion_kwargs = completion_kwargs
        self.n_samples = n_samples

    def __call__(self, **kwargs):
        # if any input kwargs is chat prompt, convert to text prompt
        kwargs = {
            k: chat_prompt_to_text_prompt(v, for_completion=False) if is_chat_prompt(v) else v
            for k, v in kwargs.items()
        }
        if is_chat_prompt(self.prompt):
            prompt = []
            for msg in self.prompt:
                formatted_msg = copy.copy(msg)
                if "content" in formatted_msg:
                    formatted_msg["content"] = format_necessary(formatted_msg["content"], **kwargs)
                prompt.append(formatted_msg)
        else:
            # Prompt is a string
            prompt = format_necessary(self.prompt, **kwargs)

        result = self.completion_fn(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=(1 if self.n_samples is None else self.n_samples),
            **self.completion_kwargs,
        )
        sampled = result.get_completions()[0]
        return sampled, prompt


def markdown_format_prompt(prompt):
    if type(prompt) == list:
        return "\n\n".join([f"**{message['role']}**: {message['content']}" for message in prompt])
    else:
        return prompt
