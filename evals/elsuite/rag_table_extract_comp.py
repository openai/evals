from io import StringIO
import json
import os
import re

from typing import List, Optional, Tuple, Union
import uuid
import numpy as np

import pandas as pd
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.rag_match import get_rag_dataset
from evals.record import RecorderBase, record_match

code_pattern = r"```[\s\S]*?\n([\s\S]+?)\n```"
json_pattern = r"```json[\s\S]*?\n([\s\S]+?)\n```"
csv_pattern = r"```csv[\s\S]*?\n([\s\S]+?)\n```"
outlink_pattern = r"\[Download[a-zA-Z0-9 ]+?\]\((https://[a-zA-Z0-9_. /]+?)\)"

mchkye = {'Hydrogen':'H',
'Helium':'He',
'Lithium':'Li',
'Beryllium':'Be',
'Boron':'B',
'Carbon':'C',
'Nitrogen':'N',
'Oxygen':'O',
'Fluorine':'F',
'Neon':'Ne',
'Sodium':'Na',
'Magnesium':'Mg',
'Aluminium(aluminum)':'Al',
'Silicon':'Si',
'Phosphorus':'P',
'Sulfur':'S',
'Chlorine':'Cl',
'Argon':'Ar',
'Potassium':'K',
'Calcium':'Ca',
'Scandium':'Sc',
'Titanium':'Ti',
'Vanadium':'V',
'Chromium':'Cr',
'Manganese':'Mn',
'Iron':'Fe',
'Cobalt':'Co',
'Nickel':'Ni',
'Copper':'Cu',
'Zinc':'Zn',
'Gallium':'Ga',
'Germanium':'Ge',
'Arsenic':'As',
'Selenium':'Se',
'Bromine':'Br',
'Krypton':'Kr',
'Rubidium':'Rb',
'Strontium':'Sr',
'Yttrium':'Y',
'Zirconium':'Zr',
'Niobium':'Nb',
'Molybdenum':'Mo',
'Technetium':'Tc',
'Ruthenium':'Ru',
'Rhodium':'Rh',
'Palladium':'Pd',
'Silver':'Ag',
'Cadmium':'Cd',
'Indium':'In',
'Tin':'Sn',
'Antimony':'Sb',
'Tellurium':'Te',
'Iodine':'I',
'Xenon':'Xe',
'Cesium':'Cs',
'Barium':'Ba',
'Lanthanum':'La',
'Cerium':'Ce',
'Praseodymium':'Pr',
'Neodymium':'Nd',
'Promethium':'Pm',
'Samarium':'Sm',
'Europium':'Eu',
'Gadolinium':'Gd',
'Terbium':'Tb',
'Dysprosium':'Dy',
'Holmium':'Ho',
'Erbium':'Er',
'Thulium':'Tm',
'Ytterbium':'Yb',
'Lutetium':'Lu',
'Hafnium':'Hf',
'Tantalum':'Ta',
'Tungsten':'W',
'Rhenium':'Re',
'Osmium':'Os',
'Iridium':'Ir',
'Platinum':'Pt',
'Gold':'Au',
'Mercury':'Hg',
'Thallium':'Tl',
'Lead':'Pb',
'Bismuth':'Bi',
'Polonium':'Po',
'Astatine':'At',
'Radon':'Rn',
'Francium':'Fr',
'Radium':'Ra',
'Actinium':'Ac',
'Thorium':'Th',
'Protactinium':'Pa',
'Uranium':'U',
'Neptunium':'Np',
'Plutonium':'Pu',
'Americium':'Am',
'Curium':'Cm',
'Berkelium':'Bk',
'Californium':'Cf',
'Einsteinium':'Es',
'Fermium':'Fm'}

def parse_csv_text(csvtext: str) -> str:
    lines = csvtext.strip().split("\n")
    tuple_pattern = r"\((\"[\s\S]*?\"),(\"[\s\S]*?\")\)"
    if re.search(tuple_pattern, lines[0]) is not None:
        lines[0] = re.sub(tuple_pattern, r"(\1|\2)", lines[0])
    lines_clr = [re.sub(r"\"[\s\S]*?\"", "", line) for line in lines]
    max_commas = max([line_clr.count(",") for line_clr in lines_clr])
    unified_lines = [line + ("," * (max_commas - line_clr.count(","))) for line, line_clr in zip(lines, lines_clr)]
    return "\n".join(unified_lines)


def parse_table_multiindex(table: pd.DataFrame) -> pd.DataFrame:
    """
    Parse a table with multiindex columns.
    """

    df = table.copy()
    if df.columns.nlevels == 1:
        coltypes = {col: type(df[col].iloc[0]) for col in df.columns}
        for col, ctype in coltypes.items():
            if ctype == str:
                if ":" in df[col].iloc[0] and "," in df[col].iloc[0]:
                    df[col] = [{key: value for key, value in [pair.split(": ") for pair in data.split(", ")]} for data
                               in df[col]]
                    coltypes[col] = dict
        dfs = []

        for col, ctype in coltypes.items():
            if ctype == dict:
                d = pd.DataFrame(df.pop(col).tolist())
                d.columns = pd.MultiIndex.from_tuples([(col, fuzzy_normalize(key)) for key in d.columns])
                dfs.append(d)
        df.columns = pd.MultiIndex.from_tuples([eval(col.replace("|", ",")) if (col[0] == "(" and col[-1] == ")") else
                                                (col, "") for col in df.columns])
        df = pd.concat([df] + dfs, axis=1)
    if df.columns.nlevels > 1:
        df.columns = pd.MultiIndex.from_tuples([(col, fuzzy_normalize(subcol)) for col, subcol in df.columns])

    return df

class FileSample(BaseModel):
    file_name: Optional[str]
    file_link: Optional[str] = None
    answerfile_name: Optional[str]
    answerfile_link: Optional[str] = None
    compare_fields: List[Union[str, Tuple]]
    index: Union[str, Tuple] = ("Compound", "")
    
    
def fuzzy(vi):
    try:
        vi =str(vi).lower()

        if "bal" in vi or "remainder" in vi or "bas" in vi:
            vi = "bal"
            return "bal"
                
        if "nan" in vi or "/" == vi or "n/a" in vi or "na" in vi  or vi == "":
            vi = "0"
        
        vi = vi.replace("~","-")

        pattern = r"\d+(?:\.\d+)?"
        vi = re.findall(pattern, vi)
        if len(vi)==2:
            vi = f"{vi[0]}-{vi[1]}"
        else:
            vi = vi[0]
        
        if "<" in vi:
            vi = vi.replace("<","")
        if ">" in vi:
            vi = vi.replace(">","")
        
        try:
            vi = float(vi)
            vi = round(vi,3)
        except:
            # print(vi)
            pass
    except:
        print("Can't fuzzy for", vi)
    
    return vi


def fuzzy_compare(a: str, b: str) -> bool:
    nan_str = ["n/a", "nan", "na", "nd"," nan","nan "]
    if a.lower() in nan_str and b.lower() in nan_str:
        return True
    else:
        a = fuzzy(a)
        b = fuzzy(b)
        if a == "bal" or b == "bal":
            return True
        if type(a)==type(b):
            return a == b
        else:
            return False



def fuzzy_normalize(s):
    if s.startswith("Unnamed"):
        return ""
    else:
        """ 标准化字符串 """
        # 定义需要移除的单位和符号
        units = ["µM", "µg/mL", "nM","%","wt.%","at.%","at%","wt%"]
        for unit in units:
            s = s.replace(unit, "")

        # 定义特定关键字
        # keywords = ["IC50", "EC50", "TC50", "GI50", "Ki", "Kd"]

        # 移除非字母数字的字符，除了空格
        s = re.sub(r'[^\w\s]', '', s)

        # 分割字符串为单词列表
        # words = s.split()

        # 将关键字移到末尾
        # reordered_words = [word for word in words if word not in keywords]
        # keywords_in_string = [word for word in words if word in keywords]
        # reordered_words.extend(keywords_in_string)
        # # 重新组合为字符串
        # return ' '.join(reordered_words)
        if s in mchkye:
            s = mchkye[s]
        return s
    

def my_sort(predict_table,correct_table):
    
    index1 = predict_table.index
    index2 = correct_table.index
    index1 = [i for i in index1]
    index2 = [i for i in index2]
    
    import difflib

    def cs(str1, str2):
        similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
        return similarity
    
    sorts = []
    
    if len(index2)>0:
        for i in index2:
            rank = np.array([cs(str(i),str(j)) for j in index1])
            j = np.argmax(rank)
            sorts.append(j)
        predict_table = predict_table.iloc[sorts]
        
    return predict_table
    
    


class TableExtract(evals.Eval):
    def __init__(
            self,
            completion_fns: list[CompletionFn],
            samples_jsonl: str,
            *args,
            instructions: Optional[str] = "",
            **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) < 3, "TableExtract only supports 3 completion fns"
        self.samples_jsonl = samples_jsonl
        self.instructions = instructions

    def eval_sample(self, sample, rng):
        try:
            assert isinstance(sample, FileSample)

            prompt = (
                    self.instructions
                    + f"\nThe header of csv should at least contain {sample.compare_fields}"
            )
            
            # try:
            #     # print(sample.file_link)
            
            result = self.completion_fn(
                prompt=prompt,
                temperature=0.0,
                max_tokens=5,
                file_name=sample.file_name,
                file_link=sample.file_link
            )
            sampled = result.get_completions()[0]
            
            compare_fields_types = [type(x) for x in sample.compare_fields]
            header_rows = [0, 1] if tuple in compare_fields_types else [0]

            correct_answer = parse_table_multiindex(pd.read_csv(sample.answerfile_name, header=header_rows,index_col=0).astype(str))
            correct_answer.to_csv("temp.csv", index=False)
            correct_str = open("temp.csv", 'r').read()
                
            # except BaseException as e:
            #     # print(sample.file_name)
            #     raise TimeoutError("Connect error")

            try:
                if re.search(outlink_pattern, sampled) is not None:
                    code = re.search(outlink_pattern, sampled).group()
                    link = re.sub(outlink_pattern, r"\1", code)

                    fname = f"/tmp/LLMEvals_{uuid.uuid4()}.csv"
                    os.system(f"wget {link} -O {fname}")
                    table = pd.read_csv(fname)
                    if pd.isna(table.iloc[0, 0]) or table.iloc[0, 0]=="" or str(table.iloc[0, 0]).lower()=="nan":
                        table = pd.read_csv(fname, header=header_rows)
                    print("local csv:")
                    print(table)
                elif "```csv" in prompt:
                    code = re.search(csv_pattern, sampled).group()
                    code_content = re.sub(csv_pattern, r"\1", code)
                    code_content_processed = parse_csv_text(code_content)
                    table = pd.read_csv(StringIO(code_content_processed))
                    if pd.isna(table.iloc[0, 0]) or table.iloc[0, 0]=="" or str(table.iloc[0, 0]).lower()=="nan":
                        table = pd.read_csv(StringIO(code_content_processed), header=header_rows,index_col=0)
                    print("sparse csv:")
                    print(table)

                else:
                    table = pd.DataFrame()

                table = parse_table_multiindex(table)
                # print(table)
                answerfile_out = sample.answerfile_name.replace(".csv", "_output.csv")
                # print(answerfile_out)
                table.to_csv(answerfile_out, index=False)
                picked_str = open(answerfile_out, 'r').read()
                table = my_sort(table,correct_answer)
            except:
                record_match(
                    prompt=prompt,
                    correct=False,
                    expected=correct_str,
                    picked=sampled,
                    file_name=sample.file_name,
                    jobtype="match_all"
                )
                return
            

            match_all = True
            for field in sample.compare_fields:
                if type(field) == tuple and len(field) > 1:
                    field = (field[0], fuzzy_normalize(field[1]))
                match_field = field in table.columns and field in correct_answer.columns #在里面
                match_all = match_all and match_field
                record_match(
                    correct=match_field,
                    expected=field,
                    picked=str(list(table.columns)),
                    file_name=sample.file_name,
                    jobtype="match_field"
                )
                if match_field:
                    match_number = table[field].shape[0] == correct_answer[field].shape[0] # 匹配样本数
                    match_all = match_all and match_number
                    record_match(
                        correct=match_number,
                        expected=correct_answer[field].shape[0],
                        picked=table[field].shape[0],
                        file_name=sample.file_name,
                        jobtype="match_number"
                    )

                    for sample_value, correct_value in zip(table[field], correct_answer[field]):
                        match_value = fuzzy_compare(str(sample_value), str(correct_value))
                        match_all = match_all and match_value
                        record_match(
                            correct=match_value,
                            expected=correct_value,
                            picked=sample_value,
                            file_name=sample.file_name,
                            jobtype=field if type(field) == str else field[0]
                        )
                        
            record_match(
                prompt=prompt,
                correct=match_all,
                expected=correct_str,
                picked=picked_str,
                file_name=sample.file_name,
                jobtype="match_all"
            )
        except BaseException as e:
            pass
            # print(e)
            # record_match(
            #     correct=False,
            #     expected=sample.file_name,
            #     picked=sample.file_name,
            #     file_name=sample.file_name,
            #     jobtype="match_all"
            # )


    def run(self, recorder: RecorderBase):
        raw_samples = get_rag_dataset(self._prefix_registry_path(self.samples_jsonl).as_posix())
        # samples_jsonl = self._prefix_registry_path(self.samples_jsonl).as_posix()
        # raw_samples = evals.get_jsonl(samples_jsonl)
        for raw_sample in raw_samples:
            raw_sample["compare_fields"] = [field if type(field) == str else tuple(field) for field in
                                            raw_sample["compare_fields"]]

        samples = [FileSample(**raw_sample) for raw_sample in raw_samples]
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
