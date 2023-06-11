import csv

#Eeach row of data_raw.csv has form (char,mandarin,cantonese,wu).
#char is the character: possibly on its own, possibly tradform-simpfor, e.g. "四" or "鳥-鸟"
#mandarin is pinyin with tone mark, then pinyin with tone number, e.g. "yī-yi1"
#cantonese is Jyutpin, then Yale, e.g. "yāt-jat1"
#Wu is Wiktionary romanization, then wugniu.com romanization, e.g. "4iq-iq7"

#Different combinations of example pronounciations are given in prompts. Both forms are always accepted as answers.

csvfile = open("data_raw.csv", "r")
data_raw = list(csv.reader(csvfile, delimiter=","))[1:] #drop header
csvfile.close()

zero_shot_json_list = []
few_shot_json_list = []

#note: since 法, 買, 立 are used in the few-shot example, they should not be used in the dataset of test questions
zero_shot_format = '{"input": [{"role": "system", "content": "You are a helpful assistant with knowledge of various spoken languages."}, {"role": "user", "content": "%s"}], "ideal": %s}'
few_shot_format = '{"input": [{"role": "system", "content": "Your role is a database of Chinese characters and their transcriptions in various Chinese languages, including Mandarin, Cantonese (also know as Yue), and Shanghainese (also known as Wu). You recognize characters in either traditional or simplified form and respond with single-syllable transcriptions in Pinyin, Jyutpin, Yale, etc."}, {"role": "user", "content": "Transcribe 法 in Mandarin"}, {"role": "assistant", "content": "fǎ"}, {"role": "user", "content": "Transcribe 立 in Shanghainese"}, {"role": "assistant", "content": "5liq"}, {"role": "user", "content": "Transcribe 買 in Cantonese"}, {"role": "assistant", "content": "maai5"}, {"role": "user", "content": "Transcribe %s in %s"}], "ideal": %s}'

for row in data_raw:
    char, mando, canto, wu = row
    #formatting traditional/simplified
    if '-' in char:
        trad, simp = char.split('-')
        char_desc = trad+" (simplified: "+simp+")"
    else:
        trad, simp = char, char
        char_desc = char
    mando_a, mando_b = mando.split('-')
    canto_a, canto_b = canto.split('-')
    wu_a, wu_b = wu.split('-')
    
    #Wu quiz prompts
    wu_prompt_1 = "The Chinese character %s is pronounced as %s in Mandarin, also written %s. In Cantonese, it is pronounced as %s, also written %s. How is %s pronounced in Wu Chinese? Answer with a single syllable and do not elaborate, like 1jin."
    wu_prompt_1 = wu_prompt_1 % (char_desc, mando_b, mando_a, canto_b, canto_a, trad)
    wu_prompt_2 = "The hanzi %s is written in Pinyin as %s with tone marks, or %s with tone numbers. Cantonese, or Yue, writes it as %s in the Yale system, or %s in Jyutpin. Tell me the Shanghainese pronounciation of %s. Respond with a single syllable and nothing else, like 2san."
    wu_prompt_2 = wu_prompt_2 % (char_desc, mando_a, mando_b, canto_a, canto_b, simp)
    wu_prompt_3 = "Different Chinese languages pronounce the same character differently. In Mandarin Chinese, %s is written %s (or %s). In Yue, %s is written %s (or %s). What is the Wu (Shanghainese) pronounciation of %s? Give just a single syllable and no elaboration, e.g. 2san."
    wu_prompt_3 = wu_prompt_3 % (char_desc, mando_a, mando_b, trad, canto_a, canto_b, trad)
    wu_prompts = [wu_prompt_1, wu_prompt_2, wu_prompt_3]
    
    #Canto quiz prompts
    canto_prompt_1 = "The Chinese character %s is pronounced as %s in Mandarin, also written %s. In Shanghainese, for contrast, it is pronounced %s, also written %s. How is %s pronounced in Cantonese? Answer with a single syllable and do not elaborate, like gam1."
    canto_prompt_1 = canto_prompt_1 % (char_desc, mando_b, mando_a, wu_a, wu_b, trad)
    canto_prompt_2 = "The hanzi %s is written in Pinyin as %s with tone marks, or %s with tone numbers. Shanghainese, or Wu, writes it as %s or %s. Tell me the Cantonese pronounciation of %s. Respond with a single syllable and nothing else, like gām."
    canto_prompt_2 = canto_prompt_2 % (char_desc, mando_a, mando_b, wu_a, wu_b, simp)
    canto_prompt_3 = "Different Chinese languages pronounce the same character differently. In Mandarin Chinese, %s is written %s (or %s). In Wu, %s can be transcribed as %s or %s. What is the Yue (Cantonese) pronounciation of %s? Give just a single syllable and no elaboration, e.g. gām."
    canto_prompt_3 = canto_prompt_3 % (char_desc, mando_a, mando_b, simp, wu_a, wu_b, simp)
    canto_prompts = [canto_prompt_1, canto_prompt_2, canto_prompt_3]
    
    mando_ans = '["%s", "%s"]' % (mando_a, mando_b)
    wu_ans = '["%s", "%s"]' % (wu_a, wu_b)
    canto_ans = '["%s", "%s"]' % (canto_a, canto_b)
    
    zero_shot_data = [(p, wu_ans) for p in wu_prompts] + [(p, canto_ans) for p in canto_prompts]
    zero_shot_json_list += [zero_shot_format % (q, a) for (q,a) in zero_shot_data]
    
    few_shot_json_list += [few_shot_format % (trad, "Cantonese", canto_ans), few_shot_format % (trad, "Shanghainese", wu_ans), few_shot_format % (trad, "Mandarin", mando_ans)]
    if trad != simp:
        few_shot_json_list += [few_shot_format % (simp, "Cantonese", canto_ans), few_shot_format % (simp, "Shanghainese", wu_ans), few_shot_format % (trad, "Mandarin", mando_ans)]

with open(r'./samples_zero.jsonl', 'w') as fp:
    for j in zero_shot_json_list:
        fp.write("%s\n" % j)
        
with open(r'./samples_few.jsonl', 'w') as fp:
    for j in few_shot_json_list:
        fp.write("%s\n" % j)
