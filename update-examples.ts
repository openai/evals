let few_shots = [
  ["Они не чувствуют себя другими", "другими", "instrumental"],
  ["Он чу́вствовал боль в спине́", "спине́", "dative"],
  [
    "Если в моё отсу́тствие что-нибудь случи́тся, попроси́ его помочь",
    "спине́",
    "genitive",
  ],
];

const firstPrompt = {
  role: "system",
  content:
    `Your job is take a sentence fragment in russian, and a word within the fragment, and determine the case of the word. For example, if the fragment is 'Они не чувствуют себя другими', and the word is 'другими', then the answer is 'instrumental'.`,
};

const getInputPrompt = (sentence_fragment: string, word: string) => {
  return {
    role: "user",
    content: `Sentence fragment: '${sentence_fragment}' Word: '${word}' Case: `,
    // "name": "example_user",
  };
};

const getResponse = (result: string) => {
  return {
    role: "user",
    content: result,
    "name": "example_assistant",
  };
};

function getFewShotLine(
  sentence_fragment: string,
  word: string,
  result: string,
) {
  const firstPromptString = JSON.stringify(firstPrompt);
  const secondPromptString = JSON.stringify(
    getInputPrompt(sentence_fragment, word),
  );
  const response = JSON.stringify(getResponse(result));

  return `{"sample": [${firstPromptString},${secondPromptString},${response}]}`;
}

await Deno.writeTextFile(
  "./evals/registry/data/russian_cases/few_shot.jsonl",
  few_shots.map((few_shot) => {
    return getFewShotLine(few_shot[0], few_shot[1], few_shot[2]);
  }).join("\n"),
);

let examples = [
  ["Книга на столе.", "столе", "prepositional"],
  ["Майк любит Веру.", "Веру", "accusative"],
  ["офис компании, сестра ребенка, его команда", "его", "genitive"],
  ["Я пишу ручкой", "ручкой", "instrumental"],
  ["Я заказал цветы подруге", "подруге", "dative"],
];

function getExampleLine(
  sentence_fragment: string,
  word: string,
  result: string,
) {
  const firstPromptString = JSON.stringify(firstPrompt);
  const secondPromptString = JSON.stringify(
    getInputPrompt(sentence_fragment, word),
  );

  return `{"input":[${firstPromptString},${secondPromptString}],"ideal": "${result}"}`;
}

await Deno.writeTextFile(
  "./evals/registry/data/russian_cases/basic_samples.jsonl",
  examples.map((example) => {
    return getExampleLine(example[0], example[1], example[2]);
  }).join("\n"),
);
