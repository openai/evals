const name = "animal-speed-size";
const dataName = name.replace("-", "_");

let few_shots = [
  ["mouse", "horse", "No"],
  ["dolphin", "pig", "Yes"],
  ["elephant", "dog", "No"],
  ["shark", "snail", "Yes"],
];

const firstPrompt = {
  role: "system",
  content: `You are a helpful assistant.`,
};

const getInputPrompt = (sentence_fragment: string, word: string) => {
  return {
    role: "user",
    content:
      `Is a ${sentence_fragment} faster and larger than a ${word}? Answer with Yes or No`,
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

try {
  await Deno.mkdir(`./evals/registry/data/${dataName}`);
} catch {}

await Deno.writeTextFile(
  `./evals/registry/data/${dataName}/few_shot.jsonl`,
  few_shots.map((few_shot) => {
    return getFewShotLine(few_shot[0], few_shot[1], few_shot[2]);
  }).join("\n"),
);

let examples = [
  ["cat", "stallion", "No"],
  ["mosquito", "octopus", "No"],
  ["cow", "gemsbok", "No"],
  ["snail", "polar bear", "No"],
  ["horse", "rooster", "Yes"],
  ["lion", "starfish", "Yes"],
  ["swordfish", "bison", "No"],
  ["rabbit", "sloth", "No"],
  ["gorrila", "koalla", "Yes"],
  ["panther", "otter", "Yes"],
  ["horsefly", "lynx", "No"],
  ["horsefly", "lynx", "No"],
  ["giraffe", "wombat", "Yes"],
  ["kangaroo", "badger", "No"],
  ["saltwater crocodile", "monitor lizard", "Yes"],
  ["narwhal", "lynx", "No"],
  ["emu", "ostrich", "No"],
  ["warthog", "panda", "No"],
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
  `./evals/registry/data/${dataName}/samples.jsonl`,
  examples.map((example) => {
    return getExampleLine(example[0], example[1], example[2]);
  }).join("\n"),
);

await Deno.writeTextFile(
  `evals/registry/evals/${name}.yaml`,
  `
${name}:
  id: ${name}.test.v0
  metrics: [accuracy]

${name}.test.v0:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: ${dataName}/samples.jsonl
    few_shot_jsonl: ${dataName}/few_shot.jsonl
    num_few_shot: ${few_shots.length}
`.trim(),
);
