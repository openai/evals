const name = "color-wheel";
const dataName = name.replace("-", "_");

const colorWheel = [
  "red",
  "vermillion",
  "orange",
  "amber",
  "yellow",
  "chartreuse",
  "green",
  "teal",
  "blue",
  "violet",
  "purple",
  "magenta",
];

// function which takes two colors on the color wheel and calculates the distance between them
// the color wheel is circular, so the distance between red and violet is 3
function colorDistance(color1: string, color2: string) {
  const color1Index = colorWheel.indexOf(color1);
  const color2Index = colorWheel.indexOf(color2);
  const distance = Math.abs(color1Index - color2Index);
  return Math.min(distance, colorWheel.length - distance);
}

const randomColors = [
  ["yellow", "green"],
  ["chartreuse", "red"],
  ["green", "purple"],
  ["teal", "vermillion"],
  ["blue", "green"],
  ["violet", "teal"],
  ["purple", "orange"],
  ["magenta", "purple"],
  ["red", "green"],
  ["vermillion", "magenta"],
  ["orange", "blue"],
  ["amber", "purple"],
  ["purple", "teal"],
  ["blue", "amber"],
  ["magenta", "teal"],
];

// const colorSet = new Map();

// for (const randomColorPair of randomColors) {
//   colorSet.set(randomColorPair[0], randomColorPair[1]);
// }

// throw new Error(JSON.stringify([...colorSet.entries()]));

// A mapping of the random colors, with whether they are close or not as the third value
// [["vermillion", "magenta", "Yes"]]

const randomColorsClose = [
  ["yellow", "amber"],
  ["chartreuse", "red"],
  ["green", "purple"],
  ["teal", "vermillion"],
  ["blue", "green"],
  ["violet", "teal"],
  ["purple", "orange"],
  ["magenta", "purple"],
  ["red", "green"],
  ["vermillion", "magenta"],
  ["orange", "blue"],
  ["amber", "purple"],
  ["amber", "green"],
  ["purple", "teal"],
  ["blue", "amber"],
  ["magenta", "teal"],
];
const randomColorsCloseDistance = randomColorsClose.map(([colorA, colorB]) => {
  return [colorA, colorB, colorDistance(colorA, colorB)];
});
// throw new Error(JSON.stringify(randomColorsCloseDistance));

let few_shots: [
  colorA: string,
  colorB: string,
  distance: string,
][] = [
  ["yellow", "chartreuse", "1"],
  ["yellow", "blue", "4"],
  ["yellow", "green", "2"],
  ["yellow", "orange", "2"],
  ["yellow", "vermillion", "4"],
];

const firstPrompt = {
  role: "system",
  content: `You are a helpful assistant.`,
};

const getInputPrompt = (sentence_fragment: string, word: string) => {
  return {
    role: "user",
    content: `Given a color wheel with the values ${
      colorWheel.join(", ")
    }, what is the distance from ${sentence_fragment} to ${word}? Answer with 1, 2, 3, 4, 5 or 6. Your answer should only be that number.`,
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

let examples = randomColorsCloseDistance;

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
