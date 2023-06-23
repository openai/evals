/*
	SQuAD2.0 data converter

	BEFORE PROCEEDING: Download train.json from the official repo:
	https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

	Download the file, rename it to train.json and put it in this folder and you are ready to go. :)
 */


const fs = require("fs");
const { Transform } = require("stream");

const inputFile = fs.readFileSync("train.json");
const outputFile = fs.createWriteStream("samples.jsonl");

const processLine = new Transform({
  objectMode: true,
  transform(line, _, done) {
    if (!line) return done();

    let obj;
    try {
      obj = JSON.parse(line);
    } catch (err) {
      return done(new Error(`Failed to parse JSON: ${err.message}`));
    }

    const indexToAnswer = (bool) => {
      if (bool) {
        return "Unsolveable";
      }
      return "Solveable";
    };

    const transformedObj = {
      input: [
        { role: "system", content: 'You are UnsolvableGPT. You will be provided a question and some context for the question. Using only the context to answer the question determine if it is "Solveable" or "Unsolveable". Respond with only one word without punctuation, either: "Solveable": The submitted question is solveable with the context provided alongside it and no other outside information OR "Unsolveable": The submitted question is unsolveable with the context provided alongside it. There is not enough context to answer the question. Remember, only answer with "Solveable" OR "Unsolveable", do not include anything else.'},
        { role: "user", content: `Question: ${obj["question"]}\n Context: ${obj["context"]}` }
      ], ideal: indexToAnswer(obj["is_impossible"]),
    };

    this.push(JSON.stringify(transformedObj) + "\n");
    done();
  },
});

const parsedFile = JSON.parse(inputFile);
console.log(parsedFile);

for (let i = 0; i < parsedFile.data.length; i++) {
  const currentDocument = parsedFile.data[i];
  currentDocument.paragraphs.forEach((paragraph) => {
    const context = paragraph.context;
    // Generate a diverse training set by picking only 2 questions from each topic
    // Ensure an equal balance of unsolvable and solvable questions
    // Randomly select questions
    if (context.length > 1500 || context.length < 500 || Math.random() < 0.98) {
      return;
    }
    const possibleToAnswer = paragraph.qas.find(qa => !Boolean(qa.is_impossible));
    const impossibleToAnswer = paragraph.qas.find(qa => Boolean(qa.is_impossible));
    if(possibleToAnswer && impossibleToAnswer) {
      possibleToAnswer.context = context;
      impossibleToAnswer.context = context;
      processLine.write(JSON.stringify(possibleToAnswer) + "\n");
      processLine.write(JSON.stringify(impossibleToAnswer) + "\n");
    }
  });
}
processLine.pipe(outputFile);
outputFile.on("error", (err) => console.error(`Error: ${err.message}`)).on("finish", () => console.log("Output file created successfully."));
