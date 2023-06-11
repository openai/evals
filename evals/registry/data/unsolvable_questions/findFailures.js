/*
  Grab the file from the last run: /tmp/evallogs/<idxxx_gpt-3.5-turbo_impossible_detector>.jsonl
  Rename it to run.jsonl and put it in a folder called logs/ within this folder.

  Run this file and the failures will be outputted to a file named failure-samples.jsonl
  You can now keep these examples and combine them with other runs to get a good sample set that GPT struggles with
*/


const fs = require("fs");
const readline = require("readline");
const { Transform } = require("stream");

const inputFile = fs.createReadStream("logs/run.jsonl");
const outputFile = fs.createWriteStream("failure-samples.jsonl");

const processLine = new Transform({
  objectMode: true,
  transform(line, _, done) {
    if (!line) return done();

    let obj;
    try {
      obj = line;
    } catch (err) {
      return done(new Error(`Failed to parse JSON: ${err.message}`));
    }

    const transformedObj = obj;

    this.push(transformedObj + "\n");
    done();
  },
});

const parseLines = (line, previousLine) => {
  if(line && previousLine && !JSON.parse(previousLine).spec && JSON.parse(line).type === "match" && !JSON.parse(line).data.correct) {
    const promptData = JSON.parse(previousLine).data;
    let newLine = {};
    newLine.input = promptData.prompt;
    newLine.ideal = JSON.parse(line).data.expected;
    processLine.write(JSON.stringify(newLine));
  }
};

const lineLimit = 1500;

const rl = readline.createInterface({ input: inputFile, crlfDelay: Infinity });
let i=0;
let previousLine;
rl.on("line", (line) => {
  if(i < lineLimit) {
    parseLines(line, previousLine);
    previousLine = line;
    i++
  }
});
processLine.pipe(outputFile);
outputFile.on("error", (err) => console.error(`Error: ${err.message}`)).on("finish", () => console.log("Output file created successfully."));
rl.on("close", () => { processLine.end(); });
