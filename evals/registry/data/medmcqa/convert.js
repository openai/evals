/*
	MedMCQA data converter

	BEFORE PROCEEDING: Download train.json from the official repo:
	https://github.com/MedMCQA/MedMCQA

	(direct data link: https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky)

	Extract the archive and you are ready to go. :)
 */


const fs = require("fs");
const readline = require("readline");
const { Transform } = require("stream");

const inputFile = fs.createReadStream("train.json");
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


    // IMPORTANT: Option to include expert answer explanation for training purpose (in addition to correct answer)
    const includeExpertAnswer = false; // true;

    const indexToAnswer = (index) => {
      let output = ""
      if(index === 1) output = `a) ${obj["opa"]}`
      if(index === 2) output = `b) ${obj["opb"]}`
      if(index === 3) output = `c) ${obj["opc"]}`
      if(index === 4) output = `d) ${obj["opd"]}`

      if(includeExpertAnswer) output += `\n\n${obj.exp}`

      return output
    }
    const transformedObj = {
      input: [{ role: "system", content: "You are a highly intelligent doctor who answers the following multiple choice question correctly.\nOnly write the answer down." },{ role: "user", content: `Subject: ${obj.subject_name}\n\n${obj.question}\n\na) ${obj.opa}\nb) ${obj.opb}\nc) ${obj.opc}\nd) ${obj.opd}` },],ideal: indexToAnswer(obj["cop"]),
    };

    this.push(JSON.stringify(transformedObj) + "\n");
    done();
  },
});

// IMPORTANT:
// Hard coded limit (increase this to 182823 lines to use the full dataset)
const lineLimit = 300;

const rl = readline.createInterface({ input: inputFile, crlfDelay: Infinity });
let i=0;
rl.on("line", (line) => {
  if(i < lineLimit) {
    processLine.write(line + "\n");
    i++
  }
});
processLine.pipe(outputFile);
outputFile.on("error", (err) => console.error(`Error: ${err.message}`)).on("finish", () => console.log("Output file created successfully."));
rl.on("close", () => { processLine.end(); });
