// server.js
const express = require('express');
const app = express();
const port = process.env.PORT || 5000;
const { exec } = require('child_process');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

app.use(cors());
app.use(express.json());
app.use(cors({
    origin: 'http://localhost:3000' // replace with your trusted origin
  }));

app.use(express.json());

app.post('/run-eval', (req, res) => {
    const { model, evalName } = req.body;
  
    // Set response headers for SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
  
    const child = exec(`oaieval ${model} ${evalName}`);
  
    child.stdout.on('data', (data) => {
        res.write(`data: ${data}\n\n`); // Send a SSE with the data
    });
    
    child.stderr.on('data', (data) => {
        res.write(`data: ${data}\n\n`); // Send a SSE with the error
    });
  
    child.on('close', (code) => {
      res.end(); // Close the connection when the command finishes
    });
  });

  app.get('/run-eval', (req, res) => {
    const { model, evalName } = req.query;
  
    // Set response headers for SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
  
    const child = exec(`oaieval ${model} ${evalName}`);
  
    child.stdout.on('data', (data) => {
      res.write(`data: ${data}\n\n`); // Send a SSE with the data
    });
  
    child.stderr.on('data', (data) => {
      res.write(`data: ${data}\n\n`); // Send a SSE with the error
    });
  
    child.on('close', (code) => {
      res.end(); // Close the connection when the command finishes
    });
  });

app.get('/get-evals', (req, res) => {
    const evalsDir = path.join(__dirname, '../evals/registry/evals');
    fs.readdir(evalsDir, (err, files) => {
      if (err) {
        console.log(err);
        res.status(500).send('Error reading evals directory');
        return;
      }
      const evalNames = files.map(file => file.replace('.yaml', ''));
      res.send(evalNames);
    });
  });

app.listen(port, () => {
console.log(`Server is running on port ${port}`);
});
