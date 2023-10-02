import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [model, setModel] = useState('');
  const [evalName, setEvalName] = useState('');
  const [result, setResult] = useState('');
  const [evals, setEvals] = useState([]);
  const [evalRun, setEvalRun] = useState(false);

  useEffect(() => {
    const fetchEvals = async () => {
      const response = await axios.get('http://localhost:5000/get-evals');
      setEvals(response.data);
    };
    fetchEvals();
  }, []);

  useEffect(() => {
    const resultBox = document.querySelector('.result');
    if (resultBox) {
      resultBox.scrollTop = resultBox.scrollHeight;
    }
  }, [result]);

  const runEval = () => {
    setResult(''); // clear results for new run
    setEvalRun(true);
    const eventSource = new EventSource(`http://localhost:5000/run-eval?model=${model}&evalName=${evalName}`);
  
    eventSource.onmessage = (event) => {
      setResult(prevResult => prevResult + event.data);
    };
  
    eventSource.onerror = (event) => {
      console.error('EventSource failed:', event);
      eventSource.close();
    };
};
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to OpenAI Evals</h1>
        <div className="select-container">
          <select className="select" value={model} onChange={e => setModel(e.target.value)}>
            <option value="">Select a model</option>
            <option value="gpt-4">gpt-4</option>
            <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
            <option value="gpt-3.5-turbo-16k">gpt-3.5-turbo-16k</option>
          </select>
          <select className="select" value={evalName} onChange={e => setEvalName(e.target.value)}>
            <option value="">Select an eval</option>
            {evals.map(evalName => <option key={evalName} value={evalName}>{evalName}</option>)}
          </select>
        </div>
        <button className="run-button" onClick={runEval}>Run Eval</button>
        {evalRun && <div className="result" style={{whiteSpace: 'pre-wrap'}}>{result}</div>}
      </header>
    </div>
  );
}

export default App;