import json
import os
from typing import List, Dict
import requests
from datasets import load_dataset
from openai import OpenAI
import time
import io
import soundfile as sf
import subprocess
import base64
import sacrebleu
import multiprocessing as mp
from pathlib import Path
import shutil
from functools import partial
import numpy as np
import hashlib

AUDIO_PLACEHOLDER = "<|reserved_special_token_0|>"
TASK_PROMPT = f"Please translate the text to {{language}}. Your response should only include the {{language}} translation, without any additional words. \n\n{AUDIO_PLACEHOLDER}"

# Initialize API clients
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
FIXIE_API_KEY = os.environ.get("ULTRAVOX_API_KEY")

def load_covost_dataset(language_pair: str = "en_ca", split: str = "test") -> List[Dict]:
    """Load CoVoST dataset for the specified language pair in streaming mode."""
    dataset = load_dataset(
        "fixie-ai/covost2", 
        name=language_pair, 
        split=split, 
        streaming=True
    )
    return dataset

def translate_text_fireworks(text: str, target_language: str) -> str:
    """Translate text using Fireworks API directly."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}"
    }
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Please translate this text to {target_language}. Your response should only include the {target_language} translation, without any additional words:\n\n{text}"}
    ]
    
    data = {
        "model": "accounts/fireworks/models/llama-v3p1-70b-instruct#accounts/fixie/deployments/d2630f9a",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    response = requests.post(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Fireworks API error: {response.text}")

def get_auth_token() -> str:
    """Get authentication token by impersonating service account."""
    cmd = [
        "gcloud", "auth", "print-identity-token",
        "--impersonate-service-account=developer-service-account@fixie-frame.iam.gserviceaccount.com",
        "--audiences=https://api.ultravox.ai",
        "--include-email"
    ]
    
    try:
        token = subprocess.check_output(cmd).decode('utf-8').strip()
        return token
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to get auth token: {str(e)}")

def translate_audio_fixie(audio_data: dict, target_language: str) -> str:
    """Translate audio using Fixie/Ultravox API from raw audio data."""
    auth_token = get_auth_token()
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    # Get the original audio data
    audio_array = audio_data['array']
    sampling_rate = audio_data['sampling_rate']
    
    # Calculate number of samples for 1.5 seconds of silence
    silence_samples = int(1.5 * sampling_rate)
    
    # Create silence padding and concatenate with original audio at both ends
    silence_padding = np.zeros(silence_samples, dtype=audio_array.dtype)
    padded_audio = np.concatenate([silence_padding, audio_array, silence_padding])
    
    # Convert padded audio to base64
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, padded_audio, sampling_rate, format='wav')
    audio_buffer.seek(0)
    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
    data_url = f"data:audio/wav;base64,{audio_base64}"
    
    # Make the chat completion request
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": data_url
                    }
                },
                {
                    "type": "text",
                    "text": TASK_PROMPT.format(language=target_language)
                }
            ]
        }
    ]
    
    data = {
        "model": "fixie-ai/ultravox-70B-dev",
        "messages": messages
    }
    
    response = requests.post(
        "https://api.ultravox.ai/api/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Fixie API error: {response.text}")

def evaluate_translation(reference: str, hypothesis: str) -> float:
    """Evaluate translation using sentence BLEU score."""
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score

def calculate_bleu(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score between reference and hypothesis."""
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score

def transcribe_audio_whisper(audio_data: dict) -> str:
    """Transcribe audio using OpenAI's Whisper API from raw audio data."""
    # Convert the audio array to a temporary file-like object
    audio_array = audio_data['array']
    sampling_rate = audio_data['sampling_rate']
    
    # Create an in-memory bytes buffer
    audio_buffer = io.BytesIO()
    # Write the audio data to the buffer in WAV format
    sf.write(audio_buffer, audio_array, sampling_rate, format='wav')
    # Seek to the beginning of the buffer
    audio_buffer.seek(0)
    
    # Send to Whisper API
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_buffer, "audio/wav"),
    )
    print(transcript)
    return transcript.text

def get_stable_hash(audio_array: np.ndarray) -> str:
    """Generate a stable hash from audio array content."""
    # Convert array to bytes in a consistent way
    array_bytes = audio_array.tobytes()
    
    # Use SHA-256 for consistent hashing
    hasher = hashlib.sha256()
    hasher.update(array_bytes)
    
    # Return first 16 characters of hex digest
    return hasher.hexdigest()[:16]

def process_single_sample(sample, cache_dir: str, target_language: str = "Catalan"):
    """Process a single sample with caching."""
    try:
        # Create cache directories
        audio_cache_dir = Path(cache_dir) / "audio"
        results_cache_dir = Path(cache_dir) / "results"
        audio_cache_dir.mkdir(parents=True, exist_ok=True)
        results_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filenames using stable hash
        audio_data = sample['audio']
        audio_hash = get_stable_hash(audio_data['array'])
        cache_path = audio_cache_dir / f"{audio_hash}.wav"
        results_path = results_cache_dir / f"{audio_hash}.json"
        print(f"Processing sample {audio_hash}")
        # Check if results already exist
        if results_path.exists():
            print("Loading results from cache")
            with open(results_path, 'r') as f:
                return json.load(f)
        
        # Cache the audio file if it doesn't exist
        if not cache_path.exists():
            sf.write(str(cache_path), audio_data['array'], audio_data['sampling_rate'])
        
        # Get Whisper transcription
        whisper_transcription = transcribe_audio_whisper(audio_data)
        ground_truth_transcription = sample['sentence']
        reference_translation = sample['translation']
        
        # Calculate BLEU score for transcription
        transcription_bleu = calculate_bleu(ground_truth_transcription, whisper_transcription)
        
        # Get translations
        whisper_fireworks_translation = translate_text_fireworks(
            whisper_transcription,
            target_language
        )
        
        fixie_translation = translate_audio_fixie(
            audio_data,
            target_language
        )
        
        # Evaluate translations using BLEU
        whisper_fireworks_bleu = evaluate_translation(reference_translation, whisper_fireworks_translation)
        fixie_bleu = evaluate_translation(reference_translation, fixie_translation)
        
        result = {
            "cache_path": str(cache_path),
            "ground_truth_transcription": ground_truth_transcription,
            "whisper_transcription": whisper_transcription,
            "transcription_bleu": transcription_bleu,
            "reference_translation": reference_translation,
            "whisper_fireworks_translation": whisper_fireworks_translation,
            "fixie_translation": fixie_translation,
            "whisper_fireworks_bleu": whisper_fireworks_bleu,
            "fixie_bleu": fixie_bleu,
            "bleu_delta": fixie_bleu - whisper_fireworks_bleu
        }
        
        # Cache the results
        with open(results_path, 'w') as f:
            json.dump(result, f)
        
        return result
    
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None

def print_detailed_example(result: dict, idx: int, category: str):
    """Print detailed information for a single example."""
    print(f"\nExample {idx + 1} ({category})")
    print("=" * 80)
    print(f"BLEU Delta: {result['bleu_delta']:.4f}")
    print(f"Audio Path: {result['cache_path']}")
    print("\nTranscription Comparison:")
    print(f"Ground Truth:  {result['ground_truth_transcription']}")
    print(f"Whisper:      {result['whisper_transcription']}")
    print(f"Transcription BLEU: {result['transcription_bleu']:.4f}")
    print("\nTranslation Comparison:")
    print(f"Reference:          {result['reference_translation']}")
    print(f"Whisper+Fireworks:  {result['whisper_fireworks_translation']}")
    print(f"Whisper+Fireworks BLEU: {result['whisper_fireworks_bleu']:.4f}")
    print(f"Fixie:              {result['fixie_translation']}")
    print(f"Fixie BLEU:         {result['fixie_bleu']:.4f}")
    print("=" * 80)

def main():
    dataset = load_covost_dataset()
    max_samples = 500
    cache_dir = ".cache"
    
    # List of hashes to rerun
    hashes_to_rerun = {

    }
    
    # Collect samples
    samples = []
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
            
        if hashes_to_rerun:
            # Use the stable hash function
            audio_hash = get_stable_hash(sample['audio']['array'])
            print(f"Sample {idx} hash: {audio_hash}")  # Debug print
            if audio_hash in hashes_to_rerun:
                samples.append(sample)
                print(f"Found sample with matching hash: {audio_hash}")
        else:
            samples.append(sample)
    
    if hashes_to_rerun:
        print(f"Found {len(samples)} samples to reprocess")
        
        # Delete existing cache files for these hashes
        results_cache_dir = Path(cache_dir) / "results"
        audio_cache_dir = Path(cache_dir) / "audio"
        for hash_val in hashes_to_rerun:
            # Remove cached results
            result_file = results_cache_dir / f"{hash_val}.json"
            if result_file.exists():
                result_file.unlink()
            
            # Remove cached audio
            audio_file = audio_cache_dir / f"{hash_val}.wav"
            if audio_file.exists():
                audio_file.unlink()
    
    # Process samples in parallel
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        process_fn = partial(process_single_sample, cache_dir=cache_dir)
        results = list(pool.imap(process_fn, samples))
    
    # Filter out None results (failed processes)
    results = [r for r in results if r is not None]
    
    # Sort results by BLEU score delta (Fixie - Whisper+Fireworks)
    results.sort(key=lambda x: x['bleu_delta'], reverse=True)
    

    # Print summary statistics
    if results:
        whisper_fireworks_avg = np.mean([r["whisper_fireworks_bleu"] for r in results])
        fixie_avg = np.mean([r["fixie_bleu"] for r in results])
        transcription_bleu_avg = np.mean([r["transcription_bleu"] for r in results])
        bleu_delta_avg = np.mean([r["bleu_delta"] for r in results])
        
        print("\nResults Summary:")
        print("=" * 80)
        print(f"Total samples processed: {len(results)}")
        print(f"Average Transcription BLEU Score: {transcription_bleu_avg:.4f}")
        print(f"Whisper + Fireworks Average BLEU: {whisper_fireworks_avg:.4f}")
        print(f"Fixie (Ultravox) Average BLEU: {fixie_avg:.4f}")
        print(f"Average BLEU Delta (Fixie - Whisper+Fireworks): {bleu_delta_avg:.4f}")
        
        # Filter results
        fixie_better_results = [r for r in results if r['bleu_delta'] > 0]
        fireworks_better_results = [r for r in results if r['bleu_delta'] < 0]
        tied_results = [r for r in results if r['bleu_delta'] == 0]
        
        # Print top 20 examples where Fixie performed better
        print(f"\nTop 20 examples where Fixie performed better (out of {len(fixie_better_results)} total wins):")
        print("=" * 80)
        for idx, result in enumerate(fixie_better_results[:20]):
            print_detailed_example(result, idx, "Fixie Better")
            
        # Print top 20 examples where Fireworks performed better
        print(f"\nTop 20 examples where Whisper+Fireworks performed better (out of {len(fireworks_better_results)} total wins):")
        print("=" * 80)
        for idx, result in enumerate(sorted(fireworks_better_results, key=lambda x: x['bleu_delta'])[:20]):
            print_detailed_example(result, idx, "Whisper+Fireworks Better")
            
        # Print up to 20 tied examples
        print(f"\nUp to 20 examples where scores were tied (out of {len(tied_results)} total ties):")
        print("=" * 80)
        for idx, result in enumerate(tied_results[:20]):
            print_detailed_example(result, idx, "Tied Score")
            
        # Print summary of wins
        print("\nWin Statistics:")
        print("=" * 80)
        print(f"Fixie wins: {len(fixie_better_results)} samples")
        print(f"Whisper+Fireworks wins: {len(fireworks_better_results)} samples")
        print(f"Ties: {len(tied_results)} samples")
        
        # Print average BLEU scores for wins
        if fixie_better_results:
            fixie_win_avg = np.mean([r['bleu_delta'] for r in fixie_better_results])
            print(f"Average BLEU delta when Fixie wins: {fixie_win_avg:.4f}")
        if fireworks_better_results:
            fireworks_win_avg = np.mean([abs(r['bleu_delta']) for r in fireworks_better_results])
            print(f"Average BLEU delta when Whisper+Fireworks wins: {fireworks_win_avg:.4f}")

if __name__ == "__main__":
    main()