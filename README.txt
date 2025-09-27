CSV Header Description Generator
Model Choice

Primary: google/flan-t5-small – a lightweight, instruction-tuned model that generates short, meaningful descriptions for CSV headers.

Fallback: gpt2 – used automatically if flan-t5-small is unavailable.

Reason: Runs locally on CPU/GPU and works efficiently with minimal resources.

How to Run

Setup Environment

pip install -r requirements.txt


Run with a CSV File

python generate.py data.csv


Example CSV header row:

Invoice_ID, Vendor_Name, Amount, Payment_Date


Output: Descriptions printed in console and saved to output.txt.

Run Without a CSV

python generate.py


Uses built-in sample headers if no CSV is provided.

Challenges & Solutions

Running Hugging Face models locally with limited resources was challenging.

Solution: Used flan-t5-small for lightweight performance and added GPT-2 as a fallback to ensure reliable output.
