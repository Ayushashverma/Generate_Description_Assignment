Model choice and why:
I used the open-source model "google/flan-t5-small" because it is a small instruction-tuned model that generates short, clear descriptions for column headers. It runs locally on CPU and can also use GPU if available. The script automatically falls back to "gpt2" if flan-t5-small is not available.

How to run:
1. Create a virtual environment and install dependencies:
   pip install -r requirements.txt

2. Run with a CSV file:
   python generate.py data.csv

   (Example: data.csv should have a header row like:
    Invoice_ID, Vendor_Name, Amount, Payment_Date)

   The script will print descriptions to the console and save them into output.txt.

3. Run without a CSV:
   python generate.py
   (This will use built-in sample headers.)

Challenges:
The main challenge was running Hugging Face models locally with limited resources. I solved this by choosing flan-t5-small (lightweight, instruction-tuned) and adding GPT-2 as a fallback model in case of compatibility issues.
