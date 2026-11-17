import re # preserve newline characters
import os # get local path name
# claude helped figure out the os instead of the glob stuff, and the reconstruct_unprocessed_sample function.

# this function assumes the only pre-processing is space before and after punctuation.
# anything like a change to quote type, etc. is not checked for / cleaned.
def reconstruct_unprocessed_sample(original_path, brat_path):
    with open(original_path, "r") as f:
        original = f.read()
    with open(brat_path, "r") as f:
        processed = f.read()

    # Count how many tokens the brat sample contains
    processed_char_count = len(re.sub(r'\s+','',processed))

    # this will return the index of the original text each non white-space character lies in
    substantive_char_indices = [i for i, c in enumerate(original) if not c.isspace()]

    # Split the original into tokens and rejoin the same number
    cutoff = substantive_char_indices[processed_char_count - 1] + 1
    
    return original[:cutoff]

# Paths anchored to this script's location (take2/code). After the repo reorg
# the inputs and outputs all live under BookNLP_Litbank/data/.
HERE         = os.path.dirname(os.path.abspath(__file__))
LITBANK_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ORIGINAL_DIR    = os.path.join(LITBANK_ROOT, "data", "raw", "texts")
BRAT_DIR        = os.path.join(LITBANK_ROOT, "data", "raw", "brat_texts")
RECONSTRUCT_DIR = os.path.join(LITBANK_ROOT, "data", "intermediate","reconstructed_samples")


def main() :
    os.makedirs(RECONSTRUCT_DIR, exist_ok=True)
    for filename in os.listdir(ORIGINAL_DIR):
        name = filename.replace(".txt","")
        original_path = os.path.join(ORIGINAL_DIR, filename)
        processed_path = os.path.join(BRAT_DIR, name + "_brat.txt")

        if os.path.exists(processed_path):
             sample = reconstruct_unprocessed_sample(original_path, processed_path)
             with open(os.path.join(RECONSTRUCT_DIR, f"{name}_sampled.txt"), "w") as f:
                  f.write(sample)


if __name__ == "__main__":
     main()
