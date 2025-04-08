import openai
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import Draw
from dotenv import load_dotenv
import os

# === CONFIG ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPSIN_JAR_PATH = "opsin-cli-2.8.0-jar-with-dependencies.jar"

# Initialize the client with the new API format
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_chemical_name_from_gpt(prompt):
    system_prompt = "You are a chemistry assistant. Extract only the IUPAC name of a single chemical compound from the user's prompt. Respond with just the chemical name, nothing else."
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo" if needed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def get_smiles_from_name(chemical_name):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as infile, \
         tempfile.NamedTemporaryFile(mode='r', delete=False) as outfile:
        infile.write(chemical_name)
        infile.flush()

        command = [
            "java", "-jar", OPSIN_JAR_PATH,
            "-osmi", infile.name, outfile.name
        ]

        subprocess.run(command, check=True)
        outfile.seek(0)
        smiles = outfile.read().strip()
        return smiles

def draw_smiles(smiles, output_file="structure.png"):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        Draw.MolToFile(mol, output_file)
        print(f"✅ Structure image saved to {output_file}")
    else:
        print("❌ Could not parse SMILES.")

def main():
    print("Welcome to the Chemical Structure Generator!")
    print("Type 'quit' or 'exit' to stop the program.")
    print("------------------------------------------")
    
    while True:
        user_prompt = input("\nDescribe a chemical (natural language): ")
        
        # Check if user wants to quit
        if user_prompt.lower() in ['quit', 'exit']:
            print("Goodbye! 👋")
            break
            
        try:
            chem_name = get_chemical_name_from_gpt(user_prompt)
            print(f"🔍 GPT identified: {chem_name}")

            smiles = get_smiles_from_name(chem_name)
            print(f"🧪 SMILES: {smiles}")

            draw_smiles(smiles)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with a different chemical description.")

if __name__ == "__main__":
    main()

