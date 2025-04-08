import openai
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from dotenv import load_dotenv
import os
import io
import base64
from PIL import Image
import re

# === CONFIG ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPSIN_JAR_PATH = "opsin-cli-2.8.0-jar-with-dependencies.jar"

# Initialize the client with the new API format
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class ChemistryAssistant:
    def __init__(self):
        self.conversation_history = []
        self.current_chemical = None
        self.current_smiles = None
        self.step_by_step_info = None
        
    def add_to_history(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
    def extract_chemical_name(self, prompt):
        """Extract the chemical name from a user prompt."""
        system_prompt = """
        You are a chemistry assistant. Extract only the IUPAC or common name of a single chemical 
        compound from the user's prompt. Respond with just the chemical name, nothing else.
        If no specific chemical is mentioned, respond with "NO_CHEMICAL_FOUND".
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        chemical_name = response.choices[0].message.content.strip()
        return None if chemical_name == "NO_CHEMICAL_FOUND" else chemical_name
        
    def is_clarification_request(self, prompt):
        """Determine if the user is asking for clarification rather than a new chemical."""
        system_prompt = """
        Determine if the user is asking for clarification about previously discussed 
        information, or if they're asking about a new chemical compound.
        
        Respond with only one of:
        - "CLARIFICATION" - if they want more details or have questions about the current topic
        - "NEW_CHEMICAL" - if they're asking about a different chemical compound
        """
        
        # Include part of conversation history for context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add up to 3 most recent conversation turns for context
        if len(self.conversation_history) > 0:
            context_history = self.conversation_history[-min(6, len(self.conversation_history)):]
            messages.extend(context_history)
            
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        result = response.choices[0].message.content.strip()
        return result == "CLARIFICATION"
        
    def get_smiles_from_name(self, chemical_name):
        """Convert a chemical name to SMILES using OPSIN."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as infile, \
             tempfile.NamedTemporaryFile(mode='r', delete=False) as outfile:
            infile.write(chemical_name)
            infile.flush()

            command = [
                "java", "-jar", OPSIN_JAR_PATH,
                "-osmi", infile.name, outfile.name
            ]

            try:
                subprocess.run(command, check=True)
                outfile.seek(0)
                smiles = outfile.read().strip()
                return smiles
            except subprocess.CalledProcessError:
                print(f"❌ OPSIN couldn't process: {chemical_name}")
                return None
            finally:
                # Clean up temp files
                os.unlink(infile.name)
                os.unlink(outfile.name)
    
    def generate_molecule_image(self, smiles, highlight_substructures=None):
        """Generate an image of the molecule from SMILES, with optional highlighting."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
            
        # Add 2D coordinates
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        mol = Chem.RemoveHs(mol)
        
        # Create the drawing object
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
        drawer.drawOptions().addStereoAnnotation = True
        
        # Handle highlighting if specified
        if highlight_substructures:
            highlights = {}
            for i, substructure in enumerate(highlight_substructures):
                substructure_mol = Chem.MolFromSmarts(substructure["smarts"])
                if substructure_mol:
                    matches = mol.GetSubstructMatches(substructure_mol)
                    for match in matches:
                        for atom_idx in match:
                            highlights[atom_idx] = substructure["color"]
            
            # Apply highlights
            hit_atoms = list(highlights.keys())
            hit_bonds = []
            atom_colors = {atom: highlights[atom] for atom in hit_atoms}
            drawer.DrawMolecule(mol, highlightAtoms=hit_atoms, 
                               highlightBonds=hit_bonds,
                               highlightAtomColors=atom_colors)
        else:
            drawer.DrawMolecule(mol)
            
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        
        # Save to file and return the path
        output_path = "structure.png"
        with open(output_path, 'wb') as f:
            f.write(png_data)
            
        return output_path
        
    def get_step_by_step_explanation(self, chemical_name, smiles):
        """Generate a step-by-step explanation of the chemical."""
        system_prompt = """
        You are an expert chemistry teacher. Given a chemical compound name and its SMILES 
        representation, provide a comprehensive step-by-step explanation about the compound.
        
        Structure your response in these sections:
        1. Overview - Brief introduction to the chemical
        2. Molecular Structure - Detailed explanation of the structure (bonds, functional groups, etc.)
        3. Properties - Physical and chemical properties
        4. Applications - Common uses and importance
        5. Related Chemistry - How this connects to broader chemical concepts
        
        Format your response as markdown, with clear section headings.
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Chemical: {chemical_name}\nSMILES: {smiles}"}
            ]
        )
        
        return response.choices[0].message.content
    
    def get_clarification(self, user_prompt):
        """Generate a clarification response based on the user's follow-up question."""
        system_prompt = """
        You are an expert chemistry teacher. The user has asked for clarification about 
        a chemical compound you've been discussing. Based on the conversation history and 
        the current question, provide a clear, detailed response that addresses their specific 
        question. If they're confused about a particular concept, make sure to explain it in 
        simpler terms and relate it to concepts they might already understand.
        """
        
        # Include the conversation history for context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history
        context_history = self.conversation_history[-min(6, len(self.conversation_history)):]
        messages.extend(context_history)
        
        # Add the current prompt
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def extract_highlight_requests(self, text):
        """Extract functional groups or substructures to highlight from the explanation."""
        # This is a simplified version - in practice, you might use GPT to identify 
        # important substructures based on the text
        
        # Example: Look for common functional groups mentioned in the text
        functional_groups = [
            {"name": "hydroxyl", "smarts": "[OH]", "color": (1,0,0)},
            {"name": "carbonyl", "smarts": "[CX3]=[OX1]", "color": (0,0,1)},
            {"name": "carboxyl", "smarts": "[CX3](=[OX1])[OX2H1]", "color": (0,1,0)},
            {"name": "amino", "smarts": "[NX3;H2,H1;!$(NC=O)]", "color": (1,1,0)},
            {"name": "amide", "smarts": "[NX3][CX3](=[OX1])", "color": (1,0,1)}
        ]
        
        highlights = []
        for group in functional_groups:
            if re.search(r'\b' + re.escape(group["name"]) + r'\b', text, re.IGNORECASE):
                highlights.append(group)
                
        return highlights
        
    def handle_user_prompt(self, user_prompt):
        """Process a user prompt and generate an appropriate response."""
        if user_prompt.lower() in ['quit', 'exit', 'bye']:
            return "Goodbye! 👋"
            
        # Add the user prompt to conversation history
        self.add_to_history("user", user_prompt)
        
        # Check if this is a request for clarification
        if self.current_chemical and self.is_clarification_request(user_prompt):
            # User wants clarification on the current topic
            clarification = self.get_clarification(user_prompt)
            self.add_to_history("assistant", clarification)
            return clarification
            
        # This is a request for a new chemical
        chemical_name = self.extract_chemical_name(user_prompt)
        
        if not chemical_name:
            response = "I couldn't identify a specific chemical compound in your message. Could you please mention the chemical name more explicitly?"
            self.add_to_history("assistant", response)
            return response
            
        # Get SMILES representation
        smiles = self.get_smiles_from_name(chemical_name)
        
        if not smiles:
            response = f"I couldn't generate a SMILES representation for '{chemical_name}'. Please check the spelling or try a different chemical name."
            self.add_to_history("assistant", response)
            return response
            
        # Store the current chemical information
        self.current_chemical = chemical_name
        self.current_smiles = smiles
        
        # Generate the step-by-step explanation
        explanation = self.get_step_by_step_explanation(chemical_name, smiles)
        self.step_by_step_info = explanation
        
        # Extract substructures to highlight
        highlights = self.extract_highlight_requests(explanation)
        
        # Generate the molecular structure image
        image_path = self.generate_molecule_image(smiles, highlights)
        
        if not image_path:
            image_info = "❌ Could not generate molecular structure image."
        else:
            image_info = f"✅ Molecular structure image generated: {image_path}"
            
        # Construct the full response
        response = f"# {chemical_name}\n\n{image_info}\n\n{explanation}\n\nIs there a specific aspect of this chemical you'd like me to explain in more detail?"
        
        # Add to conversation history
        self.add_to_history("assistant", response)
        
        return response

def main():
    """Main function to run the Chemistry Assistant."""
    print("🧪 Welcome to the Interactive Chemistry Assistant! 🧪")
    print("Ask about any chemical compound, and I'll provide a visualization and step-by-step explanation.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("-" * 70)
    
    assistant = ChemistryAssistant()
    
    while True:
        user_input = input("\n🔍 What would you like to learn about? ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for using the Chemistry Assistant! Goodbye! 👋")
            break
            
        try:
            response = assistant.handle_user_prompt(user_input)
            print(f"\n{response}")
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()