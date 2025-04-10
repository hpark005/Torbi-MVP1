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
import uuid
import json
import time

# === CONFIG ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPSIN_JAR_PATH = "opsin-cli-2.8.0-jar-with-dependencies.jar"
MODEL = "gpt-4"  # Using GPT-4 for better chemistry understanding

# Initialize the client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class ImprovedChemistryAssistant:
    def __init__(self):
        self.conversation_history = []
        self.generated_images = {}  # Track generated molecule images by name
        
        # Create molecules directory if it doesn't exist
        self.molecules_dir = "molecules"
        os.makedirs(self.molecules_dir, exist_ok=True)
        
        # Initialize common chemical data
        self.init_chemical_data()

    def init_chemical_data(self):
        """Initialize mappings for functional groups and common chemicals"""
        # Common functional groups with example compounds
        self.functional_group_examples = {
            "aldehyde": "formaldehyde",
            "ketone": "acetone",
            "alcohol": "ethanol",
            "acid": "acetic acid",
            "carboxylic acid": "acetic acid",
            "amine": "methylamine",
            "ester": "ethyl acetate",
            "ether": "diethyl ether",
            "alkene": "ethene",
            "alkyne": "ethyne",
            "alkane": "methane",
            "aromatic": "benzene",
            "benzene ring": "benzene",
            "phenyl": "benzene",
            "carboxyl": "acetic acid",
            "hydroxyl": "ethanol",
            "carbonyl": "acetone",
            "amino": "methylamine",
            "nitro": "nitromethane",
            "sulfide": "dimethyl sulfide",
            "thiol": "methanethiol",
            "amide": "acetamide"
        }
        
        # Common chemicals with direct SMILES notation
        self.common_chemicals = {
            "water": "O",
            "carbon dioxide": "O=C=O",
            "hydrogen peroxide": "OO",
            "ammonia": "N",
            "methane": "C",
            "ethane": "CC",
            "propane": "CCC",
            "butane": "CCCC",
            "pentane": "CCCCC",
            "hexane": "CCCCCC",
            "heptane": "CCCCCCC",
            "octane": "CCCCCCCC",
            "nonane": "CCCCCCCCC",
            "decane": "CCCCCCCCCC",
            "methanol": "CO",
            "ethanol": "CCO",
            "acetic acid": "CC(=O)O",
            "formic acid": "C(=O)O",
            "glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
            "benzene": "c1ccccc1",
            "toluene": "Cc1ccccc1",
            "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "paracetamol": "CC(=O)Nc1ccc(O)cc1",
            "sulfuric acid": "O=S(=O)(O)O",
            "nitric acid": "O=[N+]([O-])O",
            "hydrochloric acid": "Cl",  # Not perfect, but OPSIN struggles with this
            "ethylene": "C=C",
            "acetylene": "C#C",
            "acetone": "CC(=O)C",
            "methyl alcohol": "CO",
            "formaldehyde": "C=O"
        }
        
        # Categories of molecules for conceptual queries
        self.molecule_categories = {
            "acid": ["sulfuric acid", "nitric acid", "acetic acid", "formic acid", "hydrochloric acid"],
            "base": ["ammonia", "sodium hydroxide"],
            "alcohol": ["methanol", "ethanol"],
            "hydrocarbon": ["methane", "ethane", "propane", "butane", "benzene", "toluene"],
            "organic": ["acetone", "ethanol", "acetic acid", "benzene", "toluene", "glucose"],
            "inorganic": ["water", "carbon dioxide", "ammonia", "sulfuric acid", "nitric acid"],
            "polar": ["water", "ethanol", "acetone"],
            "nonpolar": ["methane", "ethane", "benzene"],
            "common": ["water", "methane", "ethanol", "carbon dioxide", "glucose", "benzene"]
        }
        
    def add_to_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        # Keep history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def get_smiles_from_name(self, chemical_name):
        """Convert chemical name to SMILES using direct mapping or OPSIN."""
        # Clean up the chemical name
        original_name = chemical_name
        chemical_name = chemical_name.strip().lower()
        
        # Try direct SMILES mapping first (faster and more reliable for common chemicals)
        if chemical_name in self.common_chemicals:
            return self.common_chemicals[chemical_name]
            
        # If it's a functional group, use the example compound
        if chemical_name in self.functional_group_examples:
            # For functional groups, use the representative example
            representative = self.functional_group_examples[chemical_name]
            print(f"Using {representative} as an example of {chemical_name}")
            
            # Check if the representative is in our common chemicals
            if representative in self.common_chemicals:
                return self.common_chemicals[representative]
                
            # Otherwise, use the representative name with OPSIN
            chemical_name = representative
            
        # Use OPSIN for other chemical names
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as infile, \
             tempfile.NamedTemporaryFile(mode='r', delete=False) as outfile:
            infile.write(chemical_name)
            infile.flush()

            command = [
                "java", "-jar", OPSIN_JAR_PATH,
                "-osmi", infile.name, outfile.name
            ]

            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                outfile.seek(0)
                smiles = outfile.read().strip()
                if not smiles:
                    print(f"❌ OPSIN returned empty SMILES for: {chemical_name}")
                    return None
                return smiles
            except subprocess.CalledProcessError as e:
                print(f"❌ OPSIN couldn't process: {chemical_name}")
                print(f"Error output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No error output'}")
                return None
            finally:
                # Clean up temporary files
                try:
                    os.unlink(infile.name)
                    os.unlink(outfile.name)
                except:
                    pass

    def generate_molecule_image(self, smiles, chemical_name, highlight_substructures=None):
        """Generate molecule image from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"❌ Invalid SMILES: {smiles}")
            return None

        # Generate 2D coordinates
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        mol = Chem.RemoveHs(mol)

        # Create the drawing
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)  # Larger size for better visibility
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().bondLineWidth = 2
        drawer.drawOptions().useBWAtomPalette()  # More consistent coloring

        # Handle highlighting if requested
        if highlight_substructures:
            highlights = {}
            for i, substructure in enumerate(highlight_substructures):
                substructure_mol = Chem.MolFromSmarts(substructure["smarts"])
                if substructure_mol:
                    matches = mol.GetSubstructMatches(substructure_mol)
                    for match in matches:
                        for atom_idx in match:
                            highlights[atom_idx] = substructure["color"]

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

        # Generate a unique filename
        safe_name = "".join(c if c.isalnum() else "_" for c in chemical_name) if chemical_name else "molecule"
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{safe_name}_{unique_id}.png"
        
        output_path = os.path.join(self.molecules_dir, filename)
        
        try:
            with open(output_path, 'wb') as f:
                f.write(png_data)
            # Verify the file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                print(f"❌ Failed to save diagram for {chemical_name}")
                return None
        except Exception as e:
            print(f"❌ Error saving diagram for {chemical_name}: {str(e)}")
            return None

    def generate_diagrams_for_chemicals(self, chemicals):
        """Generate diagrams for a list of specific chemical names."""
        results = []
        
        for chemical_name in chemicals:
            if not chemical_name.strip():
                continue
                
            # Skip if we've already processed this chemical
            if chemical_name.lower() in self.generated_images:
                print(f"✓ Already generated diagram for {chemical_name}")
                results.append({
                    "name": chemical_name,
                    "smiles": self.generated_images[chemical_name.lower()]["smiles"],
                    "image_path": self.generated_images[chemical_name.lower()]["image_path"],
                    "status": "already_exists"
                })
                continue
                
            smiles = self.get_smiles_from_name(chemical_name)
            if not smiles:
                print(f"❌ Could not generate SMILES for {chemical_name}")
                continue
                
            image_path = self.generate_molecule_image(smiles, chemical_name)
            
            if image_path:
                # Store with lowercase key for case-insensitive lookups
                self.generated_images[chemical_name.lower()] = {
                    "smiles": smiles,
                    "image_path": image_path
                }
                results.append({
                    "name": chemical_name,
                    "smiles": smiles,
                    "image_path": image_path,
                    "status": "success"
                })
                print(f"✅ Generated diagram for {chemical_name} at {image_path}")
            else:
                print(f"❌ Failed to generate diagram for {chemical_name}")
                
        return results

    def get_llm_response_with_diagrams(self, user_prompt):
        """Get a response from the LLM first, then generate any diagrams it suggests."""
        system_prompt = """
        You are a helpful chemistry assistant who can discuss chemistry topics and create molecular structure diagrams.
        
        When responding to a user query about chemistry:
        1. Provide a clear, educational answer first
        2. If specific molecules are relevant to the answer, mention them so they can be visualized
        3. At the end of your response, add a section called "DIAGRAMS" where you list all chemical compounds 
           that would benefit from visualization (max 3)
        
        Example:
        User: "Tell me about alcohols"
        
        Your response might be:
        "Alcohols are organic compounds characterized by the presence of a hydroxyl (-OH) group attached to a carbon atom. 
        They have the general formula R-OH, where R represents an alkyl group. Alcohols are classified as primary, 
        secondary, or tertiary depending on how many carbon atoms are bonded to the carbon with the hydroxyl group.
        
        For example, ethanol (CH3CH2OH) is a common primary alcohol used in beverages and as a fuel additive. 
        Isopropyl alcohol (CH3CHOHCH3) is a secondary alcohol commonly used as a disinfectant.
        
        DIAGRAMS:
        ethanol
        isopropyl alcohol
        methanol"
        
        CRITICAL INSTRUCTIONS:
        - Never apologize for not being able to show diagrams - you CAN create them!
        - Always put the list of molecules to visualize at the end under the heading "DIAGRAMS:"
        - List one chemical per line with no additional formatting or description
        - Only list specific molecular compounds (not general types like "acids" or "alkanes")
        - List a maximum of 3 specific chemicals to visualize
        - For questions like "show me an acidic molecule", suggest a specific acid molecule
        """

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        if len(self.conversation_history) > 0:
            # Only add the last 5 exchanges (10 messages) to keep context manageable
            messages.extend(self.conversation_history[-min(10, len(self.conversation_history)):])
        
        # Add the current user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7  # Slightly higher temperature for more varied responses
        )
        
        full_response = response.choices[0].message.content
        
        # Extract the list of chemicals to visualize
        chemicals_to_visualize = []
        if "DIAGRAMS:" in full_response:
            # Split at the DIAGRAMS: marker and get the part after it
            diagrams_section = full_response.split("DIAGRAMS:", 1)[1].strip()
            
            # Extract the individual chemical names, clean them up
            chemicals_to_visualize = [
                chem.strip() for chem in diagrams_section.split("\n") 
                if chem.strip() and not chem.strip().startswith("-")
            ]
            
            # Limit to first 3 chemicals
            chemicals_to_visualize = chemicals_to_visualize[:3]
            
            # Remove the DIAGRAMS section from the response
            main_response = full_response.split("DIAGRAMS:", 1)[0].strip()
        else:
            main_response = full_response
        
        # Generate the diagrams
        diagram_results = self.generate_diagrams_for_chemicals(chemicals_to_visualize)
        
        # Create an enhanced response that includes diagram information
        enhanced_response = main_response
        
        if diagram_results:
            # Generate a summary of the diagrams created
            diagram_info = []
            for result in diagram_results:
                if result["status"] in ["success", "already_exists"]:
                    diagram_info.append(f"{result['name']} ({result['image_path']})")
            
            if diagram_info:
                enhanced_response += "\n\nI've generated the following molecular structure diagrams:\n"
                enhanced_response += "\n".join(f"- {info}" for info in diagram_info)
        
        return enhanced_response

    def handle_user_prompt(self, user_prompt):
        """Process user prompt - first get LLM response, then generate diagrams."""
        # Special commands
        if user_prompt.lower() in ['quit', 'exit', 'bye']:
            return "Thank you for using the Chemistry Assistant! Goodbye! 👋"
        
        # Add to conversation history
        self.add_to_history("user", user_prompt)
        
        # Get response and generate diagrams
        response = self.get_llm_response_with_diagrams(user_prompt)
        
        # Add the response to conversation history
        self.add_to_history("assistant", response)
        
        return response

def main():
    print("🧪 Welcome to the Improved Chemistry Assistant! 🧪")
    print("Ask me anything about chemistry. I'll explain concepts and generate molecule diagrams.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("-" * 70)

    assistant = ImprovedChemistryAssistant()

    while True:
        user_input = input("\n🔍 > ")

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for using the Chemistry Assistant! Goodbye! 👋")
            break

        try:
            response = assistant.handle_user_prompt(user_input)
            print(f"\n{response}")
        except Exception as e:
            import traceback
            print(f"❌ An error occurred: {str(e)}")
            print(traceback.format_exc())
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()