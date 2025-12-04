import json
import sys

# Read the notebook
notebook_path = 'EGEAT_Colab_Notebook.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"Error: {notebook_path} not found!")
    sys.exit(1)

# Find the cell with ensemble_snapshots issue
fixed = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        # Join all source lines to check content
        full_source = ''.join(cell['source'])
        
        # Check if this cell has the problematic code
        if 'snapshot = torch.load(snapshot_path, map_location=device)' in full_source and 'ensemble_snapshots.append(snapshot)' in full_source:
            print(f"Found problematic code in cell {i}")
            
            # Replace the problematic code block
            old_pattern = 'snapshot = torch.load(snapshot_path, map_location=device)'
            new_code = '''# Create a new model instance and load the state_dict
                snapshot_model = model_class(
                    input_channels=config.model.input_channels,
                    num_classes=config.model.num_classes
                )
                snapshot_model.load_state_dict(torch.load(snapshot_path, map_location=device))
                snapshot_model = snapshot_model.to(device)
                snapshot_model.eval()  # Set to eval mode'''
            
            # Reconstruct the source lines
            new_source_lines = []
            lines = cell['source']
            j = 0
            while j < len(lines):
                line = lines[j]
                
                # Check if this line contains the problematic code
                if old_pattern in line:
                    # Get the indentation from the line
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    
                    # Add the replacement code with proper indentation
                    for new_line in new_code.split('\n'):
                        new_source_lines.append(indent_str + new_line.strip() + '\n')
                    
                    # Skip the next line if it's ensemble_snapshots.append(snapshot)
                    j += 1
                    if j < len(lines) and 'ensemble_snapshots.append(snapshot)' in lines[j]:
                        # Replace snapshot with snapshot_model
                        new_line = lines[j].replace('snapshot)', 'snapshot_model)')
                        new_source_lines.append(new_line)
                        j += 1
                    continue
                else:
                    new_source_lines.append(line)
                    j += 1
            
            cell['source'] = new_source_lines
            print(f"Fixed cell {i}")
            fixed = True
            break

if not fixed:
    print("Warning: Could not find the problematic code. The notebook may already be fixed or the code structure is different.")
    sys.exit(1)

# Write the fixed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook {notebook_path} fixed successfully!")
