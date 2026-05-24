import os

def main():
    filepath = 'frontend/streamlit_app.py'
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find the start line
    start_idx = -1
    for i, line in enumerate(lines):
        if '── Patient Input Form ──' in line:
            start_idx = i
            break
            
    if start_idx == -1:
        print("Could not find start index")
        return
        
    tab_code = [
        '    tab1, tab2 = st.tabs(["Patient Evaluation", "Model Monitoring"])',
        '    with tab1:'
    ]
    
    new_lines = lines[:start_idx] + tab_code
    
    for line in lines[start_idx:]:
        if not line.strip():
            new_lines.append('')
        elif line.startswith('    '):
            new_lines.append('        ' + line[4:])
        else:
            new_lines.append('        ' + line)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
        
    print("Injected successfully!")

if __name__ == '__main__':
    main()
