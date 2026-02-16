import json
import os
import re
from pathlib import Path


def load_replacements(terms_map_path: str) -> tuple[dict, dict]:
    with open(terms_map_path, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)
    
    full_replacements = {}
    first_word_replacements = {}
    
    sorted_keys = sorted(terms_map.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        value = terms_map[key]
        full_replacements[key] = value
        
        first_word = key.split()[0]
        first_word_replacements[first_word] = value
    
    return full_replacements, first_word_replacements


def remove_lines_starting_with_asterisk(content: str) -> str:
    lines = content.split('\n')
    filtered_lines = [line for line in lines if not line.startswith('*')]
    return '\n'.join(filtered_lines)


def apply_replacements(content: str, full_replacements: dict, first_word_replacements: dict) -> str:
    result = content
    
    for key, value in full_replacements.items():
        result = result.replace(key, value)
    
    for first_word, value in first_word_replacements.items():
        result = result.replace(first_word, value)
    
    return result


def process_files(source_dir: str, output_dir: str, full_replacements: dict, first_word_replacements: dict):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    for file_path in source_path.glob('*'):
        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = remove_lines_starting_with_asterisk(content)
            
            content = apply_replacements(content, full_replacements, first_word_replacements)
            
            output_file = output_path / file_path.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Processed: {file_path.name} -> {output_file.name}")


def main():
    script_dir = Path(__file__).parent
    source_dir = script_dir / 'source_base'
    output_dir = script_dir / 'knowledge_base'
    terms_map_path = script_dir / 'terms_map.json'
    
    full_replacements, first_word_replacements = load_replacements(terms_map_path)
    
    print(f"Loaded {len(full_replacements)} full replacements")
    print(f"Loaded {len(first_word_replacements)} first word replacements")
    
    process_files(str(source_dir), str(output_dir), full_replacements, first_word_replacements)
    
    print("Done!")


if __name__ == '__main__':
    main()
