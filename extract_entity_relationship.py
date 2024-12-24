import torch
from tqdm import tqdm
import re


def extract_elements_from_chunks(LLM, chunks, extraction_prompt, max_gleanings=3):
    CONTINUE_PROMPT = (
        "MANY entities and relationships were missed in the last extraction. "
        "Remember to ONLY emit entities that match any of the previously extracted types. "
        "Add them below using the same format:\n"
    )
    LOOP_PROMPT = (
        "It appears some entities and relationships may have still been missed. "
        "Answer YES | NO if there are still entities or relationships that need to be added.\n"
    )
    elements = []
    pbar = tqdm(enumerate(chunks), total=len(chunks))
    
    for index, chunk in pbar:
        pbar.set_description(f"Processing chunk {index+1} of {len(chunks)}")
        
        # Initialize the conversation with the initial extraction prompt
        current_prompt = extraction_prompt.replace('{input_text}', chunk)
        messages = [{"role": "user", "content": current_prompt}]
        
        try:
            with torch.no_grad():  # Prevent storing computation graph
                response = LLM(messages, max_new_tokens=1000, 
                              pad_token_id=LLM.tokenizer.eos_token_id)
        except Exception as e:
            print(f"Error during initial extraction for chunk {index+1}: {e}")
            elements.append("")
            continue  # Skip to the next chunk
        
        # Extract the assistant's reply (adjust according to your LLM's response format)
        try:
            # Example for OpenAI-like response structure
            # entities_and_relations = response['choices'][0]['message']['content'].strip()
            # Adjust the above line based on your LLM's actual response format
            entities_and_relations = response[-1]['generated_text'][-1]['content'].strip()
        except (IndexError, KeyError) as e:
            print(f"Error parsing response for chunk {index+1}: {e}")
            elements.append("")
            continue
        
        results = entities_and_relations
        messages.append({"role": "assistant", "content": entities_and_relations})
        
        # Begin multi-glean checking loop
        for gleaning in range(max_gleanings):
            try:
                # Append CONTINUE_PROMPT to prompt for more entities
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
                
                with torch.no_grad():
                    response = LLM(messages, max_new_tokens=1000,
                                  pad_token_id=LLM.tokenizer.eos_token_id)
                
                # Extract new entities
                new_entities = response[-1]['generated_text'][-1]['content'].strip()
                results += '\n' + new_entities
                messages.append({"role": "assistant", "content": new_entities})
                
                # Check if this is the last iteration
                if gleaning >= max_gleanings - 1:
                    break
                
                # Append LOOP_PROMPT to check for remaining entities
                messages.append({"role": "user", "content": LOOP_PROMPT})
                
                with torch.no_grad():
                    response = LLM(messages, max_new_tokens=10,
                                  pad_token_id=LLM.tokenizer.eos_token_id)
                
                loop_response = response[-1]['generated_text'][-1]['content'].strip().upper()
                
                # Append the loop response
                messages.append({"role": "assistant", "content": loop_response})

 
            except Exception as e:
                print(f"Error during gleaning {gleaning+1} for chunk {index+1}: {e}")
                break  # Exit the gleaning loop on error
            
            finally:
                # Clear CUDA cache after each gleaning to free memory
                torch.cuda.empty_cache()
        
        # Append the final results for the chunk
        elements.append(results)
        
        # Clear messages list to free memory
        del messages
        torch.cuda.empty_cache()
    return elements

def parse_line(text):
    # Remove the leading '('
    text = re.sub(r'^\(', '', text)
    
    # Remove the trailing patterns
    text = re.sub(r'(\)\#\#|<\|COMPLETE\|>)$', '', text)
    
    # Split the string using the updated regular expression
    components = re.split(r'<\|>', text)
    
    # Remove surrounding quotes and any extra parentheses
    components = [component.strip('"').strip(')') for component in components]
    
    return components 

def collect_elements_relationship(elements):
    group_dic = {}
    for idx, group in enumerate(elements):
        entities = group.split('\n')
        node_dic = {}
        edge_dic = {}
        for entity in entities:
            if '##' in entity or '<|COMPLETE|>' in entity:
                elements = parse_line(entity)
                entity_type = elements[0]
                if entity_type == "entity":
                    node_name = elements[1]
                    node_type = elements[2]
                    node_content = elements[3]
                    
                    node_dic[node_name] = {'type':node_type, 'node_content':node_content}
                elif entity_type == 'relationship':
                    source_node = elements[1]
                    target_node = elements[2]
                    relationship_type = elements[3]
                    strength = elements[4]
                    edge_dic[f"{source_node}-{target_node}"] = {'source':source_node, 'target':target_node, 'relationship':relationship_type, "strength":strength}

        group_dic[idx] = {'node':node_dic, 'edge':edge_dic}
    return group_dic






