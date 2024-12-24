


def entity_type_extraction(LLM, prompt, document, domain_name):
    prompt = prompt.replace('{domain}', domain_name)
    prompt = prompt.replace('{document}', document)

    messages=[
        {"role": "user", "content": prompt},
    ]    
    response = LLM(messages, max_new_tokens=100, 
                   pad_token_id=LLM.tokenizer.eos_token_id)
    entities_types = response[0]['generated_text'][-1]['content']
    return entities_types.split('\n\n')[1]