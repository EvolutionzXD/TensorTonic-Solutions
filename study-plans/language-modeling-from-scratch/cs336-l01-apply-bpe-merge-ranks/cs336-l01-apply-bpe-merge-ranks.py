def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns token IDs after applying the ordered merge rules.
    """
    word= list(text.encode('utf-8'))

    for merge in merges:
        [left, right, newID] = merge 
        new_word=[]
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i+1] == right:
                new_word.append(newID)
                i+=2
            else:
                new_word.append(word[i])
                i+=1

        word=new_word
    return word    
    pass

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns the text reconstructed from the token bytes.
    """
    word=[]
    token_bytes = {i:(i, ) for i in range(256)}

    if isinstance(vocab, dict):
        for k, v in vocab.items():
            token_bytes[int(k)] = v
    else:
        for entry in vocab:
            token_bytes[entry[0]] = entry[1]

    for id in ids:
        word.extend(token_bytes[id])

    return bytes(word).decode('utf-8')
    pass
