def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """

    merge=[]
    vocab=[]

    words=[list(s.encode('utf-8')) for s in corpus]
    token_bytes={i: (i,) for i in range(256)}
    
    next_id=256
    num_merge=vocab_size-256

    for _ in range(num_merge):

        pair_cnt = {}

        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_cnt[pair] = pair_cnt.get(pair, 0) + 1

        if not pair_cnt:
            break

        best_pair = max(
            pair_cnt.keys(),
            key=lambda p: (pair_cnt[p],(token_bytes[p[0]], token_bytes[p[1]]))
        )

        left, right = best_pair
        new_id = next_id
        next_id += 1

        merge.append([left, right, new_id])
        
        new_byte=token_bytes[left]+token_bytes[right]
        token_bytes[new_id]=new_byte
        vocab.append([new_id, list(new_byte)])

        new_words=[]

        for word in words:
            new_word=[]
            i=0
            while (i < len(word)):
                if i < len(word)-1 and word[i]==left and word[i+1]==right:
                    new_word.append(new_id)
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1

            new_words.append(new_word)

        words=new_words

    return {
        "vocab": vocab, 
        "merges": merge
    }
    pass
