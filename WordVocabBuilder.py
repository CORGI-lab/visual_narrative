def preProBuildWordVocab(sentence_iterator, word_count_threshold=3):
    # borrowed this function from NeuralTalk
    print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))

    word_counts = {}
    nsents = 0

    for sent in sentence_iterator:
        nsents += 1
        tmp_sent = sent.lower().split(' ')
        if '' in tmp_sent:
            tmp_sent.remove('')

        for w in tmp_sent:
            if w !='':
                word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<bos>'
    ixtoword[1] = '<eos>'
    ixtoword[2] = '<pad>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<bos>'] = 0
    wordtoix['<eos>'] = 1
    wordtoix['<pad>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 4
        ixtoword[idx+4] = w

    word_counts['<eos>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<pad>'] = nsents
    word_counts['<unk>'] = nsents


    return wordtoix, ixtoword
