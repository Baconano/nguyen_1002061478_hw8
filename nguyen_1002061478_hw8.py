import numpy as np

def load_data(file_path):
    sentences = []
    current_sentence = []
    all_words = set()
    all_tags = set()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                parts = line.rsplit('/', 1)
                if len(parts) == 2:
                    word, tag = parts[0].lower(), parts[1]
                    current_sentence.append((word, tag))
                    all_words.add(word)
                    all_tags.add(tag)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Check your directory!")
        
    return sentences, list(all_words), list(all_tags)

class EMPosTagger:
    def __init__(self, tags, vocab):
        self.tags = tags
        self.vocab = vocab
        self.N = len(tags)
        self.V = len(vocab)
        self.tag2idx = {tag: i for i, tag in enumerate(tags)}
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        
        
        self.pi = np.full(self.N, 1.0 / self.N)
        self.A = np.random.dirichlet([10.0]*self.N, self.N) 
        self.B = np.full((self.N, self.V), 1e-6)
        for s in train_raw[:100]: 
            for w, t in s:
                self.B[self.tag2idx[t], word2idx[w.lower()]] += 1
                self.B /= np.sum(self.B, axis=1, keepdims=True)

    def forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, obs[0]]
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = alpha[t-1].dot(self.A[:, j]) * self.B[j, obs[t]]
            norm = np.sum(alpha[t])
            if norm > 0: alpha[t] /= norm
        return alpha

    def backward(self, obs):
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[T-1] = 1.0
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = (self.A[i, :] * self.B[:, obs[t+1]]).dot(beta[t+1])
            norm = np.sum(beta[t])
            if norm > 0: beta[t] /= norm
        return beta

    def train(self, corpus, iterations=5):
        for i in range(iterations):
            new_A_num = np.zeros((self.N, self.N))
            new_A_den = np.zeros(self.N)
            new_B_num = np.zeros((self.N, self.V))
            new_B_den = np.zeros(self.N)
            total_ll = 0

            for obs in corpus:
                if not obs: continue
                alpha = self.forward(obs)
                total_ll += np.log(np.sum(alpha[-1]) + 1e-10)
                beta = self.backward(obs)
                
                
                gamma = (alpha * beta)
                gamma /= (np.sum(gamma, axis=1, keepdims=True) + 1e-9)
                
                for t in range(len(obs) - 1):
                    num = self.A * self.B[:, obs[t+1]] * alpha[t][:, None] * beta[t+1]
                    new_A_num += num / (np.sum(num) + 1e-9)
                
                for t, word_idx in enumerate(obs):
                    new_B_num[:, word_idx] += gamma[t]
                
                new_A_den += np.sum(gamma[:-1], axis=0)
                new_B_den += np.sum(gamma, axis=0)

            
            self.A = new_A_num / (new_A_den[:, None] + 1e-9)
            self.B = new_B_num / (new_B_den[:, None] + 1e-9)
            print(f"Iteration {i+1} | Log-Likelihood: {total_ll:.2f}")

if __name__ == "__main__":
    
    train_file = 'WSJ_02-21.pos' 
    test_file = 'WSJ_24.pos' 

    print("Loading data...")
    train_raw, vocab, tags = load_data(train_file)
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    
    train_corpus = [[word2idx[w] for w, t in s] for s in train_raw[:1000]]
    
    tagger = EMPosTagger(tags, vocab)
    tagger.train(train_corpus, iterations=3) 
    
    print("\nEvaluating on Test Data...") 
    test_raw, _, _ = load_data(test_file)
    correct, total = 0, 0
    for sentence in test_raw:
        obs = [word2idx.get(w.lower(), 0) for w, t in sentence]
        true_tags = [t for w, t in sentence]
        for t, word_idx in enumerate(obs):
            pred_tag = tags[np.argmax(tagger.B[:, word_idx])]
            if pred_tag == true_tags[t]:
                correct += 1
            total += 1
    
    print(f"Final Accuracy: {(correct/total)*100:.2f}%")