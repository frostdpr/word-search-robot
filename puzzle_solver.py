from collections import defaultdict

class PuzzleSolver:

    def __init__(self, rows, cols, puzzle, bank):
        self.r = rows
        self.c = cols
        self.puzzle = puzzle
        self.min_word_len = 23
        self.max_word_len = 0
        self.bank = self.load_word_bank(bank)
    
    #Takes in a list of strings to insert into wordbank dictionary
    def load_word_bank(self, bank):
        ret = defaultdict(lambda: None)

        for word in bank:
            self.min_word_len = min(self.min_word_len, len(word))
            self.max_word_len = max(self.max_word_len, len(word))
            ret[word] = word

        return ret

    #switch case for direction
    def get_word_in_grid(self, startRow, startCol, d, l):
        word = ""
        pos, r, c = 0, startRow, startCol

        for i in range(l):
            #if current row or column is out of bounds, break
            if c >= self.c or r >= self.r or r < 0 or c < 0:
                return "", r, c

            #pos += 1
            #print(self.puzzle[r][c])
            word += self.puzzle[r][c]

            if d == 0:
                r -= 1
            elif d == 1:
                r -=1
                c += 1
            elif d == 2:
                c += 1
            elif d == 3:
                r += 1
                c += 1
            elif d == 4:
                r += 1
            elif d == 5:
                r += 1
                c -= 1
            elif d == 6:
                c -= 1
            elif d == 7:
                r -= 1
                c -= 1
            
        return word, r, c

    def solve(self):
        word_count = 0
        not_found = list(self.bank.keys())
        found = []
        for i in range(self.r):
            for j in range(self.c):
                for d in range(8):
                    for length in range(self.min_word_len, self.max_word_len + 1):
                        word, r, c = self.get_word_in_grid(i, j, d, length)

                        if self.bank[word] == word:
                            found.append(([[[i],[j]]],[[[r],[c]]]))
                            print(i, j, word)
                            try:
                                not_found.remove(word)
                            except:
                                pass

                            word_count += 1

        print('words found', word_count)
        return not_found, found

    def potential_words_solve(self, incorrect_words, potential_words):

        retry_bank = []
        _inbounds = True
        i = 0
        found = []

        while _inbounds:
            retry_bank = []
            _inbounds = False
            for words in potential_words:

                if i >= len(words):
                    continue

                _inbounds = True
                retry_bank.append(words[i])

            self.bank = self.load_word_bank(retry_bank)
            incorrect_retry, found = self.solve()

            for word in retry_bank:
                # if the word was found, delete it from potential words list
                # and delete from the incorrect words list
                if word not in incorrect_retry:
                    index = self.get_index(potential_words, word)
                    incorrect_words.remove(potential_words[index][-1])
                    potential_words.pop(index)

            # Found every word! Done
            if len(incorrect_words) == 0:
                return found

            i += 1

        return found

    def get_index(self, arr, target):

        for words in enumerate(arr):

            if target in words[1]:
                return words[0]

        return -1