# %% [markdown]
# # EX 1.1- basic python: Pyramid case
# Implement a function that get a string input and outputs the same word, only each odd char is lower
# case and each even letter is upper case
# You can assume that the input is a valid string which contains only english letters.
# %%


def pyramid_case(in_word):
    # TODO: return the pyramid case word.
    # WRITE IN 10 CODE LINES MAX!!!
    result = ""
    for i, char in enumerate(in_word):
        if i % 2 == 0:
            result += char.lower()
        else:
            result += char.upper()
    return result

# %%


def pyramid_case_one_liner(in_word):
    # TODO: ~~~BONUS~~~
    # return the pyramid case word in one line of code inside the function.
    # DO NOT USE ";" IN YOUR CODE.
    # - Loop through each character with its index using enumerate
    # - Convert to lowercase if the index is even, uppercase if odd
    # - Join all modified characters into a single string and return it
    return "".join(char.lower() if i % 2 == 0 else char.upper() for i, char in enumerate(in_word))


# %%
# test functions here
input_words = ["hello", "world", "", "I", "am", "LEARNING", "Python"]

print("==== pyramid_case() results:")
for word in input_words:
    print(pyramid_case(word))

print("\n==== pyramid_case_one_liner() results:")
for word in input_words:
    print(pyramid_case_one_liner(word))

# %%
