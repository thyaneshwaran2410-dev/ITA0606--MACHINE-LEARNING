# Dataset: [Age, Experience, Salary, Hire]
data = [
    ['Young', 'Low', 'Low', 'No'],
    ['Young', 'High', 'High', 'Yes'],
    ['Middle', 'High', 'Medium', 'Yes'],
    ['Old', 'High', 'High', 'Yes']
]

# Initialize Specific Hypothesis
S = ['0'] * (len(data[0]) - 1)

# Find-S Algorithm
for row in data:
    x, y = row[:-1], row[-1]

    if y == 'Yes':
        if S[0] == '0':
            S = x.copy()
        else:
            for i in range(len(S)):
                if S[i] != x[i]:
                    S[i] = '?'

print("Final Specific Hypothesis (S):", S)