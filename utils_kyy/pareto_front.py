import numpy as np

# [Reference - 사용법] https://pythonhealthcare.org/tag/pareto-front/
# Scores (np array)를 input으로 받아서, 해당 scores 에서의 파레토 프론티어에 해당하는 점들의 인덱스를 반환해주는 함수
# e.g. scores = (300, 2) np array
#       => 점이 300개 & 최대화해야하는 objective 가 2개
#    output = [1, 3, 5, 150, 199]
#       => scores[1], scores[3], ..., scores[199] 가 파레토 프론티어임
def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]