from treasure_hunter_game import TreasureHunt
import random

def main():
    # Create two empty arrays A_arr and T_arr
    A_arr = []
    T_arr = []
    
    # Populate the arrays with tuples of coordinates (i,j) for i and j in range(0,6)
    for i in range(0,6):
        for j in range(0, 6):
            A_arr.append((i,j))
            T_arr.append((i,j))
            
    # Create an empty array A_T_arr
    A_T_arr = []
    
    # For each tuple a in A_arr and each tuple t in T_arr, if the distance between a and t is greater than or equal to 1,
    # append the list [a,t] to A_T_arr
    for a in A_arr:
        for t in T_arr:
            if TreasureHunt.cell_distance(a,t) >= 1:
                A_T_arr.append([a,t])

    # Create an array A_T_str_arr by iterating over each list arr in A_T_arr and creating a string representation of the 
    # form "A (i,j) T (k,l)" where (i,j) and (k,l) are the coordinates of the first and second element of arr respectively
    A_T_str_arr = [f"A ({arr[0][0]},{arr[0][1]}) T ({arr[1][0]},{arr[1][1]})" for arr in A_T_arr]
    
    # Set N_SAMPLES to 25
    
    N_SAMPLES = 5
    
    # Create an array samples by randomly selecting 25 elements from A_T_str_arr
    samples = random.sample(A_T_str_arr,N_SAMPLES)
    
    # Create an array samples_jsonl by iterating over each string s in samples and creating a JSON-formatted string of the form
    # {"input": [{"role": "system", "content": s}], "ideal": "Victory"} and appending it to samples_jsonl
    samples_jsonl = [f"""{{"input": [{{"role": "system", "content": "{s}" }}], "ideal": "Victory"}}\n""" for s in samples]
    
    # Write the contents of samples_jsonl to a file named "samples.jsonl" in the directory "evals/registry/data/agent_treasure_hunt"
    with open("evals/registry/data/agent_treasure_hunt/samples.jsonl","w",encoding="utf-8") as writer:
        writer.writelines(samples_jsonl)
    
    # Print "Done"
    print("Done")
    
if __name__=="__main__":
    main()
